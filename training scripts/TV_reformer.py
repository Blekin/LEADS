# -*- coding: utf-8 -*-

import os, sys, random, math
import numpy as np
import torch
from transformers import ReformerModelWithLMHead

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# Adjustable parameters (modify as needed)
REFORMER_BATCH = 32       # Reformer inference batch, VERY IMPORTANT: keep small, e.g., 1,2,4,8
TRAIN_BATCH = 16384        # Keras training batch (generator output size)
EPOCHS = 50
WINDOW = 100             # sliding window length
REFORMER_FP16 = True     # whether to use fp16 inference for Reformer (if GPU supports it)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, file=sys.stderr)

# Optional: set this environment variable to reduce fragmentation (enable if experiencing fragmentation issues)
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Load Reformer (local)
reformer_model_path = "./reformer"
print("Loading Reformer from", reformer_model_path, file=sys.stderr)
model = ReformerModelWithLMHead.from_pretrained(reformer_model_path)
model.to(device)
model.eval()

# Convert model parameters to half precision to save memory (only during inference)
if REFORMER_FP16 and device.type == 'cuda':
    try:
        model.half()  # convert parameters to fp16
        print("Model converted to half precision (fp16).", file=sys.stderr)
    except Exception as e:
        print("Warning: model.half() failed:", e, file=sys.stderr)

# Read text and create bytes sliding window
with open('tv_p.txt', 'r', encoding='utf-8', errors='ignore') as f:
    all_text = f.read()

all_bytes = all_text.encode('utf-8', errors='ignore')
samples = []
targets = []
for i in range(0, len(all_bytes) - WINDOW):
    samples.append(all_bytes[i:i+WINDOW])
    targets.append(int(all_bytes[i+WINDOW]))
num_samples = len(samples)
print("num_samples:", num_samples, file=sys.stderr)

# Encoding function: bytes -> input_ids & attention_mask (PyTorch tensors on device)
def encode(list_of_byte_strings, pad_token_id=0, device=device):
    max_length = max(len(s) for s in list_of_byte_strings)
    batch_size = len(list_of_byte_strings)
    attention_masks = torch.zeros((batch_size, max_length), dtype=torch.long, device=device)
    input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long, device=device)
    for idx, bs in enumerate(list_of_byte_strings):
        L = len(bs)
        ids = [b + 2 for b in bs]
        input_ids[idx, :L] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_masks[idx, :L] = 1
    return input_ids, attention_masks

# Compute NUM_CLASSES automatically (based on max target value)
max_target = max(targets) if targets else 255
NUM_CLASSES = int(max_target) + 3
print("NUM_CLASSES =", NUM_CLASSES, file=sys.stderr)

# Probe feature_dim: use a small batch to get last_hidden_state dimension
with torch.no_grad(), torch.inference_mode():
    tmp_ids, tmp_mask = encode([samples[0]], device=device)
    out = model(input_ids=tmp_ids, attention_mask=tmp_mask)
    # Try last_hidden_state first; fall back to outputs[0] if not available
    if hasattr(out, 'last_hidden_state'):
        sample_hidden = out.last_hidden_state  # (1, seq_len, hidden_dim)
    else:
        sample_hidden = out[0]
    feature_dim = sample_hidden.shape[-1]
print("Feature dim (hidden size) =", feature_dim, file=sys.stderr)

# TensorFlow/Keras small network (using sparse labels)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

config=tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
#sess = tf.Session(config=config)
sess=tf.compat.v1.Session(config=config)
#K.set_session(sess)
tf.compat.v1.keras.backend.set_session(sess)

keras.mixed_precision.set_global_policy('mixed_float16')

def build_small_network(input_dim, num_classes):
    net = models.Sequential([
        layers.InputLayer(input_shape=(input_dim,)),
        #layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        #layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return net

network = build_small_network(feature_dim, NUM_CLASSES)
network.summary()

optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-3)
# Use sparse_categorical_crossentropy, labels are integer indices (not one-hot)
network.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Generator: extract features from Reformer on-the-fly per batch
def train_data_generator(train_batch):
    num = num_samples
    indices = list(range(num))
    while True:
        random.shuffle(indices)
        for start in range(0, num, train_batch):
            batch_idx = indices[start:start+train_batch]
            if not batch_idx:
                continue
            batch_samples = [samples[i] for i in batch_idx]
            batch_targets = [targets[i] for i in batch_idx]

            # To reduce memory pressure, split batch into smaller chunks, or simply keep REFORMER_BATCH small
            # Here we use REFORMER_BATCH internally for inference splitting
            feat_chunks = []
            for j in range(0, len(batch_samples), REFORMER_BATCH):
                sub = batch_samples[j:j+REFORMER_BATCH]
                # Clear cache (optional)
                torch.cuda.empty_cache()
                # Inference: use autocast + inference_mode (fp16 support)
                if device.type == 'cuda' and REFORMER_FP16:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        with torch.inference_mode():
                            ids, mask = encode(sub, device=device)
                            out = model(input_ids=ids, attention_mask=mask)
                else:
                    with torch.inference_mode():
                        ids, mask = encode(sub, device=device)
                        out = model(input_ids=ids, attention_mask=mask)
                # Take last_hidden_state at the last position as features
                if hasattr(out, 'last_hidden_state'):
                    last_hidden = out.last_hidden_state[:, -1, :].cpu().numpy()
                else:
                    last_hidden = out[0][:, -1, :].cpu().numpy()
                feat_chunks.append(last_hidden.astype('float32'))

            X_batch = np.vstack(feat_chunks)  # shape (len(batch_idx), feature_dim)
            y_batch = np.array(batch_targets, dtype='int32')  # sparse labels
            yield X_batch, y_batch

# Training parameters and start
TRAIN_BATCH = min(TRAIN_BATCH, num_samples)
STEPS_PER_EPOCH = max(1, num_samples // TRAIN_BATCH)
print("TRAIN_BATCH:", TRAIN_BATCH, "REFORMER_BATCH:", REFORMER_BATCH, "STEPS_PER_EPOCH:", STEPS_PER_EPOCH, file=sys.stderr)

callbacks = [
    keras.callbacks.EarlyStopping(monitor='loss', patience=5),
    keras.callbacks.ModelCheckpoint('model_tv_reformer.h5', monitor='loss', save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.33, patience=2)
]

gen = train_data_generator(TRAIN_BATCH)
network.fit(gen, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, callbacks=callbacks, verbose=2)

print("Done", file=sys.stderr)