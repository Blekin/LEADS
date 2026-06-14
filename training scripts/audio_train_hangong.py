# -*- coding: utf-8 -*-

import os
import json
import math
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import soundfile as sf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision


# Config
mixed_precision.set_global_policy("mixed_float16")

AUDIO_PATH = "hangong.flac"
WORKDIR = "./exp_joint_topk_residual_hangong"
Path(WORKDIR).mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 48      # if OOM, change to 32
TOTAL_EPOCHS = 100
SEQ_LEN = 384
STRIDE = 32

# 1-based epoch schedule. At the beginning of these epochs, LR is lowered.
STAGE_LR_SCHEDULE = {
    1: 5e-4,
    20: 2e-4,
    40: 1e-4,
    60: 5e-5,
    80: 2e-5,
}

# Adaptive LR reduction on training loss plateau.
ADAPTIVE_PATIENCE = 3
LR_FACTOR = 0.5
MIN_LR = 1e-5
MIN_DELTA = 1e-4

EARLY_STOP_AT_MIN_LR = True
EARLY_STOP_PATIENCE_AT_MIN_LR = 6

TOPK_MID = 4096
TOPK_SIDE = 2048

TOKEN_EMBED = 48
STREAM_EMBED = 8
MODEL_DIM = 96
NUM_HEADS = 4
FF_DIM = 256
GRU_DIM = 96
DROPOUT = 0.1

RUN_TAG = f"unified_lr_sched_stride{STRIDE}_b{BATCH_SIZE}"


# Utils
def get_lr(optimizer) -> float:
    lr = optimizer.learning_rate
    try:
        return float(tf.keras.backend.get_value(lr))
    except Exception:
        return float(lr.numpy())


def set_lr(optimizer, value: float):
    try:
        tf.keras.backend.set_value(optimizer.learning_rate, value)
    except Exception:
        optimizer.learning_rate.assign(value)


def zigzag_encode_int32(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.int64)
    return ((x << 1) ^ (x >> 63)).astype(np.uint64)


def rice_bits(u: np.ndarray, k: int) -> np.ndarray:
    q = u >> k
    return q + 1 + k


def choose_best_rice_k(u: np.ndarray) -> int:
    if len(u) == 0:
        return 0
    best_k, best_mean = 0, float("inf")
    for k in range(8):
        m = rice_bits(u, k).mean()
        if m < best_mean:
            best_mean = m
            best_k = k
    return best_k


def save_quantized_int8_weights(model: keras.Model, out_npz: str) -> int:
    arrays = {}
    for i, w in enumerate(model.get_weights()):
        max_abs = float(np.max(np.abs(w))) if w.size else 0.0
        scale = max(max_abs / 127.0, 1e-8)
        q = np.round(w / scale).astype(np.int8)
        arrays[f"arr_{i}_q"] = q
        arrays[f"arr_{i}_scale"] = np.array([scale], dtype=np.float32)
        arrays[f"arr_{i}_shape"] = np.array(w.shape, dtype=np.int32)
    np.savez_compressed(out_npz, **arrays)
    return os.path.getsize(out_npz)


# Audio preprocessing
def load_audio_int16(audio_path: str):
    audio, sr = sf.read(audio_path, dtype="int16", always_2d=True)
    audio = audio.astype(np.int32)
    L = audio[:, 0]
    R = audio[:, 1]
    return L, R, sr


def mid_side_transform(L: np.ndarray, R: np.ndarray):
    mid = ((L + R) // 2).astype(np.int32)
    side = (L - R).astype(np.int32)
    return mid, side


def second_order_residual(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.int32)
    r = np.empty_like(x, dtype=np.int32)
    r[0] = x[0]
    if len(x) > 1:
        r[1] = x[1] - x[0]
    if len(x) > 2:
        r[2:] = x[2:] - 2 * x[1:-1] + x[:-2]
    return r


# Frequent-value vocab + ESC
class TokenizerWithEscape:
    def __init__(self, topk: int):
        self.topk = int(topk)
        self.esc_token = self.topk
        self.vocab_size = self.topk + 1
        self.value_to_token = {}
        self.token_to_value = None
        self.rice_k_tail = 0

    def fit(self, arr: np.ndarray):
        vals, counts = np.unique(arr.astype(np.int32), return_counts=True)
        order = np.argsort(counts)[::-1]
        vals = vals[order]
        keep = vals[: self.topk]
        self.value_to_token = {int(v): i for i, v in enumerate(keep.tolist())}
        self.token_to_value = np.array(keep, dtype=np.int32)

        mask = np.array([int(v) not in self.value_to_token for v in arr.tolist()], dtype=bool)
        tail = arr[mask]
        if len(tail):
            u = zigzag_encode_int32(tail)
            self.rice_k_tail = choose_best_rice_k(u)
        else:
            self.rice_k_tail = 0

    def encode(self, arr: np.ndarray):
        arr = arr.astype(np.int32)
        tokens = np.empty(len(arr), dtype=np.int32)
        tail_values = []
        tail_positions = []
        for i, v in enumerate(arr.tolist()):
            tok = self.value_to_token.get(int(v), self.esc_token)
            tokens[i] = tok
            if tok == self.esc_token:
                tail_positions.append(i)
                tail_values.append(int(v))
        return tokens, np.array(tail_positions, dtype=np.int64), np.array(tail_values, dtype=np.int32)

    def estimate_tail_bits(self, tail_values: np.ndarray) -> int:
        if len(tail_values) == 0:
            return 0
        u = zigzag_encode_int32(tail_values)
        return int(np.sum(rice_bits(u, self.rice_k_tail)))

    def dump(self):
        return {
            "topk": self.topk,
            "esc_token": self.esc_token,
            "vocab_size": self.vocab_size,
            "rice_k_tail": self.rice_k_tail,
            "token_to_value": self.token_to_value.tolist(),
        }


# Build streams
def prepare_streams(audio_path: str):
    L, R, sr = load_audio_int16(audio_path)
    mid, side = mid_side_transform(L, R)

    mid_r = second_order_residual(mid)
    side_r = second_order_residual(side)

    tok_mid = TokenizerWithEscape(TOPK_MID)
    tok_side = TokenizerWithEscape(TOPK_SIDE)
    tok_mid.fit(mid_r)
    tok_side.fit(side_r)

    mid_tokens, _, mid_tail_values = tok_mid.encode(mid_r)
    side_tokens, _, side_tail_values = tok_side.encode(side_r)

    stats = {
        "num_samples": int(len(L)),
        "sr": int(sr),
        "pcm_bytes": int(len(L) * 2 * 2),
        "input_bytes": int(os.path.getsize(audio_path)),
        "mid_unique": int(len(np.unique(mid_r))),
        "side_unique": int(len(np.unique(side_r))),
        "mid_esc_frac": float(np.mean(mid_tokens == tok_mid.esc_token)),
        "side_esc_frac": float(np.mean(side_tokens == tok_side.esc_token)),
        "mid_tail_rice_k": int(tok_mid.rice_k_tail),
        "side_tail_rice_k": int(tok_side.rice_k_tail),
        "mid_tail_bits_empirical": int(tok_mid.estimate_tail_bits(mid_tail_values)),
        "side_tail_bits_empirical": int(tok_side.estimate_tail_bits(side_tail_values)),
    }

    return {
        "sr": sr,
        "num_samples": len(L),
        "pcm_bytes": stats["pcm_bytes"],
        "input_bytes": stats["input_bytes"],
        "mid_tokens": mid_tokens,
        "side_tokens": side_tokens,
        "tok_mid": tok_mid,
        "tok_side": tok_side,
        "stats": stats,
        "max_vocab_size": max(tok_mid.vocab_size, tok_side.vocab_size),
    }


# Sequence
class SharedTokenSequence(tf.keras.utils.Sequence):
    def __init__(self, mid_tokens, side_tokens, seq_len, stride, batch_size, shuffle=True):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle

        starts_mid = np.arange(0, len(mid_tokens) - seq_len - 1, stride, dtype=np.int64)
        starts_side = np.arange(0, len(side_tokens) - seq_len - 1, stride, dtype=np.int64)

        items = []
        for s in starts_mid:
            items.append((0, int(s)))
        for s in starts_side:
            items.append((1, int(s)))
        self.items = np.array(items, dtype=np.int64)

        self.mid_tokens = mid_tokens.astype(np.int32)
        self.side_tokens = side_tokens.astype(np.int32)
        self.indexes = np.arange(len(self.items))
        self.on_epoch_end()

    def __len__(self):
        return len(self.indexes) // self.batch_size

    def __getitem__(self, idx):
        batch_ids = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        rows = self.items[batch_ids]

        x_ctx = np.empty((self.batch_size, self.seq_len), dtype=np.int32)
        x_stream = np.empty((self.batch_size, 1), dtype=np.int32)
        y = np.empty((self.batch_size,), dtype=np.int32)

        for i, (stream_id, s) in enumerate(rows):
            arr = self.mid_tokens if stream_id == 0 else self.side_tokens
            x_ctx[i] = arr[s : s + self.seq_len]
            y[i] = arr[s + self.seq_len]
            x_stream[i, 0] = stream_id
        return {"token_context": x_ctx, "stream_id": x_stream}, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


# Model
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim):
        super().__init__()
        self.token_embeddings = layers.Embedding(vocab_size, embed_dim)
        self.position_embeddings = layers.Embedding(sequence_length, embed_dim)

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        x = self.token_embeddings(inputs)
        pos = self.position_embeddings(positions)
        return x + pos


class TransformerBlock(layers.Layer):
    def __init__(self, model_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=model_dim // num_heads,
            value_dim=model_dim // num_heads,
            dropout=dropout,
        )
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(model_dim),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=None):
        attn = self.attn(x, x, x, use_causal_mask=True, training=training)
        x = self.norm1(x + self.drop1(attn, training=training))
        ff = self.ffn(x)
        return self.norm2(x + self.drop2(ff, training=training))


def compile_trainer(model: keras.Model, lr: float):
    opt = keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    model.compile(
        optimizer=opt,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["sparse_categorical_accuracy"],
    )


def build_model(max_vocab_size: int, compile_model: bool = True, lr: float = 5e-4):
    token_context = keras.Input(shape=(SEQ_LEN,), dtype="int32", name="token_context")
    stream_id = keras.Input(shape=(1,), dtype="int32", name="stream_id")

    tok_emb = PositionalEmbedding(SEQ_LEN, max_vocab_size, TOKEN_EMBED)(token_context)
    sid = layers.Embedding(2, STREAM_EMBED)(stream_id)
    sid = layers.Reshape((STREAM_EMBED,))(sid)
    sid = layers.RepeatVector(SEQ_LEN)(sid)

    x = layers.Concatenate(axis=-1)([tok_emb, sid])
    x = layers.Dense(MODEL_DIM, activation="gelu")(x)
    x = TransformerBlock(MODEL_DIM, NUM_HEADS, FF_DIM, DROPOUT)(x)
    x = TransformerBlock(MODEL_DIM, NUM_HEADS, FF_DIM, DROPOUT)(x)
    x = layers.GRU(GRU_DIM)(x)
    x = layers.Dense(MODEL_DIM, activation="gelu")(x)
    x = layers.Dropout(DROPOUT)(x)
    logits = layers.Dense(max_vocab_size, name="token_logits")(x)

    model = keras.Model(inputs={"token_context": token_context, "stream_id": stream_id}, outputs=logits)

    if compile_model:
        compile_trainer(model, lr)
    return model


# Custom LR callback
class StageAndPlateauLR(keras.callbacks.Callback):
    def __init__(self, stage_schedule, monitor="loss", min_delta=1e-4, patience=3,
                 factor=0.5, min_lr=1e-5, early_stop_at_min_lr=True,
                 early_stop_patience_at_min_lr=6):
        super().__init__()
        self.stage_schedule = dict(sorted(stage_schedule.items()))
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.early_stop_at_min_lr = early_stop_at_min_lr
        self.early_stop_patience_at_min_lr = early_stop_patience_at_min_lr
        self.best = float("inf")
        self.wait = 0
        self.wait_at_min_lr = 0
        self.lr_events = []

    def on_train_begin(self, logs=None):
        initial_lr = self.stage_schedule.get(1, get_lr(self.model.optimizer))
        set_lr(self.model.optimizer, initial_lr)
        print(f"[LR] train begin: lr={initial_lr:.8g}")

    def on_epoch_begin(self, epoch, logs=None):
        human_epoch = epoch + 1
        if human_epoch in self.stage_schedule:
            scheduled_lr = self.stage_schedule[human_epoch]
            current_lr = get_lr(self.model.optimizer)
            if scheduled_lr < current_lr:
                set_lr(self.model.optimizer, scheduled_lr)
                self.wait = 0
                print(f"[LR] scheduled drop at epoch {human_epoch}: {current_lr:.8g} -> {scheduled_lr:.8g}")
                self.lr_events.append({"epoch": human_epoch, "type": "scheduled", "old_lr": current_lr, "new_lr": scheduled_lr})

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        human_epoch = epoch + 1
        current = logs.get(self.monitor)
        if current is None:
            return
        current = float(current)
        current_lr = get_lr(self.model.optimizer)

        if current < self.best - self.min_delta:
            self.best = current
            self.wait = 0
            self.wait_at_min_lr = 0
            print(f"[LR] epoch {human_epoch}: {self.monitor} improved to {current:.6f}, lr={current_lr:.8g}")
            return

        self.wait += 1
        print(f"[LR] epoch {human_epoch}: no significant improvement ({self.monitor}={current:.6f}, best={self.best:.6f}), wait={self.wait}/{self.patience}, lr={current_lr:.8g}")

        if self.wait >= self.patience:
            new_lr = max(current_lr * self.factor, self.min_lr)
            if new_lr < current_lr:
                set_lr(self.model.optimizer, new_lr)
                print(f"[LR] plateau drop at epoch {human_epoch}: {current_lr:.8g} -> {new_lr:.8g}")
                self.lr_events.append({"epoch": human_epoch, "type": "plateau", "old_lr": current_lr, "new_lr": new_lr, "best": self.best, "current": current})
                self.wait = 0
            else:
                self.wait_at_min_lr += 1
                print(f"[LR] already at min_lr={self.min_lr:.8g}, wait_at_min_lr={self.wait_at_min_lr}")
                if self.early_stop_at_min_lr and self.wait_at_min_lr >= self.early_stop_patience_at_min_lr:
                    print("[LR] early stopping: loss stuck at min_lr")
                    self.model.stop_training = True


# Estimate helper
def estimate_from_loss(loss_nat: float, num_samples: int, tail_bits: int):
    avg_bits_per_token = loss_nat / math.log(2.0)
    total_token_bits_est = avg_bits_per_token * (2 * num_samples)
    total_payload_bits = total_token_bits_est + tail_bits
    return {
        "avg_bits_per_token": float(avg_bits_per_token),
        "token_bits_per_samplepair": float(total_token_bits_est / num_samples),
        "tail_bits_per_samplepair": float(tail_bits / num_samples),
        "total_bits_per_samplepair": float(total_payload_bits / num_samples),
        "payload_bytes_est": float(total_payload_bits / 8.0),
    }


# Main
def main():
    if not os.path.exists(AUDIO_PATH):
        raise FileNotFoundError(f"AUDIO_PATH not found: {AUDIO_PATH}")

    data = prepare_streams(AUDIO_PATH)
    stats = data["stats"]

    print("===== UNIFIED TRAIN DATA STATS =====")
    for k, v in stats.items():
        print(f"{k}: {v}")

    train_seq = SharedTokenSequence(
        data["mid_tokens"],
        data["side_tokens"],
        seq_len=SEQ_LEN,
        stride=STRIDE,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    print(f"\nSequence batches per epoch: {len(train_seq)}")
    print(f"Stage LR schedule: {STAGE_LR_SCHEDULE}")
    print(f"Adaptive patience: {ADAPTIVE_PATIENCE}, factor={LR_FACTOR}, min_lr={MIN_LR}")
    print(f"Stride: {STRIDE}")
    print(f"Batch size: {BATCH_SIZE}")

    initial_lr = STAGE_LR_SCHEDULE.get(1, 5e-4)
    model = build_model(data["max_vocab_size"], compile_model=True, lr=initial_lr)
    model.summary()

    best_ckpt = os.path.join(WORKDIR, f"{RUN_TAG}.best.weights.h5")
    final_ckpt = os.path.join(WORKDIR, f"{RUN_TAG}.final.weights.h5")
    csv_path = os.path.join(WORKDIR, f"{RUN_TAG}.train_log.csv")
    q_npz = os.path.join(WORKDIR, f"{RUN_TAG}.int8_weights.npz")
    meta_path = os.path.join(WORKDIR, f"{RUN_TAG}.metadata.json")

    lr_callback = StageAndPlateauLR(
        stage_schedule=STAGE_LR_SCHEDULE,
        monitor="loss",
        min_delta=MIN_DELTA,
        patience=ADAPTIVE_PATIENCE,
        factor=LR_FACTOR,
        min_lr=MIN_LR,
        early_stop_at_min_lr=EARLY_STOP_AT_MIN_LR,
        early_stop_patience_at_min_lr=EARLY_STOP_PATIENCE_AT_MIN_LR,
    )

    callbacks = [
        lr_callback,
        keras.callbacks.ModelCheckpoint(best_ckpt, monitor="loss", save_best_only=True, save_weights_only=True),
        keras.callbacks.CSVLogger(csv_path, append=False),
    ]

    history = model.fit(
        train_seq,
        epochs=TOTAL_EPOCHS,
        callbacks=callbacks,
        workers=1,
        use_multiprocessing=False,
        verbose=2,
    )

    model.save_weights(final_ckpt)
    model_bytes_est = save_quantized_int8_weights(model, q_npz)

    losses = [float(x) for x in history.history["loss"]]
    best_loss_nat = float(np.min(losses))
    final_loss_nat = float(losses[-1])
    tail_bits = stats["mid_tail_bits_empirical"] + stats["side_tail_bits_empirical"]

    best_est = estimate_from_loss(best_loss_nat, data["num_samples"], tail_bits)
    final_est = estimate_from_loss(final_loss_nat, data["num_samples"], tail_bits)

    meta = {
        "config": {
            "AUDIO_PATH": AUDIO_PATH,
            "WORKDIR": WORKDIR,
            "BATCH_SIZE": BATCH_SIZE,
            "TOTAL_EPOCHS": TOTAL_EPOCHS,
            "SEQ_LEN": SEQ_LEN,
            "STRIDE": STRIDE,
            "STAGE_LR_SCHEDULE": STAGE_LR_SCHEDULE,
            "ADAPTIVE_PATIENCE": ADAPTIVE_PATIENCE,
            "LR_FACTOR": LR_FACTOR,
            "MIN_LR": MIN_LR,
            "MIN_DELTA": MIN_DELTA,
            "TOPK_MID": TOPK_MID,
            "TOPK_SIDE": TOPK_SIDE,
            "TOKEN_EMBED": TOKEN_EMBED,
            "STREAM_EMBED": STREAM_EMBED,
            "MODEL_DIM": MODEL_DIM,
            "NUM_HEADS": NUM_HEADS,
            "FF_DIM": FF_DIM,
            "GRU_DIM": GRU_DIM,
            "DROPOUT": DROPOUT,
            "RUN_TAG": RUN_TAG,
        },
        "stats": stats,
        "history": {
            "loss": losses,
            "sparse_categorical_accuracy": [float(x) for x in history.history.get("sparse_categorical_accuracy", [])],
            "epochs_ran": len(losses),
            "best_loss_nat": best_loss_nat,
            "final_loss_nat": final_loss_nat,
            "final_lr": get_lr(model.optimizer),
            "lr_events": lr_callback.lr_events,
        },
        "estimate_best_loss": {
            **best_est,
            "model_bytes_est_int8": int(model_bytes_est),
            "container_total_bytes_est": float(best_est["payload_bytes_est"] + model_bytes_est),
            "input_flac_bytes": int(data["input_bytes"]),
            "pcm_bytes": int(data["pcm_bytes"]),
        },
        "estimate_final_loss": {
            **final_est,
            "model_bytes_est_int8": int(model_bytes_est),
            "container_total_bytes_est": float(final_est["payload_bytes_est"] + model_bytes_est),
            "input_flac_bytes": int(data["input_bytes"]),
            "pcm_bytes": int(data["pcm_bytes"]),
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n ******UNIFIED TRAIN ESTIMATE (BEST LOSS)******")
    print(f"best_loss_nat           : {best_loss_nat:.6f}")
    print(f"final_loss_nat          : {final_loss_nat:.6f}")
    print(f"avg_bits_per_token      : {best_est['avg_bits_per_token']:.6f}")
    print(f"token_bits/samplepair   : {best_est['token_bits_per_samplepair']:.6f}")
    print(f"tail_bits/samplepair    : {best_est['tail_bits_per_samplepair']:.6f}")
    print(f"total_bits/samplepair   : {best_est['total_bits_per_samplepair']:.6f}")
    print(f"payload_bytes_est       : {best_est['payload_bytes_est'] / (1024**2):.3f} MiB")
    print(f"model_bytes_est_int8    : {model_bytes_est / (1024**2):.3f} MiB")
    print(f"container_total_est     : {(best_est['payload_bytes_est'] + model_bytes_est) / (1024**2):.3f} MiB")
    print(f"input_flac_bytes        : {data['input_bytes'] / (1024**2):.3f} MiB")
    print(f"pcm_bytes               : {data['pcm_bytes'] / (1024**2):.3f} MiB")

    if best_est["payload_bytes_est"] + model_bytes_est < data["input_bytes"]:
        print(">>> estimate says: smaller than input FLAC")
    else:
        print(">>> estimate says: not yet smaller than input FLAC")

    print("\nLR events:")
    for ev in lr_callback.lr_events:
        print(ev)

    print("\nSaved:")
    print(best_ckpt)
    print(final_ckpt)
    print(csv_path)
    print(q_npz)
    print(meta_path)

if __name__ == "__main__":
    main()
