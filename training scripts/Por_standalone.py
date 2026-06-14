# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import keras_nlp
import keras
import tensorflow
import numpy as np
import pandas as pd

with open('por.txt',encoding='utf-8') as f:
  text = f.read()

import tensorflow as tf
from keras import layers, models
import keras_nlp
import keras.backend as K

config=tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.compat.v1.Session(config=config) 
tf.compat.v1.keras.backend.set_session(sess)

keras.mixed_precision.set_global_policy('mixed_float16')

#setting hyperparameters
maxlen=64
step=1
sentences = []
next_chars = []
batch_size=1024

#construct sample and target
for i in range(0,len(text)-maxlen,step):
    sentences.append(text[i:i+maxlen])
    next_chars.append(text[i+maxlen])

#construct the dict
chars=sorted(list(set(text)))
char_indices=dict((char,chars.index(char)) for char in chars)

key_list = []
value_list = []
for key,value in char_indices.items():
    key_list.append(key)
    value_list.append(value)


import tensorflow_model_optimization as tfmot
from keras_nlp.layers.modeling.transformer_layer_utils import (  # isort:skip
    merge_padding_and_attention_mask,
)


class MyTransformerEncoder(keras.layers.Layer, tfmot.sparsity.keras.PrunableLayer, tfmot.clustering.keras.ClusterableLayer):

    def __init__(
        self,
        intermediate_dim,
        num_heads,
        dropout=0,
        activation="relu",
        layer_norm_epsilon=1e-05,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        name=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = keras.activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self._built = False
        self.supports_masking = True

    def _build(self, input_shape):
        # Create layers based on input shape.
        self._built = True
        feature_size = input_shape[-1]
        self._attention_head_size = int(feature_size // self.num_heads)
        self._multi_head_attention_layer = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self._attention_head_size,
            value_dim=self._attention_head_size,
            dropout=self.dropout,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )

        self._attention_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
        )
        self._feedforward_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
        )

        self._attention_dropout = keras.layers.Dropout(rate=self.dropout)

        self._intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )
        self._output_dense = keras.layers.Dense(
            feature_size,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )
        self._output_dropout = keras.layers.Dropout(rate=self.dropout)

    def _add_and_norm(self, input1, input2, norm_layer):
        return norm_layer(input1 + input2)

    def _feed_forward(self, input):
        x = self._intermediate_dense(input)
        x = self._output_dense(x)
        return self._output_dropout(x)

    def get_prunable_weights(self):
        return self.weights

    def get_clusterable_weights(self):
        self.kernel = self.weights[0]
        return [['kernel',self.weights]]
        #return self.name, self.weights

    def call(self, inputs, padding_mask=None, attention_mask=None):

        if not self._built:
            self._build(inputs.shape)

        mask = merge_padding_and_attention_mask(
            inputs,
            padding_mask,
            attention_mask,
        )

        # Self attention.
        attended = self._multi_head_attention_layer(
            inputs, inputs, inputs, attention_mask=mask
        )
        attended = self._attention_dropout(attended)
        attended = self._add_and_norm(
            inputs,
            attended,
            self._attention_layernorm,
        )
        # Feedforward.
        feed_forward_output = self._feed_forward(attended)
        return self._add_and_norm(
            attended, feed_forward_output, self._feedforward_layernorm
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "activation": keras.activations.serialize(self.activation),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
            }
        )
        return config

#build the network
n_feature = maxlen
sigma = len(chars)
n_window = 10

inputs = keras.Input(shape=(n_feature, sigma,))

gru = layers.GRU(64)

out_list = []

for i in range(n_feature - n_window + 1):
    window = inputs[:, i : i + n_window]
    gru_out = gru(window)
    out_list.append(gru_out)

rnn_out = K.stack(out_list, axis = 1)

residual = inputs[:, (n_window-1):]
residual = layers.Conv1D(64, 1, strides=1, activation=None)(residual)

added_out = layers.add([rnn_out, residual])

normalized = layers.LayerNormalization(epsilon=1e-5)(added_out)
outputs = layers.Dropout(rate=0.1)(normalized)

for i in range(1):
    outputs = MyTransformerEncoder(
        intermediate_dim=64,
        num_heads=4,
        dropout=0.1,
        layer_norm_epsilon=1e-5,
    )(outputs)

outputs = layers.add([outputs, added_out])
outputs = layers.LayerNormalization(epsilon=1e-5)(outputs)

outputs = layers.GRU(64)(outputs)

outputs = layers.Dense(128)(outputs)
outputs = layers.Dense(sigma, activation="softmax")(outputs)

model = keras.Model(inputs, outputs)
model.summary()

import random

def train_data(train_batch):
  for p in range(train_batch):
    num_batch = len(sentences) // batch_size
    # shuffle
    lt = list(range(num_batch))
    for n in range(num_batch):
      q = random.choice(lt)
      lt.remove(q)
      train_datas = sentences[batch_size*q:batch_size*(q+1)]
      train_lables = next_chars[batch_size*q:batch_size*(q+1)]
      x = np.zeros((batch_size, maxlen, len(chars)), dtype=bool)
      y = np.zeros((batch_size, len(chars)), dtype=bool)
      for i, sentence in enumerate(train_datas):
        y[i, char_indices[train_lables[i]]]=1
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]]=1
      yield x,y

#set callbacks
callbacks_list = [
        keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=10,),
        keras.callbacks.ModelCheckpoint(
                filepath='model_por_small.h5',
                monitor='loss',
                save_best_only=True,),
        keras.callbacks.CSVLogger(
                './log_por_small.csv',
                separator=','),
        keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.33,
                patience=3,
                )]

#compile
from tensorflow import optimizers
optimizer=optimizers.RMSprop(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])

#fitting
history=model.fit(train_data(100), steps_per_epoch=len(sentences) // batch_size, shuffle=True, batch_size=1024, epochs=100, callbacks=callbacks_list[1:])

