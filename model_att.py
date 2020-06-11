# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/12/3 7:15 下午
# @Author: ffy


import tensorflow as tf
import tensorflow_addons as tf_ad


class NerModel(tf.keras.Model):
    def __init__(self, hidden_num, vocab_size, label_size, embedding_size):
        super(NerModel, self).__init__()
        self.num_hidden = hidden_num
        self.vocab_size = vocab_size
        self.label_size = label_size
        self.transition_params = None
        attention_size = 2*hidden_num

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.biLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_num, return_sequences=True))
        self.dense = tf.keras.layers.Dense(label_size)

        self.transition_params = tf.Variable(tf.random.uniform(shape=(label_size, label_size)), name='params')
        self.dropout = tf.keras.layers.Dropout(0.5)

        self.w_omega = tf.Variable(tf.random.normal([2*hidden_num, attention_size], stddev=0.1), name='w_omega')
        self.b_omega = tf.Variable(tf.random.normal([attention_size], stddev=0.1), name='b_omega')
        self.u_omega = tf.Variable(tf.random.normal([attention_size], stddev=0.1), name='u_omega')

    # @tf.function
    def call(self, text,labels=None,training=None):
        #text_lens是mask
        text_lens = tf.math.reduce_sum(tf.cast(tf.math.not_equal(text, 0), dtype=tf.int32), axis=-1)
        # -1 change 0
        inputs = self.embedding(text)
        inputs = self.dropout(inputs, training)
        inputs = self.biLSTM(inputs)

        v = tf.tanh(tf.tensordot(inputs, self.w_omega, axes=1) + self.b_omega)
        vu = tf.tensordot(v, self.u_omega, axes=1, name='vu')
        alphas = tf.nn.softmax(vu, name='alphas')
        output = tf.multiply(inputs, tf.expand_dims(alphas, -1))
        output = tf.add(output,inputs)
        logits = self.dense(output)

        if labels is not None:
            label_sequences = tf.convert_to_tensor(labels, dtype=tf.int32)
            log_likelihood,_ = tf_ad.text.crf_log_likelihood(logits, label_sequences, text_lens, self.transition_params)
            return logits, text_lens, log_likelihood
        else:
            return logits, text_lens
