# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/12/3 7:28 下午
# @Author: wuchenglong


from utils import tokenize,build_vocab,read_vocab,tokenize_pred
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tf_ad
import os
import numpy as np
import argparse
import logging
from model import NerModel



logging.basicConfig(level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S')
parser = argparse.ArgumentParser(description="train")
parser.add_argument("--train_path", type=str, default="./data/train.txt",help="train file")
parser.add_argument("--eva_path", type=str, default="./data/eva.txt",help="eva file")
parser.add_argument("--test_path", type=str, default="./data/test.txt",help="test file")
parser.add_argument("--vocab_file", type=str, default="./data/vocab.txt",help="vocab_file")
parser.add_argument("--tag_file", type=str, default="./data/tags.txt",help="tag_file")
parser.add_argument("--hidden_num", type=int, default=64,help="lstm output dim")
parser.add_argument("--embedding_size", type=int, default=32,help="embedding dim")
parser.add_argument("--lr", type=float, default=1e-3,help="lr")
parser.add_argument("--output_dir", type=str, default='./checkpoint',help="output dir")
parser.add_argument("--epoch", type=int, default=10,help="epoch")
parser.add_argument("--batch_size", type=int, default=32,help="batch size")
parser.add_argument("--gpu_num", type=int, default=2,help="batch size")
args = parser.parse_args()
if not (os.path.exists(args.vocab_file) and os.path.exists(args.tag_file)):
    logging.info("building vocab file")
    build_vocab([args.train_path], args.vocab_file, args.tag_file)
else:
    logging.info("vocab file exits!!")

gpus=tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(devices=gpus[args.gpu_num], device_type='GPU')
vocab2id, id2vocab = read_vocab(args.vocab_file)
tag2id, id2tag = read_vocab(args.tag_file)
text_sequences ,label_sequences= tokenize(args.train_path,vocab2id,tag2id)
train_dataset = tf.data.Dataset.from_tensor_slices((text_sequences, label_sequences))
train_dataset = train_dataset.shuffle(len(text_sequences)).batch(args.batch_size, drop_remainder=True)

eva_text_sequences , text_lens, eva_label_sequences= tokenize_pred(args.eva_path,vocab2id,tag2id)
eva_dataset = tf.data.Dataset.from_tensor_slices((eva_text_sequences, text_lens, eva_label_sequences))
eva_dataset = eva_dataset.shuffle(args.batch_size*100).batch(args.batch_size, drop_remainder=True).repeat(None)

model = NerModel(hidden_num = args.hidden_num, vocab_size = len(vocab2id), label_size= len(tag2id), embedding_size = args.embedding_size)

optimizer = tf.keras.optimizers.Adam(args.lr)


ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
ckpt.restore(tf.train.latest_checkpoint(args.output_dir))
ckpt_manager = tf.train.CheckpointManager(ckpt,
                                          args.output_dir,
                                          checkpoint_name='model.ckpt',
                                          max_to_keep=3)

def train_one_step(text_batch, labels_batch):
  with tf.GradientTape() as tape:
      logits, text_lens, log_likelihood = model(text_batch, labels_batch,training=True)
      loss = - tf.reduce_mean(log_likelihood)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss,logits, text_lens

def get_acc_one_step(text_batch, text_lens, labels_batch):
    paths = []
    accuracy = 0
    logits, _, _ = model(text_batch, labels_batch,training=False)
    for logit, text_len, labels in zip(logits, text_lens, labels_batch):
        viterbi_path, _ = tf_ad.text.viterbi_decode(logit[:text_len], model.transition_params)
        paths.append(viterbi_path)
        correct_prediction = tf.equal(
            tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([viterbi_path], padding='post'),
                                 dtype=tf.int32),
            tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([labels[:text_len]], padding='post'),
                                 dtype=tf.int32)
        )
        accuracy = accuracy + tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy = accuracy / len(paths)
    return accuracy

best_acc = 0
step = 0

for epoch in range(args.epoch):
    for _, (text_batch, labels_batch) in enumerate(train_dataset):
        step = step + 1
        loss, logits, text_lens = train_one_step(text_batch, labels_batch)
        logging.info('epoch %d, step %d, loss %.4f' % (epoch, step, loss))
        if step % 500 == 0:
            accuracy = 0
            i = 0
            while i < 10:
                i += 1
                text_batch, text_lens, labels_batch = next(iter(eva_dataset))
                accuracy +=  get_acc_one_step(text_batch, text_lens, labels_batch)
            accuracy = accuracy/10
            logging.info('epoch %d, step %d, accuracy %.4f' % (epoch, step, accuracy))
            if accuracy > best_acc:
              best_acc = accuracy
              ckpt_manager.save()
              logging.info("model saved")


logging.info("finished")
