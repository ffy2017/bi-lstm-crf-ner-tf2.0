# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/12/3 1:51 下午
# @Author: wuchenglong


import tensorflow as tf
from model import NerModel
from utils import tokenize_pred,read_vocab,format_result
import tensorflow_addons as tf_ad
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser(description="train")
parser.add_argument("--vocab_file", type=str, default="./data/vocab.txt",help="vocab_file")
parser.add_argument("--tag_file", type=str, default="./data/tags.txt",help="tag_file")
parser.add_argument("--test_file", type=str, default="./data/test.txt",help="test_file")
parser.add_argument("--hidden_num", type=int, default=64,help="lstm output dim")
parser.add_argument("--embedding_size", type=int, default=32,help="embedding dim")
parser.add_argument("--output_dir", type=str, default='./checkpoint',help="output dir")
parser.add_argument("--lr", type=float, default=1e-3,help="lr")
parser.add_argument("--batch_size", type=int, default=64,help="lr")
args = parser.parse_args()

gpus=tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(devices=gpus[2], device_type='GPU')
vocab2id, id2vocab = read_vocab(args.vocab_file)
tag2id, id2tag = read_vocab(args.tag_file)
text_sequences, text_lens ,label_sequences= tokenize_pred(args.test_file,vocab2id,tag2id)
train_dataset = tf.data.Dataset.from_tensor_slices((text_sequences, text_lens, label_sequences))
train_dataset = train_dataset.shuffle(len(text_sequences)).batch(args.batch_size, drop_remainder=True)

optimizer = tf.keras.optimizers.Adam(args.lr)
model = NerModel(hidden_num = args.hidden_num, vocab_size =len(vocab2id), label_size = len(tag2id), embedding_size = args.embedding_size)
# restore model
ckpt = tf.train.Checkpoint(optimizer=optimizer,model=model)
ckpt.restore(tf.train.latest_checkpoint(args.output_dir))

for text_batch, text_lens,labels_batch in train_dataset:
    logits, _ = model.predict(text_batch)
    paths = []
    for logit, text_len, labels in zip(logits, text_lens, labels_batch):
        viterbi_path, _ = tf_ad.text.viterbi_decode(logit[:text_len], model.transition_params)
        paths.append(viterbi_path)

    for i in range(len(text_batch)):
        res = {'text':[],'pred':[],'label':[]}
        for j,t in enumerate(paths[i]):
            res['text'].append(id2vocab.get(text_batch[i][j].numpy(),'<UKN>'))
            res['label'].append(id2tag[(labels_batch[i][j]).numpy()])
            res['pred'].append(id2tag[t])
        print(json.dumps(res, ensure_ascii=False))
