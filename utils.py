# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/12/3 7:30 下午
# @Author: wuchenglong


import tensorflow as tf
import json,os

def build_vocab(corpus_file_list, vocab_file, tag_file):
    words = set()
    tags = set()
    for file in corpus_file_list:
        for line in open(file, "r", encoding='utf-8').readlines():
            data = json.loads(line)
            for word in data['text']:
                words.add(word)
            for tag in data['tags']:
                tags.add(tag)

    if not os.path.exists(vocab_file):
        with open(vocab_file,"w") as f:
            for index,word in enumerate(["<UKN>"]+list(words) ):
                f.write(word+"\n")

    if not os.path.exists(tag_file):
        with open(tag_file,"w") as f:
            for index,tag in enumerate(["<UKN>"]+list(tags)):
                f.write(tag+"\n")

# build_vocab(["./data/train.utf8","./data/test.utf8"])


def read_vocab(vocab_file):
    vocab2id = {}
    id2vocab = {}
    with open(vocab_file,"r") as f:
        for index,line in enumerate(f):
            line = line[:-1]
            vocab2id[line] = index
            id2vocab[index] = line
    return vocab2id, id2vocab

# print(read_vocab("./data/tags.txt"))



def tokenize(filename,vocab2id,tag2id):
    contents = []
    labels = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            data = json.loads(line)
            content = [vocab2id.get(x, 0) for x in data['text']]
            label = [tag2id.get(x, 0) for x in data['tags']]
            contents.append(content)
            labels.append(label)

    contents = tf.keras.preprocessing.sequence.pad_sequences(contents, padding='post')
    labels = tf.keras.preprocessing.sequence.pad_sequences(labels, padding='post')
    return contents,labels

def tokenize_pred(filename,vocab2id,tag2id):
    contents = []
    labels = []
    text_lens = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            data = json.loads(line)
            content = [vocab2id.get(x, 0) for x in data['text']]
            label = [tag2id.get(x, 0) for x in data['tags']]
            contents.append(content)
            text_lens.append(len(data['text']))
            labels.append(label)

    contents = tf.keras.preprocessing.sequence.pad_sequences(contents, padding='post')
    labels = tf.keras.preprocessing.sequence.pad_sequences(labels, padding='post')
    return contents,text_lens,labels

tag_check = {
    "I":["B","I"],
    "E":["B","I"],
}


def check_label(front_label,follow_label):
    if not follow_label:
        raise Exception("follow label should not both None")

    if not front_label:
        return True

    if follow_label.startswith("B-"):
        return False

    if (follow_label.startswith("I-") or follow_label.startswith("E-")) and \
        front_label.endswith(follow_label.split("-")[1]) and \
        front_label.split("-")[0] in tag_check[follow_label.split("-")[0]]:
        return True
    return False


def format_result(chars, tags):
    entities = []
    entity = []
    for index, (char, tag) in enumerate(zip(chars, tags)):
        entity_continue = check_label(tags[index - 1] if index > 0 else None, tag)
        if not entity_continue and entity:
            entities.append(entity)
            entity = []
        entity.append([index, char, tag, entity_continue])
    if entity:
        entities.append(entity)

    entities_result = []
    for entity in entities:
        if entity[0][2].startswith("B-"):
            entities_result.append(
                {"begin": entity[0][0] + 1,
                 "end": entity[-1][0] + 1,
                 "words": "".join([char for _, char, _, _ in entity]),
                 "type": entity[0][2].split("-")[1]
                 }
            )

    return entities_result



if __name__ == "__main__":
    text = ['国','家','发','展','计','划','委','员','会','副','主','任','王','春','正']
    tags =  ['B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'E-ORG', 'O', 'O', 'O', 'B-PER', 'I-PER', 'E-PER']
    entities_result= format_result(text,tags)
    print(json.dumps(entities_result, indent=4, ensure_ascii=False))

