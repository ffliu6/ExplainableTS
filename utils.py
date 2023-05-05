# -*- coding: utf-8 -*-

import os
import xlrd
import json
import numpy as np

import nltk
nltk.data.path.append('./nltk_data/')
from tqdm import tqdm



def custom_adjacent_accuracy_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sum(np.absolute(y_pred - y_true) <= 1) / len(y_pred)


def json2list4brm(path):
    data = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        jsondata = json.load(fp=f)
        for line in jsondata:
            para_id = line['para_id']
            sent_id = line['sent_id']
            fi_id = str(para_id) + '_' + str(sent_id)
            sent = line['sent'].strip()
            label = line['label']

            data.append([fi_id, \
                label, \
                sent]) # label, input
    
    return data


def readelajson(path):
    pairs = []
    
    with open(path, 'r', encoding='utf-8') as f:
        jsondata = json.load(fp=f)
        for line in jsondata:
            id = line["id"]
            # ori_text = line["ori_text"]
            apt_text = line["apt_text"]
            # lan_text = line["lan_text"]
            ela_text = line["ela"]

            pairs.append([id, apt_text, ela_text])

    return pairs


def readwanjson(path):
    pairs = []
    
    with open(path, 'r', encoding='utf-8') as f:
        jsondata = json.load(fp=f)
        for line in jsondata:
            id = line["id"]
            sour_text = line["sour_text"]
            simp_text = line["simp_text"]
            ela_text = line['ela_text']
            # sour_feats = line['sour_feats']
            # simp_feats = line['simp_feats']

            input_text = simp_text
            output_text = ela_text
            if ela_text != "":
                pairs.append([id, input_text, output_text])
            else:
                continue

    return pairs


def readwanjson_lf(path):
    pairs = []
    
    with open(path, 'r', encoding='utf-8') as f:
        jsondata = json.load(fp=f)
        for line in jsondata:
            id = line["id"]
            sour_text = line["sour_text"]
            simp_text = line["simp_text"]
            ela_text = line['ela_text']
            sour_feats = line['sour_feats']
            simp_feats = line['simp_feats']

            input_text = simp_text
            output_text = ela_text
            input_feats = simp_feats
            if ela_text != "":
                pairs.append([id, input_text, output_text, input_feats])
            else:
                continue

    return pairs


def json2list4cae(path):
    data = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        jsondata = json.load(fp=f)
        for line in jsondata:
            para_id = line['para_id']
            sent_id = line['sent_id']
            fi_id = para_id + '_' + str(sent_id)
            data.append([fi_id, \
            int(line['label']), \
            line['sent'].strip()]) # label, input
    
    return data


def json2list4cae_lf(path):
    data = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        jsondata = json.load(fp=f)
        for line in jsondata:
            para_id = line['para_id']
            sent_id = line['sent_id']
            fi_id = para_id + '_' + str(sent_id)
            data.append([fi_id, \
            int(line['label']), \
            line['feats'], \
            line['sent'].strip()]) # label, input
    
    return data
