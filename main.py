# -*- coding: utf-8 -*-

import argparse
import torch
import torch.nn as nn
import logging

from models.BERT import BERT_K_fold
from models.AutoModel import AutoModel_K_fold

from generation.BART import BART_gen_K_fold, bart_gen_parameters
from generation.gpt2 import GPT_gen_K_fold, gpt_gen_parameters


models = {'BERT': BERT_K_fold, \
          'AutoModel': AutoModel_K_fold}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        help="Options: BERT, AutoModel")
    parser.add_argument("--state", type=str,
                        help="Options: classification, generation")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epoch_num", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding_dim", type=int, default=256,\
                        help="Length of embedding, Recommendations: 128, 256, 512")
    parser.add_argument("--dataset", type=str, default="../data/")
    parser.add_argument("--n_labels", type=int, default=12)
    parser.add_argument("--word2vec", type=str)
    parser.add_argument("--saved_path", type=str, default="trained_models")
    # parser.add_argument("--best_epoch_file", type=str, default="")
    args = parser.parse_args()
    return args


def main(config):

    if config.state == 'classification':
        if config.model not in models:
            logging.error(f"The model you chosen is not supported yet.")
            return
        
        if config.model != 'han':
            model = models[config.model](config.dataset, config.n_labels, \
                config.batch_size, config.epoch_num, config.lr, config.embedding_dim)

        if config.model == 'han':
            model = models[config.model](config.dataset, config.n_labels, \
                config.batch_size, config.epoch_num, config.lr, config.embedding_dim, \
                    config.word2vec)

    if config.state == 'generation':
        model = generations[config.model](config.dataset, \
                config.batch_size, config.epoch_num, config.lr, config.embedding_dim)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    opt = get_args()
    main(opt)