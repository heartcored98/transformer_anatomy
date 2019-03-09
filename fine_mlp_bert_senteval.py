#!/usr/bin/env python
# coding: utf-8

# In[2]:


#%load_ext autoreload
#%autoreload 2


# In[3]:


import sys
import torch
import numpy as np
import time
import hashlib
from os import listdir
from os.path import isfile, join
import pickle
import argparse
import json
from tqdm import tqdm
from copy import deepcopy
import os

PATH_BERT = '../pytorch-pretrained-BERT'
sys.path.insert(0, PATH_BERT)
from pytorch_pretrained_bert import BertTokenizer, BertModel

PATH_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data/'
PATH_TO_CACHE = './cache/'
sys.path.insert(0, PATH_SENTEVAL)
import senteval

from encoder import BERTEncoder


def save_exp_result(exp_result, task):
    del exp_result['model']
    exp_key = '{}_{}_{}'.format(exp_result['layer'], exp_result['head'], exp_result['location'])
    result_name = "{}_{}.json".format(exp_result['model_name'], task)
    result_dir = exp_result['result_path']
    onlyfiles = [f for f in listdir(result_dir) if isfile(join(result_dir, f))]

    if result_name in onlyfiles:
        with open(join(result_dir, result_name), 'r') as f:
            results = json.load(f)
            
        with open(join(result_dir, result_name), 'w') as f:
            results[exp_key] = exp_result
            json.dump(results, f)
        print("Append exp result at {} with key {}".format(result_name, exp_key))
            
    else:
        results = {}
        with open(join(result_dir, result_name), 'w') as f:
            results[exp_key] = exp_result
            json.dump(results, f)
        print("Create new exp result at {} with key {}".format(result_name, exp_key))


def prepare(params, _):
    task = params['current_task']
    model = params['model']
    location = params['location']
    model.prepare(task, location)


def batcher(params, batch):
    ts = time.time()
    model = params['model']
    layer = params['layer']
    head = params['head']
    head_size = params['head_size']
    location = params['location']


    sentences = [' '.join(s) for s in batch]
    embedding = model.encode(sentences, layer, head, head_size, location)
    return embedding


def experiment(model, task, args):
    ts = time.time()

    params = vars(args)
    params['model'] = model
    params['classifier'] = {'nhid': args.nhid,
                            'optim': args.optim,
                            'tenacity': args.tenacity,
                            'epoch_size': args.epoch_size,
                            'dropout': args.dropout,
                            'batch_size': args.cbatch_size}

    se = senteval.engine.SE(params, batcher, prepare)
    result = se.eval([task])
    params['devacc'] = result[task]['devacc']
    params['acc'] = result[task]['acc']
    model.save_cache(task, args.location)

    te = time.time()
    print("result: {}, took: {:3.1f} sec".format(result, te-ts))
    return params


tasks = ['Length', 'WordContent', 'Depth', 'TopConstituents',
         'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
         'OddManOut', 'CoordinationInversion', 'CR', 'MR', 
         'MPQA', 'SUBJ', 'SST2', 'SST5', 
         'TREC', 'MRPC', 'SNLI', 'SICKEntailment', 
         'SICKRelatedness', 'STSBenchmark', 'ImageCaptionRetrieval', 'STS12',
         'STS13', 'STS14', 'STS15', 'STS16',]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate BERT')
    parser.add_argument("--device", type=list, default=[1,2])
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--nhid", type=int, default=0)
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--usepytorch", type=bool, default=True)
    parser.add_argument("--task_path", type=str, default='./SentEval/data/')
    parser.add_argument("--cache_path", type=str, default='./cache/')
    parser.add_argument("--result_path", type=str, default='./encoder_test_results/')
    parser.add_argument("--optim", type=str, default='rmsprop')
    parser.add_argument("--cbatch_size", type=int, default=256)
    parser.add_argument("--tenacity", type=int, default=3)
    parser.add_argument("--epoch_size", type=int, default=2)
    parser.add_argument("--model_name", type=str, default='bert-base-uncased')

    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--layer", nargs='+', type=int, default=[0])
    parser.add_argument("--head", nargs='+', type=int, default=[-1]) #8, 15
    parser.add_argument("--location", type=str, default='head') #8, 15
    parser.add_argument("--head_size", type=int, default=64)

    args = parser.parse_args()
    args.seed = 123
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.device)

    """
    # ====== Fine setting ====== #
    args.kfold = 10
    args.cbatch_size = 64
    args.tenacity = 5
    args.epoch_size = 4
    args.optim = 'adam'
    # =========================== #
    """

    list_layer = range(args.layer[0], args.layer[1]+1) if len(args.layer) > 1 else [args.layer[0]]
    list_head = range(args.head[0], args.head[1]+1) if len(args.head) > 1 else [args.head[0]]
    list_dropout = [0.0] #, 0.1, 0.2]
    list_nhid = [50] #, 100, 200]
    num_exp = len(list(list_layer)) * len(list_dropout) * len(list_nhid) * len(list_head)


    print("======= Benchmark Configuration ======")
    print("Args: ", args)
    print("Device: ", args.device)
    print("model name: ", args.model_name)
    print("Task: ", tasks[args.task])
    print("range layer: ", list_layer)
    print("range head: ", list_head)
    print("location: ", args.location)
    print("Total Exps: ", num_exp)
    print("======================================")


    cnt = 0
    args.task = tasks[args.task]
    model = BERTEncoder(args.model_name)

    with tqdm(total=num_exp, file=sys.stdout) as pbar:
        for head in list_head:
            for layer in list_layer:
                best_acc = 0
                best_result = None
                list_acc = []
                for dropout in list_dropout:
                    for nhid in list_nhid:

                        print('\n---------')
                        print("L: {}. H: {}. p: {}. hid: {}".format(layer, head, dropout, nhid))
                        args.head = head
                        args.layer = layer
                        args.dropout = dropout
                        args.nhid = nhid

                        exp_result = experiment(model, args.task, deepcopy(args))
                        list_acc.append(exp_result['acc'])
                        if exp_result['acc'] > best_acc:
                            best_acc = exp_result['acc']
                            best_result = exp_result

                        pbar.set_description('P: %d' % (1 + cnt))
                        pbar.update(1)
                        cnt += 1

                print('***************')
                print("Saving Best Result of Acc: {}. L: {}. H: {}. p: {}. hid: {}".format(best_acc, best_result['layer'], best_result['head'], best_result['dropout'], best_result['nhid']))
                print("Among: ", list_acc)
                save_exp_result(best_result, args.task)

