
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
from os.path import isfile, join, isdir


TASK_DICT = {'MRPC': 'MRPC', 'STSBenchmark':'STS-B', 'SST2': 'SST-2'}



PATH_BERT = '../pytorch-pretrained-BERT'
sys.path.insert(0, PATH_BERT)


PATH_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'
PATH_TO_CACHE = './cache/'
sys.path.insert(0, PATH_SENTEVAL)
import senteval

from encoder import BERTEncoder, GPTEncoder, GPT2Encoder, TransfoXLEncoder
from encoder.downstream_single_head_exp import *


def get_ckpt_list(task, model_name, exp_name, seed, dir_path="/home/users/whwodud98/exp"):
    dir_path = "{}/{}/{}/{}/{}".format(dir_path, TASK_DICT[task], model_name, exp_name, seed)
    dirnames = [f for f in listdir(dir_path) if isdir(join(dir_path, f)) if 'checkpoint' in f]
    dirnames = [join(dir_path, dir_name) for dir_name in dirnames]
    return dirnames

if __name__ == '__main__':


    # ====== Generate Embedding of Large Model ====== #
    parser = argparse.ArgumentParser(description='Evaluate BERT')
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4000)
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--usepytorch", type=bool, default=True)
    parser.add_argument("--task_path", type=str, default='./SentEval/data/')
    parser.add_argument("--cache_path", type=str, default='./cache_ft/')
    parser.add_argument("--result_path", type=str, default='./ds_linear_head_wise_results_ft/')
    parser.add_argument("--optim", type=str, default='rmsprop')
    parser.add_argument("--cbatch_size", type=int, default=256)
    parser.add_argument("--tenacity", type=int, default=3)
    parser.add_argument("--epoch_size", type=int, default=2)
    parser.add_argument("--model_name", type=str, required=True)  #bert-base-uncased
    parser.add_argument("--exp_name", type=str, required=True)  #last3
    parser.add_argument("--seed", type=int, required=True)  #0
    parser.add_argument("--task", type=str, required=True) # MRPC->17 / STS-B -> 21 / SST-2 -> 14
    parser.add_argument("--ckpt", type=int, required=False) # MRPC->17 / STS-B -> 21 / SST-2 -> 14
    parser.add_argument("--ckpt_run", action='store_true', required=False) # MRPC->17 / STS-B -> 21 / SST-2 -> 14


    parser.add_argument("--layer", nargs='+', type=int, default=[0,11])
    parser.add_argument("--head", nargs='+', type=int, default=[0])
    parser.add_argument("--location", type=str, default='head')
    parser.add_argument("--head_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--nhid", type=int, default=0)

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7' #','.join(str(x) for x in args.device) if isinstance(args.device, list) else str(args.device) #

    list_ckpt = get_ckpt_list(args.task, args.model_name, args.exp_name, args.seed)
    list_layer = range(args.layer[0], args.layer[1]+1) if len(args.layer) > 1 else [args.layer[0]]
    list_head = range(args.head[0], args.head[1]+1) if len(args.head) > 1 else [args.head[0]]
    num_exp = len(list(list_layer)) * len(list_head) * len(list_ckpt)


    print("======= Benchmark Configuration ======")
    print("Device: ", args.device)
    print("model name: ", args.model_name)
    print("Task: ", args.task)
    print("range CKPT: ", list_ckpt)
    print("range layer: ", list_layer)
    print("range head: ", list_head)
    print("Total Exps: ", num_exp)
    print("======================================")


    if args.ckpt_run:
        num_exp = len(list(list_layer)) * len(list_head) * len(list_ckpt)
        cnt = 0
        with tqdm(total=num_exp, file=sys.stdout) as pbar:

            for i in [3, 5]:
                args.model_name = list_ckpt[i]

                #args.task = tasks[args.task]
                if 'bert-base-uncased' in args.model_name or 'bert-large-uncased' in args.model_name:
                    model = BERTEncoder(model_name=args.model_name, encode_capacity=args.batch_size, PATH_CACHE=args.cache_path)
                elif args.model_name == 'openai-gpt':
                    model = GPTEncoder(encode_capacity=args.batch_size)
                elif args.model_name == 'gpt2':
                    model = GPT2Encoder(encode_capacity=args.batch_size)
                elif args.model_name == 'transfo-xl-wt103':
                    model = TransfoXLEncoder(encode_capacity=args.batch_size)
                else:
                    raise ValueError

                args.head = 0
                args.layer = 0

                exp_result = experiment(model, args.task, deepcopy(args))

                pbar.set_description('P: %d' % (1 + cnt))
                pbar.update(1)
                cnt += 1

                if args.task in ['SICKRelatedness', 'STSBenchmark']:
                    print("** Saving Best Result of pearson: {}.".format(exp_result['pearson']))

                else:
                    print("** Saving Best Result of Acc: {}.".format(exp_result['acc']))
                save_exp_result(exp_result, args.task)

    else:
        num_exp = len(list(list_layer)) * len(list_head)
        cnt = 0

        with tqdm(total=num_exp, file=sys.stdout) as pbar:

            model_name = list_ckpt[args.ckpt]

            args.model_name = model_name
            # args.task = tasks[args.task]
            if 'bert-base-uncased' in args.model_name or 'bert-large-uncased' in args.model_name:
                model = BERTEncoder(model_name=args.model_name, encode_capacity=args.batch_size, PATH_CACHE=args.cache_path)
            elif args.model_name == 'openai-gpt':
                model = GPTEncoder(encode_capacity=args.batch_size)
            elif args.model_name == 'gpt2':
                model = GPT2Encoder(encode_capacity=args.batch_size)
            elif args.model_name == 'transfo-xl-wt103':
                model = TransfoXLEncoder(encode_capacity=args.batch_size)
            else:
                raise ValueError

            for head in list_head:
                for layer in list_layer:

                    print('\n---------')
                    print("L: {}. H: {}.".format(layer, head))
                    args.head = head
                    args.layer = layer

                    exp_result = experiment(model, args.task, deepcopy(args))

                    pbar.set_description('P: %d' % (1 + cnt))
                    pbar.update(1)
                    cnt += 1

                    if args.task in ['SICKRelatedness', 'STSBenchmark']:
                        print("** Saving Best Result of pearson: {}.".format(exp_result['pearson']))

                    else:
                        print("** Saving Best Result of Acc: {}.".format(exp_result['acc']))
                    save_exp_result(exp_result, args.task)
