import sys
from copy import deepcopy
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

PATH_BERT = '../pytorch-pretrained-BERT'
sys.path.insert(0, PATH_BERT)


PATH_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data/'
PATH_TO_CACHE = './cache/'
sys.path.insert(0, PATH_SENTEVAL)
import senteval

from encoder import BERTEncoder, GPTEncoder, GPT2Encoder, TransfoXLEncoder
from encoder.downstream_multi_head_exp import *


if __name__ == '__main__':

    # ====== Generate Embedding of Large Model ====== #
    parser = argparse.ArgumentParser(description='Evaluate BERT')
    parser.add_argument("--device", type=list, default=[1])
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--usepytorch", type=bool, default=True)
    parser.add_argument("--task_path", type=str, default='./SentEval/data/')
    parser.add_argument("--cache_path", type=str, default='./cache/')
    parser.add_argument("--result_path", type=str, default='./ds_top_head_wise_results/')
    parser.add_argument("--optim", type=str, default='rmsprop')
    parser.add_argument("--cbatch_size", type=int, default=256)
    parser.add_argument("--tenacity", type=int, default=3)
    parser.add_argument("--epoch_size", type=int, default=2)
    parser.add_argument("--model_name", type=str, default='openai-gpt') #

    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--num_head", type=int, default=12)
    parser.add_argument("--intv_head", type=int, default=1)
    parser.add_argument("--total_head", type=int, default=40)


    parser.add_argument("--location", type=str, default='head') #8, 15
    parser.add_argument("--head_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--nhid", type=int, default=0)

    args = parser.parse_args()
    args.seed = 123
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.device)




    list_num_head = [i for i in range(args.intv_head, args.total_head+args.intv_head, args.intv_head)]

    num_exp = len(list_num_head)

    print("======= Benchmark Configuration ======")
    print("Args: ", args)
    print("Device: ", args.device)
    print("model name: ", args.model_name)
    print("Task: ", tasks[args.task])
    print("location: ", args.location)
    print("Total Exps: ", num_exp)
    print("Num Heads: ", args.num_head)
    print("Interval : ", args.intv_head)
    print("======================================")

    cnt = 0
    args.task = tasks[args.task]
    if args.model_name in ['bert-base-uncased', 'bert-large-uncased'] :
        model = BERTEncoder(model_name=args.model_name, encode_capacity=args.batch_size)
    elif args.model_name == 'openai-gpt':
        model = GPTEncoder(encode_capacity=args.batch_size)
    elif args.model_name == 'gpt2':
        model = GPT2Encoder(encode_capacity=args.batch_size)
    elif args.model_name == 'transfo-xl-wt103':
        model = TransfoXLEncoder(encode_capacity=args.batch_size)
    else:
        raise ValueError

    """
    with tqdm(total=num_exp, file=sys.stdout) as pbar:
        for i in [15,16,19,17]:
            args.task = tasks[i]
            print('\n---------')

            exp_result = experiment(model, args.task, deepcopy(args))

            pbar.set_description('P: %d' % (1 + cnt))
            pbar.update(1)
            cnt += 1

            print("** Saving Best Result of Acc: {}.".format(exp_result['acc']))
            save_exp_result(exp_result, args.task)


    """

    with tqdm(total=num_exp, file=sys.stdout) as pbar:
        for num_head in list_num_head:
            args.num_head = num_head
            print('\n---------')
            print("{} heads".format(num_head))

            exp_result = experiment(model, args.task, deepcopy(args))

            pbar.set_description('P: %d' % (1 + cnt))
            pbar.update(1)
            cnt += 1

            print("** Saving Best Result of Acc: {}.".format(exp_result['acc']))
            save_exp_result(exp_result, args.task)

