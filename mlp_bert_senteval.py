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

from pytorch_pretrained_bert import BertTokenizer, BertModel

PATH_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data/'
PATH_TO_CACHE = './cache/'
sys.path.insert(0, PATH_SENTEVAL)
import senteval


# In[4]:


def convert_sentences_to_features(sentences, seq_length, tokenizer):
    """Convert sentence into Tensor"""
    
    num_sent = len(sentences)
    input_type_ids = np.zeros((num_sent, seq_length), dtype=np.int32)
    input_ids = np.zeros((num_sent, seq_length), dtype=np.int32)
    input_mask = np.zeros((num_sent, seq_length), dtype=np.int32)
    
    for idx, sent in enumerate(sentences):
        tokens = tokenizer.tokenize(sent)
        tokens = tokens[0:min((seq_length - 2), len(tokens))] # truncate tokens longer than seq_length
        tokens.insert(0, "[CLS]")
        tokens.append("[SEP]")
        
        input_ids[idx,:len(tokens)] = np.array(tokenizer.convert_tokens_to_ids(tokens), dtype=np.int32)
        input_mask[idx,:len(tokens)] = np.ones(len(tokens), dtype=np.int32)

        assert len(input_ids[idx]) == seq_length
        assert len(input_mask[idx]) == seq_length
        assert len(input_type_ids[idx]) == seq_length

    return input_ids, input_type_ids, input_mask


# In[5]:


def save_exp_result(exp_result):
    exp_key = '{}_{}'.format(exp_result['layer'], exp_result['head'])
    print(exp_key)
    result_name = "{}_{}.json".format(exp_result['model_name'], exp_result['task'])
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


# In[6]:


def load_cache(model_name, task, cache_path):
    cache_name = "{}_{}.pickle".format(model_name, task)
    cache_dir = cache_path
    onlyfiles = [f for f in listdir(cache_dir) if isfile(join(cache_dir, f))]

    # ====== Look Up existing cache ====== #
    if cache_name in onlyfiles:
        print("cache Found {}".format(cache_name))
        with open(join(cache_dir, cache_name), 'rb') as f:
            cache = pickle.load(f)
            print("cache Loaded")
            return cache
        
    else:
        print("cache not Found {}".format(cache_name))
        return None


# In[7]:


def efficient_batcher(batch):
    max_capacity = 3000
    seq_length = max([len(tokens) for tokens in batch])
    batch_size = len(batch)
    
    mini_batch = max_capacity // seq_length + 1
    return mini_batch


def prepare(params, samples):
    
    if params['cache'] is None: # check whether cache is already provided
        params['cache'] = load_cache(params.model_name, params.current_task, params.cache_path) # try to load cache

        if params['cache'] is None: # if there is no cache saved, then construct encoder model
            print("Constructing Encoder Model")
            params['cache'] = {}

            # ====== Construct Model ====== #
            model = BertModel.from_pretrained(args.model_name)
            model = torch.nn.DataParallel(model)
            tokenizer = BertTokenizer.from_pretrained(args.model_name, do_lower_case=True)

            params['model'] = model
            params['tokenizer'] = tokenizer
            params['flag_save'] = True
             
    # ====== Initializ Counter ====== #
    params['count'] = 0


def batcher(params, batch):
    ts = time.time()
    if params.cache != {}:
        output = []
        sentences = [' '.join(s) for s in batch]
        for i, sent in enumerate(sentences):
            hask_key = hashlib.sha256(sent.encode()).hexdigest()
            output.append(params.cache[hask_key])
        output = np.array(output)
        
    else:
        mini_batch_size = efficient_batcher(batch)

        idx = 0
        list_output = []
        while idx < len(batch):
            mini_batch = batch[idx:min(idx+mini_batch_size, len(batch))]

            # ====== Token Preparation ====== #
            model = params.model
            model.eval()
            seq_length = max([len(tokens) for tokens in mini_batch])
            sentences = [' '.join(s) for s in mini_batch]

            # ====== Convert to Tensor ====== #
            input_ids, input_type_ids, input_mask = convert_sentences_to_features(sentences, seq_length, params.tokenizer)
            input_ids = torch.Tensor(input_ids).long().cuda()
            input_type_ids = torch.Tensor(input_type_ids).long().cuda()
            input_mask = torch.Tensor(input_mask).long().cuda()

            # ====== Encode Tokens ====== #
            encoded_layers, _ = model(input_ids, input_type_ids, input_mask)   
            torch.cuda.synchronize()
            output = np.array([layer[:, 0, :].detach().cpu().numpy() for layer in encoded_layers]) 
            output = np.swapaxes(output, 0, 1)
            list_output.append(output)
            idx += mini_batch_size

            # ====== Construct Cache ====== #
            temp_cache = {}
            for i, sent in enumerate(sentences):
                hask_key = hashlib.sha256(sent.encode()).hexdigest()
                temp_cache[hask_key] = output[i]
            params.cache.update(temp_cache)    
            output = np.concatenate(list_output, 0)      
    
    te = time.time()
    params.count += len(batch)
    # ====== Extract Target Embedding (layer, head) ====== #
    if params.head == -1:
        embedding = output[:, params.layer, :]
    else:
        embedding = output[:, params.layer, params.head*params.head_size:(params.head+1)*params.head_size]
    
    # if params.count % 20000 == 0:
    #     print('{:6}'.format(params.count), 'encoded result', output.shape, 'return result', embedding.shape, 'took', '{:2.3f}'.format(te-ts), 'process', '{:4.1f}'.format(len(batch)/(te-ts)))

    return embedding


# In[8]:


def experiment(args, task, cache=None):
    ts = time.time()
        
    # ====== SentEval Engine Setting ====== #
    params_senteval = {'task_path': args.data_path, 
                       'seed': args.seed,
                       'usepytorch': args.usepytorch, 
                       'kfold': args.kfold,
                       'batch_size': args.batch_size}
    
    params_senteval['classifier'] = {'nhid': args.nhid, 
                                     'optim': args.optim, 
                                     'tenacity': args.tenacity,
                                     'epoch_size': args.epoch_size,
                                     'dropout': args.dropout,
                                     'batch_size': args.cbatch_size,}
    
    # ====== Experiment Setting ====== #
    params_senteval['model_name'] = args.model_name
    params_senteval['cache_path'] = args.cache_path
    params_senteval['result_path'] = args.result_path
    params_senteval['layer'] = args.layer
    params_senteval['head'] = args.head
    params_senteval['head_size'] = args.head_size
    params_senteval['cache'] = cache

    # ====== Conduct Experiment ====== #
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    result = se.eval([task])
    
    # ====== Logging Experiment Result ====== #
    exp_result = vars(deepcopy(args))
    exp_result['task'] = task
    exp_result['devacc'] = result[task]['devacc']
    exp_result['acc'] = result[task]['acc']

    # ====== Save Cache ====== #
    if 'flag_save' in se.params:
        print("Start saving cache")
        cache_name = "{}_{}.pickle".format(se.params.model_name, se.params.current_task)
        cache_dir = se.params.cache_path
        with open(join(cache_dir, cache_name), 'wb') as f:
            pickle.dump(se.params.cache, f, pickle.HIGHEST_PROTOCOL)
        print("Saved cache {}".format(cache_name))
        
    # ====== Reporting ====== #
    te = time.time()
    print("result: {}, took: {:3.1f} sec".format(result, te-ts))
    return exp_result


# In[10]:


tasks = ['Length', 'WordContent', 'Depth', 'TopConstituents',
         'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
         'OddManOut', 'CoordinationInversion', 'CR', 'MR', 
         'MPQA', 'SUBJ', 'SST2', 'SST5', 
         'TREC', 'MRPC', 'SNLI', 'SICKEntailment', 
         'SICKRelatedness', 'STSBenchmark', 'ImageCaptionRetrieval', 'STS12',
         'STS13', 'STS14', 'STS15', 'STS16',]



parser = argparse.ArgumentParser(description='Evaluate BERT')
parser.add_argument("--device", type=list, default=[1,2])
parser.add_argument("--batch_size", type=int, default=500)
parser.add_argument("--nhid", type=int, default=0)
parser.add_argument("--kfold", type=int, default=5)
parser.add_argument("--usepytorch", type=bool, default=True)
parser.add_argument("--data_path", type=str, default='./SentEval/data/')
parser.add_argument("--cache_path", type=str, default='./cache/')
parser.add_argument("--result_path", type=str, default='./fine_mlp_results/')
parser.add_argument("--optim", type=str, default='rmsprop')
parser.add_argument("--cbatch_size", type=int, default=256)
parser.add_argument("--tenacity", type=int, default=3)
parser.add_argument("--epoch_size", type=int, default=2)
parser.add_argument("--model_name", type=str, default='bert-large-uncased')

parser.add_argument("--task", type=int, default=0)
parser.add_argument("--layer", nargs='+', type=int, default=[0, 23])
parser.add_argument("--head", nargs='+', type=int, default=[-1, 15]) #8, 15
parser.add_argument("--head_size", type=int, default=64)

args = parser.parse_args()
args.seed = 123
np.random.seed(args.seed)
torch.manual_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.device)

# ====== Fine setting ====== #
args.kfold = 10
args.cbatch_size = 64
args.tenacity = 5
args.epoch_size = 4
args.optim = 'adam'
# =========================== #


list_layer = range(args.layer[0], args.layer[1]+1) if len(args.layer) > 1 else [args.layer[0]]
list_head = range(args.head[0], args.head[1]+1) if len(args.head) > 1 else [args.head[0]]
list_dropout = [0.0, 0.1, 0.2]
list_nhid = [50, 100, 200]
num_exp = len(list(list_layer)) * len(list_dropout) * len(list_nhid) * len(list_head)


print("======= Benchmark Configuration ======")
print("Device: ", args.device)
print("model name: ", args.model_name)
print("Task: ", tasks[args.task])
print("range layer: ", list_layer)
print("range head: ", list_head)
print("Total Exps: ", num_exp)
print("======================================")


cnt = 0
loaded_cache = None
target_task = tasks[args.task]

with tqdm(total=num_exp, file=sys.stdout) as pbar:
    for head in list_head:
        for layer in list_layer:
            best_acc = 0
            best_result = None
            list_acc = []
            for dropout in list_dropout:
                for nhid in list_nhid:

                    if loaded_cache is None:
                        loaded_cache = load_cache(args.model_name, target_task, args.cache_path)

                    print('\n---------')
                    print("L: {}. H: {}. p: {}. hid: {}".format(layer, head, dropout, nhid))
                    args.head = head
                    args.layer = layer
                    args.dropout = dropout
                    args.nhid = nhid

                    exp_result = experiment(args, target_task, cache=loaded_cache)
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
            save_exp_result(best_result)

