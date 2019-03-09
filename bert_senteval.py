#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#%load_ext autoreload
#%autoreload 2


# In[ ]:


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

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)


# In[ ]:


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


# In[ ]:


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


# In[ ]:


def efficient_batcher(batch):
    max_capacity = 3000
    seq_length = max([len(tokens) for tokens in batch])
    batch_size = len(batch)
    
    mini_batch = max_capacity // seq_length + 1
    return mini_batch


def prepare(params, samples):
    
    cache_name = "{}_{}.pickle".format(params.model_name, params.current_task)
    cache_dir = params.cache_path
    onlyfiles = [f for f in listdir(cache_dir) if isfile(join(cache_dir, f))]

    # ====== Look Up existing cache ====== #
    if cache_name in onlyfiles:
        print("cache found {}".format(cache_name))
        with open(join(cache_dir, cache_name), 'rb') as f:
            params['cache'] = pickle.load(f)
            params['cache_flag'] = True
        
    else:
        print("cache not found. Construct BERT model")
        params['cache'] = {}
        params['cache_flag'] = False
        
        # ====== Construct Model ====== #
        model = BertModel.from_pretrained(args.model_name)
        model = torch.nn.DataParallel(model)
        tokenizer = BertTokenizer.from_pretrained(args.model_name, do_lower_case=True)
        
        params['model'] = model
        params_senteval['tokenizer'] = tokenizer
 
    # ====== Initializ Counter ====== #
    params['count'] = 0


def batcher(params, batch):
    ts = time.time()
    if params.cache_flag:
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
            params.model.eval()
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
    
    if params.count % 20000 == 0:
        print('{:6}'.format(params.count), 'encoded result', output.shape, 'return result', embedding.shape, 'took', '{:2.3f}'.format(te-ts), 'process', '{:4.1f}'.format(len(batch)/(te-ts)))

    return embedding


# In[ ]:


def experiment(args, task):
    ts = time.time()
        
    # ====== SentEval Engine Setting ====== #
    params_senteval = {'task_path': args.data_path, 
                       'usepytorch': args.usepytorch, 
                       'seed': seed,
                       'batch_size': args.batch_size,
                       'nhid': args.nhid,
                       'kfold': args.kfold}
    params_senteval['classifier'] = {'nhid': args.nhid, 'optim': args.optim, 'batch_size': args.cbatch_size,
                                     'tenacity': args.tenacity, 'epoch_size': args.epoch_size}
    
    # ====== Experiment Setting ====== #
    params_senteval['model_name'] = args.model_name
    params_senteval['cache_path'] = args.cache_path
    params_senteval['result_path'] = args.result_path
    params_senteval['layer'] = args.layer
    params_senteval['head'] = args.head
    params_senteval['head_size'] = args.head_size

    # ====== Conduct Experiment ====== #
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    result = se.eval([task])
    
    # ====== Logging Experiment Result ====== #
    exp_result = vars(deepcopy(args))
    exp_result['task'] = task
    exp_result['devacc'] = result[task]['devacc']
    exp_result['acc'] = result[task]['acc']
    save_exp_result(exp_result)

    # ====== Save Cache ====== #
    if not se.params.cache_flag:
        cache_name = "{}_{}.pickle".format(se.params.model_name, se.params.current_task)
        cache_dir = se.params.cache_path
        with open(join(cache_dir, cache_name), 'wb') as f:
            pickle.dump(se.params.cache, f, pickle.HIGHEST_PROTOCOL)
        print("Saved cache {}".format(cache_name))
        
    # ====== Reporting ====== #
    te = time.time()
    print("result: {}, took: {:3.1f} sec".format(result, te-ts))


# In[ ]:


tasks = ['Length', 'WordContent', 'Depth', 'TopConstituents',
         'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
         'OddManOut', 'CoordinationInversion']

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser(description='Evaluate BERT')
parser.add_argument("--device", type=list, default=[1,2])
parser.add_argument("--batch_size", type=int, default=500)
parser.add_argument("--nhid", type=int, default=0)
parser.add_argument("--kfold", type=int, default=5)
parser.add_argument("--usepytorch", type=bool, default=True)
parser.add_argument("--data_path", type=str, default='./SentEval/data/')
parser.add_argument("--cache_path", type=str, default='./cache/')
parser.add_argument("--result_path", type=str, default='./results/')
parser.add_argument("--optim", type=str, default='rmsprop')
parser.add_argument("--cbatch_size", type=int, default=512)
parser.add_argument("--tenacity", type=int, default=3)
parser.add_argument("--epoch_size", type=int, default=2)
parser.add_argument("--model_name", type=str, default='bert-base-uncased')

parser.add_argument("--task", type=int, default=0)
parser.add_argument("--layer", type=int, default=[0, 11])
parser.add_argument("--head", type=int, default=[-1, 11])
parser.add_argument("--head_size", type=int, default=64)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.device)


list_layer = range(args.layer[0], args.layer[1]+1) if len(args.layer) > 1 else [args.layer[0]]
list_head = range(args.head[0], args.head[1]+1) if len(args.head) > 1 else [args.head[0]]
num_exp = len(list(list_layer)) * len(list(list_head))


print("======= Benchmark Configuration ======")
print("Device: ", args.device)
print("model name: ", args.model_name)
print("Task: ", tasks[args.task])
print("range layer: ", list_layer)
print("range head: ", list_head)
print("Total Exps: ", num_exp)
print("======================================")

cnt = 0
target_task = tasks[args.task]
with tqdm(total=num_exp, file=sys.stdout) as pbar:
    for layer in list_layer:
        for head in list_head:
            args.layer = layer
            args.head = head
            print()
            experiment(args, target_task)

            pbar.set_description('processed: %d' % (1 + cnt))
            pbar.update(1)
            cnt += 1


# In[ ]:




