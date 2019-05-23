import sys
import time
import pandas as pd
from os import listdir
from os.path import isfile, join
import json
from .tasks import *


PATH_BERT = '../pytorch-pretrained-BERT'
sys.path.insert(0, PATH_BERT)

PATH_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data/'
PATH_TO_CACHE = './cache/'
sys.path.insert(0, PATH_SENTEVAL)
import senteval



def get_results(dir_path='./mlp_results'):
    columns = ['data_path', 'cache_path', 'result_path', 'batch_size', 'cbatch_size', 'nhid', 'optim', 'kfold',
               'tenacity', 'usepytorch', 'epoch_size', 'device']
    filenames = [f for f in listdir(dir_path) if isfile(join(dir_path, f)) if '.json' in f]
    list_result = []
    for filename in filenames:
        with open(join(dir_path, filename), 'r') as infile:
            #             print(filename)
            results = json.load(infile)
            for key, result in results.items():
                list_result.append(result)

    df = pd.DataFrame(list_result)[['acc', 'devacc', 'head', 'layer', 'task', 'model_name', 'location']]

    for column in columns:
        try:
            df = df.drop(columns=column)
        except:
            pass
    return df


def get_top_heads(model_name, task):
    df = get_results(dir_path='./ds_linear_head_wise_results')
    df = df.loc[df['model_name'] == model_name]
    print(df)

    df = df.loc[df['head'] >= 0]
    df = df.loc[df['task'] == task] # Choose task
    df = df.sort_values(by=['devacc'], ascending=False)
    list_head = []
    for index, row in df.iterrows():
        list_head.append((row['layer'], row['head']))
    return list_head


def save_exp_result(exp_result, task):
    del exp_result['model']
    exp_key = '{}_{}'.format(exp_result['num_head'], exp_result['location'])
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
    model = params['model']
    location = params['location']
    head_size = params['head_size']


    sentences = [' '.join(s) for s in batch]
    embedding = model.encode(sentences, params['heads'], head_size, location)
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
    params['heads'] = get_top_heads(args.model_name, task)[:args.num_head] # select first top n-heads

    se = senteval.engine.SE(params, batcher, prepare)
    result = se.eval([task])

    if task in ['SICKRelatedness']:
        params['devpearson'] = result[task]['devpearson']
        params['pearson'] = result[task]['pearson']

    elif task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark']:
        params['pearson'] = result[task]['all']['pearson']['mean']


    else:
        params['devacc'] = result[task]['devacc']
        params['acc'] = result[task]['acc']

    model.save_cache(task, args.location)

    te = time.time()
    print("result: {}, took: {:3.1f} sec".format(result, te - ts))
    return params