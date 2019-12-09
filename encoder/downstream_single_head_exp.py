import sys
import time
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


def parse_model_name(model_name):
    temp = model_name.split('/')
    task, model, exp_name, seed, ckpt = temp[5:]
    ckpt = int(ckpt.split('-')[-1])
    return task, model, exp_name, seed, ckpt


def generate_result_name(model_name, task):
    print("in generate result name")
    print(model_name)
    if '/' in model_name:
        task, model, exp_name, seed, ckpt = parse_model_name(model_name)
        return "{}_{}_{}_{}_{}.json".format(task, model, exp_name, seed, ckpt)
    else:
        return "{}_{}.json".format(model_name, task)


def save_exp_result(params, task):
    del params['model']
    exp_key = '{}_{}_{}'.format(params['layer'], params['head'], params['location'])
    # result_name = "{}_{}.json".format(params['model_name'], task)
    result_name = generate_result_name(params['model_name'], task)

    result_dir = params['result_path']
    onlyfiles = [f for f in listdir(result_dir) if isfile(join(result_dir, f))]

    if '/' in params['model_name']:
        task, model, exp_name, seed, ckpt = parse_model_name(params['model_name'])
        params.update({'exp_name':exp_name, 'seed':seed, 'ckpt':ckpt})

    if result_name in onlyfiles:
        with open(join(result_dir, result_name), 'r') as f:
            results = json.load(f)

        with open(join(result_dir, result_name), 'w') as f:
            results[exp_key] = params
            json.dump(results, f)
        print("Append exp result at {} with key {}".format(result_name, exp_key))

    else:
        results = {}
        with open(join(result_dir, result_name), 'w') as f:
            results[exp_key] = params
            json.dump(results, f)
        print("Create new exp result at {} with key {}".format(result_name, exp_key))


def prepare(params, _):
    task = params['current_task']
    model = params['model']
    location = params['location']
    model.prepare(task, location)


def batcher(params, batch):
    model = params['model']
    layer = params['layer']
    head = params['head']
    head_size = params['head_size']
    location = params['location']

    sentences = [' '.join(s) for s in batch]
    heads = [(layer, head)]
    embedding = model.encode(sentences, heads, head_size, location)
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

    if task in ['SICKRelatedness']:
        params['devpearson'] = result[task]['devpearson']
        params['pearson'] = result[task]['pearson']

    elif task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
        params['pearson'] = result[task]['all']['pearson']['mean']

    elif task in ['STSBenchmark']:
        params['devpearson'] = result[task]['devpearson']
        params['pearson'] = result[task]['pearson']
        params['spearman'] = result[task]['spearman']


    else:
        params['devacc'] = result[task]['devacc']
        params['acc'] = result[task]['acc']

    model.save_cache(task, args.location)

    te = time.time()
    print("result: {}, took: {:3.1f} sec".format(result, te - ts))
    return params