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
sys.path.insert(0, PATH_SENTEVAL)
import senteval


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
    params['devacc'] = result[task]['devacc']
    params['acc'] = result[task]['acc']
    model.save_cache(task, args.location)

    te = time.time()
    print("result: {}, took: {:3.1f} sec".format(result, te - ts))
    return params