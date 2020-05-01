from os import listdir
from os.path import isfile, join
import json

import pandas as pd


def get_results(dir_path, model_name=None, part=None, task=None, metrics=['devacc']):
    if part and not part in ['head', 'layer']:
        raise ValueError(f"part={part} should be 'head' or 'layer'.")

    columns = ['data_path', 'cache_path', 'result_path', 'batch_size', 'cbatch_size', 'nhid', 'optim', 'kfold', 'tenacity', 'usepytorch', 'epoch_size', 'device']
    filenames = [f for f in listdir(dir_path) if isfile(join(dir_path, f)) if '.json' in f]
    list_result = []
    for filename in filenames:
        with open(join(dir_path, filename), 'r') as infile:
            results = json.load(infile)
            for key, result in results.items():
                list_result.append(result)
                
    df = pd.DataFrame(list_result)[['acc', 'head', 'layer', 'task', 'model_name', 'location'] + metrics] 

    # Filter by model name
    if model_name:
        df = df.loc[df['model_name'] == model_name]
    # Filter by pooling location
    if part == 'head':
        df = df.loc[df['head'] >= 0]
    elif part == 'layer':
        df = df.loc[df['head'] == -1]
    # Filter by task name
    if task:
        df = df.loc[df['task'] == task]

    for column in columns:
        try:
            df = df.drop(columns=column)
        except:
            pass

    return df


def find_top_n_layer(model_name, task, dir_path, n_layer=1):
    df = get_results(dir_path=dir_path, task=task, model_name=model_name, part='layer')
    if task in ['STSBenchmark', 'SICKRelatedness']:
        df = df.sort_values(by='devpearson', ascending=False)
    else:
        df = df.sort_values(by='devacc', ascending=False)

    top_n_layers = df['layer'][:n_layer].values
    return top_n_layers