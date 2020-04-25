from os import listdir
from os.path import isfile, join
import json

import pandas as pd


def get_results(dir_path, model_name=None, part=None):
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
                
    df = pd.DataFrame(list_result)[['acc', 'head', 'layer', 'task', 'model_name', 'location', 'devacc']]

    # Filter out results
    if model_name:
        df = df.loc[df['model_name'] == model_name]
    if part == 'head':
        df = df.loc[df['head'] >= 0]
    elif part == 'layer':
        df = df.loc[df['head'] == -1]

    for column in columns:
        try:
            df = df.drop(columns=column)
        except:
            pass

    return df