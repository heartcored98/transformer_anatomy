from os import listdir
from os.path import isfile, join
import pickle
import numpy as np

TASK_DICT = {'MRPC': 'mrpc', 'STS-B': 'STSBenchmark', 'SST-2': 'SST2'}


class BaseEncoder():

    def __init__(self, model_name, encode_capacity, path_cache):
        self.model_name = model_name

        self.encode_capacity = encode_capacity
        self.path_cache = path_cache

        self.model = None
        self.tokenizer = None
        self.count = 0

    def parse_model_name_to_cache_name(self, model_name, task, location):
        if '/' in model_name:
            temp = model_name.split('/')
            task, model, exp_name, seed, ckpt = temp[5:]
            task = TASK_DICT[task]
            return "{}_{}_{}_{}_{}.pickle".format(task, model, exp_name, seed, ckpt)
        else:
            return "{}_{}_{}.pickle".format(model_name, task, location)

    def load_cache(self, task, location):
        cache_name = self.parse_model_name_to_cache_name(self.model_name, task, location)
        onlyfiles = [f for f in listdir(self.path_cache) if isfile(join(self.path_cache, f))]

        # ====== Look Up existing cache ====== #
        if cache_name in onlyfiles:
            print("cache Found {}".format(cache_name))
            with open(join(self.path_cache, cache_name), 'rb') as f:
                cache = pickle.load(f)
                print("cache Loaded")
                self.flag_cache_save = False
                return cache
        else:
            print("cache not Found {}".format(cache_name))
            self.flag_cache_save = True
            return {}

    def save_cache(self, task, location):
        if self.flag_cache_save:
            print("Start saving cache")
            cache_name = self.parse_model_name_to_cache_name(self.model_name, task, location)
            with open(join(self.path_cache, cache_name), 'wb') as f:
                pickle.dump(self.cache, f, pickle.HIGHEST_PROTOCOL)
            print("Saved cache {}".format(cache_name))
        else:
            print("Skipping saving cache")

    def prepare(self, task, location):
        self.cache = self.load_cache(task, location)
        if bool(self.cache):
            self.model = None
            self.tokenizer = None
            self.count = 0
        else:
            self.model, self.tokenizer = self.construct_encoder()

    def get_mini_batch_size(self, sentences):
        seq_length = max([len(tokens) for tokens in sentences])
        mini_batch_size = self.encode_capacity // seq_length + 1
        return mini_batch_size

    def get_head_embedding(self, output, layer, head, head_size):
        if head == -1:
            embedding = output[:, layer, :]
        else:
            embedding = output[:, layer, head * head_size:(head + 1) * head_size]
        return embedding

    def get_multi_head_embedding(self, output, heads, head_size):
        if len(heads) == 1: # If single attention head is probed
            layer, head = heads[0]
            embedding = self.get_head_embedding(output, layer, head, head_size)
        else: # If multiple attention head is selected
            list_embedding = []
            for layer, head in heads:
                embedding = self.get_head_embedding(output, layer, head, head_size)
                list_embedding.append(embedding)
            embedding = np.concatenate(list_embedding, axis=1)
        return embedding

    def construct_encoder(self):
        raise NotImplementedError

    def convert_sentences_to_features(self, sentences, seq_length):
        raise NotImplementedError

    def encode(self, sentences, heads, head_size, location):
        raise NotImplementedError


if __name__ == '__main__':
    model = BERTEncoder('bert-base-uncased')
    model.prepare('Length')
    model.construct_encoder()


