import sys
from os import listdir
from os.path import isfile, join
import pickle
import hashlib
import time
import os

import numpy as np
import torch

PATH_BERT = '/home/jovyan/drmoacl/transformer_anatomy'
sys.path.insert(0, PATH_BERT)

from transformers import BertTokenizer, BertModel
from transformer_anatomy.extractor import BertExtractor
from .encoder import BaseEncoder


class BERTEncoder(BaseEncoder):

    def __init__(self, model_name, encode_capacity=3000, path_cache='.cache'):
        super(BERTEncoder, self).__init__(model_name, encode_capacity, path_cache)

    def construct_encoder(self):
        model = BertModel.from_pretrained(self.model_name, output_hidden_states=True, output_attentions=True)
        model = BertExtractor(model, location=None, heads=None)
        model.cuda()
        model = torch.nn.DataParallel(model)
        model.eval()

        print('tokenizer', self.model_name)
        tokenizer = BertTokenizer.from_pretrained(self.model_name, do_lower_case=True)
        print("Model and tokenzier are constructed!")
        return model, tokenizer

    def convert_sentences_to_features(self, sentences, seq_length):
        """Convert sentence into Tensor"""

        num_sent = len(sentences)
        input_type_ids = np.zeros((num_sent, seq_length), dtype=np.int32)
        input_ids = np.zeros((num_sent, seq_length), dtype=np.int32)
        input_mask = np.zeros((num_sent, seq_length), dtype=np.int32)

        for idx, sent in enumerate(sentences):
            tokens = self.tokenizer.tokenize(sent)
            tokens = tokens[0:min((seq_length - 2), len(tokens))]  # truncate tokens longer than seq_length
            tokens.insert(0, "[CLS]")
            tokens.append("[SEP]")

            input_ids[idx, :len(tokens)] = np.array(self.tokenizer.convert_tokens_to_ids(tokens), dtype=np.int32)
            input_mask[idx, :len(tokens)] = np.ones(len(tokens), dtype=np.int32)

            assert len(input_ids[idx]) == seq_length
            assert len(input_mask[idx]) == seq_length
            assert len(input_type_ids[idx]) == seq_length

        return input_ids, input_type_ids, input_mask

    def encode(self, sentences, heads, head_size, location):
        ts = time.time()
        # self.model.eval()

        if not self.flag_cache_save:
            output = []
            for i, sent in enumerate(sentences):
                hask_key = hashlib.sha256(sent.encode()).hexdigest()
                output.append(self.cache[hask_key])
            output = np.array(output)
        else:
            mini_batch_size = self.get_mini_batch_size(sentences)
            idx = 0
            list_output = []
            while idx < len(sentences):
                mini_batch = sentences[idx:min(idx + mini_batch_size, len(sentences))]
                seq_length = max([len(tokens) for tokens in mini_batch])

                # ====== Convert to Tensor ====== #
                input_ids, input_type_ids, input_mask = self.convert_sentences_to_features(mini_batch, seq_length)
                input_ids = torch.Tensor(input_ids).long().cuda()
                input_type_ids = torch.Tensor(input_type_ids).long().cuda()
                input_mask = torch.Tensor(input_mask).long().cuda()

                # ====== Encode Tokens ====== #
                all_hidden_states, all_head_states = self.model(input_ids, input_type_ids, input_mask)
                torch.cuda.synchronize()

                if location == 'fc':
                    output = np.array([layer[:, 0, :].detach().cpu().numpy() for layer in all_hidden_states])
                elif location == 'head':
                    output = np.array([layer[:, 0, :].detach().cpu().numpy() for layer in all_head_states])

                output = np.swapaxes(output, 0, 1)
                list_output.append(output)

                # ====== Construct Cache ====== #
                temp_cache = {}
                for i, sent in enumerate(mini_batch):
                    hask_key = hashlib.sha256(sent.encode()).hexdigest()
                    temp_cache[hask_key] = output[i]
                self.cache.update(temp_cache)

                idx += mini_batch_size
                self.count += mini_batch_size
            output = np.concatenate(list_output, 0)
            te = time.time()
            print('encoding with model', len(sentences), 'processed', self.count, 'took', '{:4.1f}'.format(te-ts))


        te = time.time()
        embedding = self.get_multi_head_embedding(output, heads, head_size)
        return embedding


if __name__ == '__main__':
    model = BertModel.from_pretrained("/home/users/whwodud98/exp/MRPC/bert-base-uncased/last3/0/checkpoint-0")
    model.cuda()
    model = torch.nn.DataParallel(model)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained("/home/users/whwodud98/exp/MRPC/bert-base-uncased/last3/0/checkpoint-0", do_lower_case=True)


