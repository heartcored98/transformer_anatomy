import sys
from os import listdir
from os.path import isfile, join
import pickle
import hashlib
import time

import numpy as np
import torch

PATH_BERT = '../../pytorch-pretrained-BERT'
sys.path.insert(0, PATH_BERT)
from pytorch_pretrained_bert import BertTokenizer, BertModel
from .encoder import BaseEncoder


class BERTEncoder(BaseEncoder):

    def __init__(self, model_name, encode_capacity=3000, PATH_CACHE='./cache'):
        super(BERTEncoder, self).__init__(model_name, encode_capacity, PATH_CACHE)

    def construct_encoder(self):
        model = BertModel.from_pretrained(self.model_name)
        model.cuda()
        model = torch.nn.DataParallel(model)
        model.eval()
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
                encoded_layers, _, self_attention_layers = self.model(input_ids, input_type_ids, input_mask)
                torch.cuda.synchronize()

                if location == 'fc':
                    output = np.array([layer[:, 0, :].detach().cpu().numpy() for layer in encoded_layers])
                elif location == 'head':
                    output = np.array([layer[:, 0, :].detach().cpu().numpy() for layer in self_attention_layers])

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
    model = BERTEncoder('bert-base-uncased')
    model.prepare('Length')
    model.construct_encoder()


