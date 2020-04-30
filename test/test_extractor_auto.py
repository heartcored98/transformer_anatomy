import sys
sys.path.insert(0, '../')

import torch
import transformer_anatomy
from transformers import ElectraModel, ElectraTokenizer, BertModel, BertTokenizer 
from transformer_anatomy.extractor import AutoExtractor, ElectraExtractor



if __name__ == '__main__':

    model_name = 'google/electra-small-discriminator'
    print(f"==== Testing model={model_name} ====")

    model = ElectraModel.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
    tokenizer = ElectraTokenizer.from_pretrained(model_name)
    new_model = AutoExtractor.from_model(model)
    print(type(new_model))

    # assert new_model==ElectraExtractor
    input_ids = torch.tensor([tokenizer.encode("Let's see all hidden-states and attentions on this text")])
    all_hidden_states, all_head_states = new_model(input_ids)
    print(len(all_hidden_states), all_hidden_states[0].shape)
    print(len(all_head_states), all_head_states[0].shape)


    model_name = 'bert-base-uncased'
    print(f"==== Testing model={model_name} ====")

    model = BertModel.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    new_model = AutoExtractor.from_model(model)
    print(type(new_model))

    # assert new_model==ElectraExtractor
    input_ids = torch.tensor([tokenizer.encode("Let's see all hidden-states and attentions on this text")])
    all_hidden_states, all_head_states = new_model(input_ids)
    print(len(all_hidden_states), all_hidden_states[0].shape)
    print(len(all_head_states), all_head_states[0].shape)