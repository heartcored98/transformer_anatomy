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
    model = AutoExtractor.from_model(model)
    print(type(model))

    # assert model==ElectraExtractor
    input_ids = torch.tensor([tokenizer.encode("Let's see all hidden-states and attentions on this text")])
    all_hidden_states, all_head_states = model(input_ids)
    assert len(all_hidden_states) == 13, "number of layer does not match"
    assert all_hidden_states[0].shape == torch.Size([1, 16, 256]), "0-th layer output does not match shape"
    print(len(all_hidden_states), all_hidden_states[0].shape)
    print(len(all_head_states), all_head_states[0].shape)


    print(f"==== Pooling from Single Layer ====")
    model.set_location('layer')
    model.set_pooling_position([3])
    layer_embedding = model(input_ids)
    print(layer_embedding.shape)

    print(f"==== Pooling from Single Head ====")
    model.set_location('head')
    model.set_pooling_position([(3, 2)])
    head_embedding = model(input_ids)
    print(head_embedding.shape)

    print(f"==== Pooling from Multi Head ====")
    model.set_location('head')
    model.set_pooling_position([(3, 2), (4,2), (11,2)])
    multi_head_embedding = model(input_ids)
    print(multi_head_embedding.shape)

    model_name = 'bert-base-uncased'
    print(f"==== Testing model={model_name} ====")

    model = BertModel.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = AutoExtractor.from_model(model)
    print(type(model))

    # assert model==ElectraExtractor
    input_ids = torch.tensor([tokenizer.encode("Let's see all hidden-states and attentions on this text")])
    all_hidden_states, all_head_states = model(input_ids)
    print(len(all_hidden_states), all_hidden_states[0].shape)
    print(len(all_head_states), all_head_states[0].shape)