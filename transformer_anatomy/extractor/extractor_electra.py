import types

from .extractor_base import BaseExtractor
from .extractor_bert import bert_layer_forward, bert_attention_forward, bert_encoder_forward


class ElectraExtractor(BaseExtractor):

    def __init__(self, model, location, heads):
        super(ElectraExtractor, self).__init__(model, location, heads)

    def __call__(self, *args, **kwargs):
        last_hidden_state, all_hidden_states, all_attentions, all_head_states = self.model(*args, **kwargs)
        return self.extract_embedding(all_hidden_states, all_head_states)

    def override_forward(self, model):
        model.encoder.forward = types.MethodType(bert_encoder_forward, model.encoder)
        for layer in model.encoder.layer:
            layer.forward = types.MethodType(bert_layer_forward, layer)
            layer.attention.forward = types.MethodType(bert_attention_forward, layer.attention)
        return model

    
if __name__ == '__main__':
    import torch
    from transformers import ElectraModel, ElectraConfig, ElectraTokenizer

    pretrained_weights = 'google/electra-small-discriminator'

    # Initializing a model from the electra-base-uncased style configuration
    model = ElectraModel.from_pretrained(pretrained_weights, output_hidden_states=True, output_attentions=True)
    tokenizer = ElectraTokenizer.from_pretrained(pretrained_weights)
    input_ids = torch.tensor([tokenizer.encode("Let's see all hidden-states and attentions on this text")])

    new_model = ElectraExtractor.from_model(model)
    print(type(new_model))
    output = new_model(input_ids)
    print(len(output))

