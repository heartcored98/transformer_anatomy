
import torch
from torch import nn

class BaseExtractor(nn.Module):

    def __init__(self, model, location, pooling_position):
        super(BaseExtractor, self).__init__()

        self.is_output_hidden_states(model)
        self.model = self.override_forward(model)
        self.location = location
        self.pooling_position = pooling_position

        self.head_size = int(self.model.config.hidden_size / self.model.config.num_attention_heads)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def override_forward(self, model):
        raise NotImplementedError
        
    def set_location(self, location):
        self.location = location

    def set_pooling_position(self, pooling_position):
        self.pooling_position = pooling_position

    def get_location(self):
        return self.location
    
    def get_pooling_position(self):
        return self.pooling_position

    def is_output_hidden_states(self, model):
        if model.config.output_hidden_states:
            return
        raise ValueError("model should be created with 'output_hidden_states' option as 'True'. ")
        
        if model.config.output_attentions:
            return
        raise ValueError("model should be created with 'output_attentions' option as 'True'. ")

    def get_head_embedding(self, all_head_states, layer, head):
        if head < 0:
            raise ValueError(f"head={head} should be greater or equal to 0")
        head_embedding = all_head_states[layer][:, 0, head * self.head_size:(head + 1) * self.head_size]
        return head_embedding

    def get_multi_head_embedding(self, all_head_states):
        """
        all_head_states : L x (BS x LEN x H)
        """

        pooling_position = self.pooling_position
        if len(pooling_position) == 1: # If single attention head is probed
            layer, head = pooling_position[0]
            head_embedding = self.get_head_embedding(all_head_states, layer, head)
            return head_embedding
        else: # If multiple attention head is selected
            list_embedding = []
            for layer, head in pooling_position:
                head_embedding = self.get_head_embedding(all_head_states, layer, head)
                list_embedding.append(head_embedding)
            multi_head_embedding = torch.cat(list_embedding, axis=1)
            return multi_head_embedding

    def get_multi_layer_embedding(self, all_hidden_states):
        """
        all_hidden_states : L x (BS x LEN x H)
        """
        layers = self.pooling_position
        if len(layers) == 1:
            layer = layers[0]
            layer_embedding = all_hidden_states[layer][:, 0, :] # return first token's representation
            return layer_embedding
        else:
            raise NotImplementedError("Currently, only single layer supported")

    def extract_embedding(self, all_hidden_states, all_head_states):
        if self.location == 'last' or self.location == 'best':
            return self.get_multi_layer_embedding(all_hidden_states)
        elif self.location == 'head':
            return self.get_multi_head_embedding(all_head_states)
        else:
            return all_hidden_states, all_head_states