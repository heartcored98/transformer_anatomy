import types


from .extractor_base import BaseExtractor

def bert_attention_forward(
    self,
    hidden_states,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
):
    self_outputs = self.self(
        hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
    )
    attention_output = self.output(self_outputs[0], hidden_states)
    outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
    head_states = self_outputs[0] # Add. Extract head states
    return outputs, head_states


def bert_layer_forward(
    self,
    hidden_states,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
):
    self_attention_outputs, head_states = self.attention(hidden_states, attention_mask, head_mask)
    attention_output = self_attention_outputs[0]
    outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

    if self.is_decoder and encoder_hidden_states is not None:
        cross_attention_outputs = self.crossattention(
            attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
        )
        attention_output = cross_attention_outputs[0]
        outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

    intermediate_output = self.intermediate(attention_output)
    layer_output = self.output(intermediate_output, attention_output)
    outputs = (layer_output,) + outputs + (head_states,)
    return outputs


def bert_encoder_forward(
    self,
    hidden_states,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
):
    all_hidden_states = ()
    all_attentions = ()
    all_head_states = ()

    for i, layer_module in enumerate(self.layer):
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer_outputs = layer_module(
            hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask
        )
        hidden_states = layer_outputs[0]
        head_states = layer_outputs[-1]

        if self.output_attentions:
            all_attentions = all_attentions + (layer_outputs[1],)
        all_head_states = all_head_states + (head_states,)

    # Add last layer
    if self.output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    outputs = (hidden_states,)
    if self.output_hidden_states:
        outputs = outputs + (all_hidden_states,)
    if self.output_attentions:
        outputs = outputs + (all_attentions,)
    outputs = outputs + (all_head_states,)
    return outputs  # last-layer hidden state, (all hidden states), (all attentions), (all head states)


class BertExtractor(BaseExtractor):

    def __init__(self, model, location, heads):
        super(BertExtractor, self).__init__(model, location, heads)

    def __call__(self, *args, **kwargs):
        sequence_output, pooled_output, all_hidden_states, all_attentions, all_head_states = self.model(*args, **kwargs)
        return self.extract_embedding(all_hidden_states, all_head_states)

    def override_forward(self, model):
        model.encoder.forward = types.MethodType(bert_encoder_forward, model.encoder)
        for layer in model.encoder.layer:
            layer.forward = types.MethodType(bert_layer_forward, layer)
            layer.attention.forward = types.MethodType(bert_attention_forward, layer.attention)
        return model
    
