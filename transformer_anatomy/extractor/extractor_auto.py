from .extractor_electra import ElectraExtractor

class AutoExtractor():

    @classmethod
    def from_model(cls, model):
        model_name = type(model).__name__
        if model_name == 'ElectraModel':
            return ElectraExtractor(model)
        else:
            raise ValueError(f"{model_name} is not registered model!")


if __name__ == '__main__':
    import torch
    from transformers import ElectraModel, ElectraConfig, ElectraTokenizer

    # Initializing a ELECTRA electra-base-uncased style configuration
    pretrained_weights = 'google/electra-small-discriminator'

    # Initializing a model from the electra-base-uncased style configuration
    model = ElectraModel.from_pretrained(pretrained_weights, output_hidden_states=True, output_attentions=True)
    tokenizer = ElectraTokenizer.from_pretrained(pretrained_weights)
    input_ids = torch.tensor([tokenizer.encode("Let's see all hidden-states and attentions on this text")])
    outputs = model(input_ids)
    print(len(outputs))
    for i, o in enumerate(outputs):
        print(i, len(o), type(o))
    print(len(all_hidden_states))
    print(all_hidden_states[0].shape)

    print("==== new model ===")
    new_model = AutoExtractor.from_model(model)
    print(type(new_model))
    print(new_model.location)

    input_ids = torch.tensor([tokenizer.encode("Let's see all hidden-states and attentions on this text")])
    output = new_model(input_ids)
    all_head_states = output[-1]
    print(len(all_head_states))
    print(all_head_states[0].shape)


    

