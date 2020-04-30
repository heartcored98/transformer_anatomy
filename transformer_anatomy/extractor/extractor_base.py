
class BaseExtractor():

    def __init__(self, model, location, heads):
        self.is_output_hidden_states(model)
        self.model = self.override_forward(model)
        self.location = location
        self.heads = heads

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def override_forward(self, model):
        raise NotImplementedError
        
    def set_location(self, location):
        self.location = location

    def set_heads(self, heads):
        self.heads = heads

    def get_location(self):
        return self.location
    
    def get_heads(self):
        return self.heads

    def is_output_hidden_states(self, model):
        if model.config.output_hidden_states:
            return
        raise ValueError("model should be created with 'output_hidden_states' option as 'True'. ")
        
        if model.config.output_attentions:
            return
        raise ValueError("model should be created with 'output_attentions' option as 'True'. ")
