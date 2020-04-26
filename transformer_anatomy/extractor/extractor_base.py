
class BaseExtractor():

    def __init__(self, model):
        self.model = model

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

    @classmethod
    def is_output_hidden_states(cls, model):
        if model.config.output_hidden_states:
            return
        raise ValueError("model should be created with 'output_hidden_states' option as 'True'. ")
