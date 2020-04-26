from extractor_base import BaseExtractor

class ElectraBert(BaseExtractor):

    def __init__(self, model, location, heads):
        super(ElectraBert, self).__init__(model, location, heads)
        self.model = self.override_forward(model)

    def override_forward(self, model):
        return model

    
