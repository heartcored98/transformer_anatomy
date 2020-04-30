from .extractor_electra import ElectraExtractor
from .extractor_bert import BertExtractor

EXTRACTOR_ARCHIVE_MAP = {
    'ElectraModel': ElectraExtractor,
    'BertModel': BertExtractor
}

class AutoExtractor():

    @classmethod
    def from_model(cls, model, location=None, heads=None):
        model_name = type(model).__name__
        if model_name not in EXTRACTOR_ARCHIVE_MAP.keys():
            raise ValueError(f"{model_name} is not registered model!")
        return EXTRACTOR_ARCHIVE_MAP[model_name](model, location, heads)


    

