from models.base_model import ModelConfig
import torch


class FlanT5SmallConfig(ModelConfig):
    config_params = {
        'model_id': 'google/flan-t5-small',
        'dtype': torch.bfloat16,
        'load_in_8bit': False,
        'requires_gpu': False,
        'model_type': 'seq2seq',
        'prompt_style': 'default',
    }
    gen_params = {
        'temperature': 0.9,
        'max_new_tokens': 512,
        'top_k': 50,
        'top_p': 0.4,
        'repetition_penalty': 1.0,
    }

    def __init__(self):
        super().__init__(**self.config_params)


class FlanT5XLConfig(FlanT5SmallConfig):
    config_params = {
        'model_id': 'google/flan-t5-xl',
        'dtype': torch.float16,
        'load_in_8bit': False,
        'requires_gpu': False,
        'model_type': 'seq2seq',
        'prompt_style': 'default',
    }

    def __init__(self):
        super().__init__()
