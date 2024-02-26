from models.base_model import ModelConfig
import torch


class Falcon7BConfig(ModelConfig):
    config_params = {
        'model_id': 'tiiuae/falcon-7b-instruct',
        'dtype': torch.bfloat16,
        'load_in_8bit': True,
        'requires_gpu': True,
        'prompt_style': 'falcon',
    }
    gen_params = {
        'do_sample': True,
        'top_k': 10,
        'stop': ['\nUser:', "<|endoftext|>"]
    }

    def __init__(self):
        super().__init__(**self.config_params)
        self.gen_params['pad_token_id'] = self.tokenizer.eos_token_id  # https://jaketae.github.io/study/gpt2/#setup


class Falcon40BConfig(Falcon7BConfig):
    config_params = {
        'model_id': 'tiiuae/falcon-40b-instruct',
        'dtype': torch.bfloat16,
        'load_in_8bit': True,
        'requires_gpu': True,
        'prompt_style': 'falcon',
        'stop_words': ['\nUser:', "<|endoftext|>"]
    }

    def __init__(self):
        super().__init__()
