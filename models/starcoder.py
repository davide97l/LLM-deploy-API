from models.base_model import ModelConfig
import torch


class StarcoderConfig(ModelConfig):
    config_params = {
        'model_id': "bigcode/starcoder",
        'dtype': torch.bfloat16,
        'load_in_8bit': True,
        'requires_gpu': True,
        'prompt_style': 'starcoder',
    }
    gen_params = {
        'temperature': 0.2,
        'top_k': 50,
        'top_p': 0.95,
        'repetition_penalty': 1.2,
        'do_sample': True,
        'min_new_tokens': 32,
        'max_new_tokens': 256,
    }

    def __init__(self):
        super().__init__(**self.config_params)
        self.gen_params['pad_token_id'] = self.tokenizer.eos_token_id  # https://jaketae.github.io/study/gpt2/#setup


class StarCoderPlusConfig(StarcoderConfig):
    config_params = {
        'model_id': "bigcode/starcoderplus",
        'dtype': torch.bfloat16,
        'load_in_8bit': True,
        'requires_gpu': True,
        'prompt_style': 'starcoder',
    }

    def __init__(self):
        super().__init__()


class StarChatConfig(StarcoderConfig):
    config_params = {
        'model_id': "HuggingFaceH4/starchat-alpha",
        'dtype': torch.bfloat16,
        'load_in_8bit': True,
        'requires_gpu': True,
        'prompt_style': 'starcoder',
    }
    gen_params = {
        'temperature': 0.2,
        'top_k': 50,
        'top_p': 0.95,
        'repetition_penalty': 1.2,
        'do_sample': True,
        'min_new_tokens': 32,
        'max_new_tokens': 1024,
        'eos_token_id': 49155  # "<|end|>"
    }

    def __init__(self):
        super().__init__()
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|system|>", "<|assistant|>", "<|user|>", "<|end|>"]})
