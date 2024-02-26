from models.base_model import ModelConfig
import replicate
from transformers import AutoTokenizer
import os
from models.utils.streamer import AsyncStreamer


class ReplicateConfig(ModelConfig):
    config_params = {
        'model_id': None,
        'tokenizer_id': None,
        'prompt_style': 'default',
        'model_author': 'replicate',
        'no_tokenizer': True,
    }
    gen_params = {
        "max_length": 1024,
        "seed": -1,
        "debug": False
    }
    env_params = {
        'replicate_api_token': ''
    }

    def __init__(self):
        self.no_tokenizer = self.config_params.pop('no_tokenizer', False)
        super().__init__(**self.config_params)
        replicate_api_token = self.env_params.get('replicate_api_token')
        if replicate_api_token is not None and len(replicate_api_token) > 0:
            os.environ['REPLICATE_API_TOKEN'] = replicate_api_token

    def create_stream(self, timeout=5):
        pass

    def generate(self, **kwargs):
        output = replicate.run(
            self.model_id,
            input=kwargs
        )
        output = AsyncStreamer(output)
        return output

    def is_cuda(self):
        return False

    def memory_footprint(self, verbose=False):
        memory = 0
        if verbose:
            print(f"Memory footprint: {memory:.2f} MB")
        return memory

    def load_model(self):
        if not self.no_tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_id,
                cache_dir=self.cache_dir,
            )
            self.max_tokens = self.tokenizer.model_max_length
        else:
            self.tokenizer = None
            self.max_tokens = 2048

    def tokenize(self, text, return_tensors='str'):
        if not self.no_tokenizer:
            return super().tokenize(text, return_tensors)
        return text

    def n_tokens(self, text):
        if not self.no_tokenizer:
            return super().n_tokens(text)
        return len(text)

    def preprocess_gen_kwargs(self, kwargs):
        kwargs = super().preprocess_gen_kwargs(kwargs)
        if 'max_new_tokens' in kwargs:
            kwargs["max_length"] = kwargs['max_new_tokens']
        return kwargs


class ReplicateVicunaConfig(ReplicateConfig):
    # https://replicate.com/replicate/vicuna-13b
    config_params = {
        'model_id': "replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b",
        'tokenizer_id': 'huggyllama/llama-7b',
        'prompt_style': 'vicuna',
        'no_tokenizer': True,
    }
    gen_params = {
        "max_length": 500,
        "temperature": 0.1,
        "top_p": 1,
        "repetition_penalty": 1,
        "seed": -1,
        "debug": False,
        'stop': ['</s>']
    }

    def __init__(self):
        super().__init__()


class ReplicateStablelmTunedAlpha7bConfig(ReplicateConfig):
    # https://huggingface.co/stabilityai/stablelm-tuned-alpha-7b
    # https://replicate.com/stability-ai/stablelm-tuned-alpha-7b
    config_params = {
        'model_id': "stability-ai/stablelm-tuned-alpha-7b:c49dae362cbaecd2ceabb5bd34fdb68413c4ff775111fea065d259d577757beb",
        'tokenizer_id': "StabilityAI/stablelm-tuned-alpha-7b",
        'prompt_style': "stablelm",
        'model_author': 'replicate',
        'no_tokenizer': True,
    }
    gen_params = {
        "max_length": 500,
        "temperature": 0.1,
        "top_p": 1,
        "repetition_penalty": 1.2,
        'stop': []
    }

    def __init__(self):
        super().__init__()


class ReplicateDolly12bConfig(ReplicateConfig):
    # https://huggingface.co/databricks/dolly-v2-12b
    # https://replicate.com/replicate/dolly-v2-12b
    config_params = {
        'model_id': "replicate/dolly-v2-12b:ef0e1aefc61f8e096ebe4db6b2bacc297daf2ef6899f0f7e001ec445893500e5",
        'tokenizer_id': "EleutherAI/gpt-neox-20b",
        'prompt_style': "dolly_v2",
        'model_author': 'replicate',
        'no_tokenizer': True,
    }
    gen_params = {
        "max_length": 500,
        "temperature": 0.1,
        "decoding": "top_p",  # in ["top_k", "top_p"]
        "top_p": 1,
        "top_k": 50,
        "repetition_penalty": 1.2,
        'stop': []
    }

    def __init__(self):
        super().__init__()


class ReplicateLlama2Config(ReplicateConfig):
    # https://replicate.com/replicate/llama-2-70b
    config_params = {
        'model_id': "replicate/llama-2-70b:14ce4448d5e7e9ed0c37745ac46eca157aab09061f0c179ac2b323b5de56552b",
        'tokenizer_id': 'huggyllama/llama-7b',
        'prompt_style': "llama",
        'model_author': 'replicate',
        'no_tokenizer': True,
    }
    gen_params = {
        "max_length": 500,
        "temperature": 0.1,
        "top_p": 1,
        'stop': []
    }

    def __init__(self):
        super().__init__()


class ReplicateLlama2ChatConfig(ReplicateConfig):
    # https://replicate.com/replicate/llama-2-70b-chat
    config_params = {
        'model_id': "replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1",
        'tokenizer_id': 'huggyllama/llama-7b',
        'prompt_style': "llama",
        'model_author': 'replicate',
        'no_tokenizer': True,
    }
    gen_params = {
        "max_length": 500,
        "temperature": 0.75,
        'stop': [],
        'system_prompt string': '',
        'min_new_tokens': -1,
        "top_p": 0.95,
        "top_k": 250,
        "repetition_penalty": 1.15,
        'repetition_penalty_sustain': 250,
        'token_repetition_penalty_decay integer': 128,
    }

    def __init__(self):
        super().__init__()


class ReplicateCodeLlama13bConfig(ReplicateConfig):
    # not used because it is only a completion mode, does not support chat/instruction mode
    # https://replicate.com/replicate/codellama-13b
    # https://github.com/facebookresearch/codellama
    config_params = {
        'model_id': "replicate/codellama-13b:1c914d844307b0588599b8393480a3ba917b660c7e9dfae681542b5325f228db",
        'tokenizer_id': 'huggyllama/llama-7b',
        'prompt_style': "only_question",
        'model_author': 'replicate',
        'no_tokenizer': True,
    }
    gen_params = {
        "max_new_tokens": 1024,
        "min_new_tokens": 1,
        'stop': [],
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.15,
        'repetition_penalty_sustain': 250,
        'token_repetition_penalty_decay integer': 128,
    }

    def __init__(self):
        super().__init__()


class ReplicateCodeLLama13bInstructConfig(ReplicateConfig):
    # https://replicate.com/replicate/codellama-13b-instruct
    # https://github.com/facebookresearch/codellama
    config_params = {
        'model_id': "replicate/codellama-13b-instruct:da5676342de1a5a335b848383af297f592b816b950a43d251a0a9edd0113604b",
        'tokenizer_id': 'huggyllama/llama-7b',
        'prompt_style': "llama",
        'model_author': 'replicate',
        'no_tokenizer': True,
    }
    gen_params = {
        "max_length": 500,
        "temperature": 0.95,
        'stop': [],
        'system_prompt string': '',
        'min_new_tokens': -1,
        "top_p": 0.95,
        "top_k": 250,
        "repetition_penalty": 1.15,
        'repetition_penalty_sustain': 250,
        'token_repetition_penalty_decay integer': 128,
    }

    def __init__(self):
        super().__init__()
