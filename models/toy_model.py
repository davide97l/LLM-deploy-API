from models.base_model import ModelConfig
import torch
from models.utils.streamer import AsyncStreamer, StringStreamer


class ToyModelConfig(ModelConfig):
    config_params = {
        'model_id': 'test/toy-model',
        'dtype': torch.bfloat16,
        'load_in_8bit': False,
        'requires_gpu': False,
        'prompt_style': 'default',
    }
    gen_params = {
    }

    def __init__(self):
        super().__init__(**self.config_params)

    def load_model(self):
        self.model = None
        self.tokenizer = None
        self.max_tokens = 10000

    def tokenize(self, text, return_tensors='str'):
        return text

    def n_tokens(self, text):
        return 0

    def memory_footprint(self, verbose=False):
        return 0

    def generate(self, **kwargs):
        output = 'Hi, this is a test. Can you see this message?'
        streamer = StringStreamer(output)
        streamer = AsyncStreamer(streamer.generate())
        return streamer

    def is_cuda(self):
        return False
