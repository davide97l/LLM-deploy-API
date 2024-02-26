import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from typing import Optional
from models.utils.streamer import AsyncStreamer


from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


class ModelConfig:
    def __init__(self,
                 model_id: str,
                 tokenizer_id: Optional[str] = None,
                 model_author: Optional[str] = None,
                 model_type: str = 'causal',
                 dtype: Optional[torch.dtype] = torch.float16,
                 load_in_8bit: Optional[bool] = False,
                 trust_remote_code: Optional[bool] = True,
                 device_map: Optional[str] = 'auto',
                 requires_gpu: Optional[bool] = False,
                 cache_dir: Optional[str] = 'llm/',
                 prompt_style: Optional[str] = None,):
        """
        :param model_id: model identifier
        :param tokenizer_id: tokenizer identifier or None, in which case the model_id is used
        :param model_type: model type, choose from 'causal' or 'seq2seq'
        :param dtype: data type
        :param load_in_8bit: whether to load the model in 8-bit
        :param trust_remote_code: whether to trust the remote code
        :param device_map: device map string
        :param requires_gpu: whether the model requires a GPU
        :param cache_dir: path to the folder where the model should be saved
        """
        self.model_id = model_id
        self.tokenizer_id = tokenizer_id if tokenizer_id is not None else model_id
        self.model_type = model_type
        self.model_name = model_id.split('/')[-1]
        self.model_author = model_id.split('/')[0] if model_author is None else model_author
        self.dtype = dtype
        self.load_in_8bit = load_in_8bit
        self.trust_remote_code = trust_remote_code
        self.device_map = device_map
        self.requires_gpu = requires_gpu
        self.cache_dir = cache_dir
        self.prompt_style = prompt_style
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.requires_gpu:
            assert self.device == 'cuda'
        self.load_model()

    def load_model(self):
        if self.model_type == 'causal':
            model_class = AutoModelForCausalLM
        elif self.model_type == 'seq2seq':
            model_class = AutoModelForSeq2SeqLM
        else:
            raise ValueError("Invalid model type provided. Choose from 'causal' or 'seq2seq'.")

        self.model = model_class.from_pretrained(
            self.model_id,
            device_map=self.device_map,
            load_in_8bit=self.load_in_8bit,
            torch_dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            cache_dir=self.cache_dir,
        )
        if not self.requires_gpu and self.device == 'cuda':
            self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_id,
            cache_dir=self.cache_dir,
        )
        self.max_tokens = self.tokenizer.model_max_length

    def tokenize(self, text, return_tensors='pt'):
        return self.tokenizer(text, return_tensors=return_tensors).to(self.device)

    def n_tokens(self, text):
        return len(self.tokenize(text)['input_ids'][0])

    def create_stream(self, timeout=5, make_async=True):
        streamer = TextIteratorStreamer(self.tokenizer, timeout=timeout, skip_prompt=True, skip_special_tokens=True)
        if make_async:
            streamer = AsyncStreamer(streamer)
        return streamer

    def preprocess_gen_kwargs(self, kwargs):
        if "token_type_ids" in kwargs and not hasattr(self.model.config, "use_token_type_ids"):
            # to avoid error - ValueError: The following model_kwargs are not used by the model: ['token_type_ids']
            kwargs.pop("token_type_ids")
        if kwargs.get('temperature', None):
            kwargs['temperature'] = max(0.01, min(2.0, float(kwargs['temperature'])))
        return kwargs

    def memory_footprint(self, verbose=False):
        memory = self.model.get_memory_footprint() / 1e6
        if verbose:
            print(f"Memory footprint: {memory:.2f} MB")
        return memory

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

    def is_cuda(self):
        if self.model.device.index is not None and self.model.device.type == 'cuda':
            return True
        return False
