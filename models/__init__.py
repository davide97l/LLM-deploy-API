from dotenv import load_dotenv
load_dotenv()  # load .env file
from models.base_model import ModelConfig
from models.falcon import Falcon7BConfig, Falcon40BConfig
from models.flanT5 import FlanT5SmallConfig, FlanT5XLConfig
from models.starcoder import StarcoderConfig, StarCoderPlusConfig, StarChatConfig
from models.replicate import ReplicateVicunaConfig, ReplicateStablelmTunedAlpha7bConfig, ReplicateDolly12bConfig,\
    ReplicateLlama2ChatConfig, ReplicateLlama2Config, ReplicateCodeLLama13bInstructConfig
from models.toy_model import ToyModelConfig

__all__ = ['ModelConfig', 'Falcon7BConfig']

config_dict = {
    'tiiuae/falcon-7b-instruct': lambda: Falcon7BConfig(),
    'tiiuae/falcon-40b-instruct': lambda: Falcon40BConfig(),
    'google/flan-t5-small': lambda: FlanT5SmallConfig(),
    'google/flan-t5-xl': lambda: FlanT5XLConfig(),
    'bigcode/starcoder': lambda: StarcoderConfig(),
    'bigcode/starcoderplus': lambda: StarCoderPlusConfig(),
    'HuggingFaceH4/starchat-alpha': lambda: StarChatConfig(),
    'replicate/vicuna-13b': lambda: ReplicateVicunaConfig(),
    'replicate/stablelm-tuned-alpha-7b': lambda: ReplicateStablelmTunedAlpha7bConfig(),
    'replicate/dolly-v2-12b': lambda: ReplicateDolly12bConfig(),
    'replicate/llama2-70b': lambda: ReplicateLlama2Config(),
    'replicate/llama-2-70b-chat': lambda: ReplicateLlama2ChatConfig(),
    'replicate/codellama-13b-instruct': lambda: ReplicateCodeLLama13bInstructConfig(),
    'test/toy-model': lambda: ToyModelConfig(),
}
