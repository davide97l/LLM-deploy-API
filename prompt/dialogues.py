import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from huggingface_hub import ModelHubMixin, hf_hub_download

# Generic variable that is either ModelHubMixin or a subclass thereof
T = TypeVar("T", bound="ModelHubMixin")
TEMPLATE_FILENAME = "dialogue_template.json"
IGNORE_INDEX = -100


@dataclass
class DialogueTemplate(ModelHubMixin):
    """Converts all turns of a dialogue between a user and assistant to a standardized format.
    Adapted from OpenAI's ChatML (https://github.com/openai/openai-python/blob/main/chatml.md) and Vicuna (https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py)
    """
    system: str
    messages: List[Dict[str, str]] = None
    system_token: str = "system"
    user_token: str = "user"
    assistant_token: str = "assistant"
    end_token: str = ""  # token that signals end of message
    role_sep_token: str = ": "  # token that separates user_token and message
    user_sep_token: str = "\n"  # token that separates user_token and message
    assistant_sep_token: str = "\n"  # token that separates end_token and next user_token or assistant_token
    system_sep_token: str = "\n"  # token that separates end_token and next user_token or assistant_token

    def get_training_prompt(self) -> str:
        prompt = ''
        if self.system is not None and len(self.system) > 0:
            if len(self.system_token) > 0:
                prompt = self.system_token + self.role_sep_token + self.system + self.end_token + self.system_sep_token
            else:
                prompt = self.system + self.end_token + self.system_sep_token
        if self.messages is None:
            raise ValueError("Dialogue template must have at least one message.")
        for message in self.messages:
            if message["role"] == "user":
                prompt += self.user_token + self.role_sep_token + message["content"] + self.end_token + self.user_sep_token
            else:
                prompt += self.assistant_token + self.role_sep_token + message["content"] + self.end_token + self.assistant_sep_token
        return prompt

    def get_inference_prompt(self) -> str:
        prompt = ''
        if self.system is not None and len(self.system) > 0:
            if len(self.system_token) > 0:
                prompt = self.system_token + self.role_sep_token + self.system + self.end_token + self.system_sep_token
            else:
                prompt = self.system + self.end_token + self.system_sep_token
        if self.messages is None:
            raise ValueError("Dialogue template must have at least one message.")
        for i, message in enumerate(self.messages):
            if i == len(self.messages) - 1 and message["role"] == "assistant" and len(message["content"]) < 1:
                break
            if message["role"] == "user":
                prompt += self.user_token + self.role_sep_token + message["content"] + self.end_token + self.user_sep_token
            elif message["role"] == "assistant":
                prompt += self.assistant_token + self.role_sep_token + message["content"] + self.end_token + self.assistant_sep_token
        if self.assistant_token is not None and len(self.assistant_token) > 0:
            prompt += self.assistant_token + self.role_sep_token.rstrip(' ')  # append assistant: <empty_message> at the end
        return prompt

    def get_dialogue(self):
        """Helper function to format the messages as an easy-to-read dialogue."""
        prompt = ""
        if self.messages is None:
            raise ValueError("Dialogue template must have at least one message.")
        for message in self.messages:
            if message["role"] == "user":
                prompt += "\n\nHuman: " + message["content"]
            else:
                prompt += "\n\nAssistant: " + message["content"]
        return prompt

    def get_special_tokens(self) -> List[str]:
        return [self.system_token, self.user_token, self.assistant_token, self.end_token]

    def copy(self):
        return DialogueTemplate(
            system=self.system,
            messages=self.messages,
            system_token=self.system_token,
            user_token=self.user_token,
            assistant_token=self.assistant_token,
            end_token=self.end_token,
            role_sep_token=self.role_sep_token,
            assistant_sep_token=self.assistant_sep_token,
            system_sep_token=self.system_sep_token,
            user_sep_token=self.user_sep_token,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data):
        return DialogueTemplate(
            system=data["system"] if "system" in data else "",
            messages=data["messages"] if "messages" in data else None,
            system_token=data["system_token"] if "system_token" in data else "<|system|>",
            user_token=data["user_token"] if "user_token" in data else "<|user|>",
            assistant_token=data["assistant_token"] if "assistant_token" in data else "<|assistant|>",
            end_token=data["end_token"] if "end_token" in data else "<|end|>",
        )

    def _save_pretrained(self, save_directory: Union[str, Path]) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(exist_ok=True)
        with open(save_directory / "dialogue_template.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def _from_pretrained(
            cls: Type[T],
            *,
            model_id: str,
            revision: Optional[str],
            cache_dir: Optional[Union[str, Path]],
            force_download: bool,
            proxies: Optional[Dict],
            resume_download: bool,
            local_files_only: bool,
            token: Optional[Union[str, bool]],
            **model_kwargs,
    ) -> T:
        """Loads the dialogue template from a local directory or the Huggingface Hub.
        Args:
            model_id (`str`):
                ID of the model to load from the Huggingface Hub (e.g. `bigscience/bloom`).
            revision (`str`, *optional*):
                Revision of the model on the Hub. Can be a branch name, a git tag or any commit id. Defaults to the
                latest commit on `main` branch.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether to force (re-)downloading the model weights and configuration files from the Hub, overriding
                the existing cache.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether to delete incompletely received files. Will attempt to resume the download if such a file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint (e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`).
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. By default, it will use the token
                cached when running `huggingface-cli login`.
            cache_dir (`str`, `Path`, *optional*):
                Path to the folder where cached files are stored.
            local_files_only (`bool`, *optional*, defaults to `False`):
                If `True`, avoid downloading the file and return the path to the local cached file if it exists.
            model_kwargs:
                Additional keyword arguments passed along to the [`~ModelHubMixin._from_pretrained`] method.
        """
        if os.path.isdir(model_id):  # Can either be a local directory
            print("Loading dialogue template from local directory")
            template_file = os.path.join(model_id, TEMPLATE_FILENAME)
        else:  # Or a template on the Hub
            template_file = hf_hub_download(  # Download from the hub, passing same input args
                repo_id=model_id,
                filename=TEMPLATE_FILENAME,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )

        # Load template
        with open(template_file, "r") as f:
            data = json.load(f)
        return cls.from_dict(data=data)


# OpenAI prompt template
default_template = DialogueTemplate(
    system="You're a helpful assistant",
    system_token="system",
    user_token="user",
    assistant_token="assistant",
)

starcoder_template = DialogueTemplate(
    system="Below is a dialogue between a human user and an AI assistant. The assistant is happy to help with almost anything, and will do its best to understand exactly what is needed.",
    system_token="<|system|>",
    user_token="<|user|>",
    assistant_token="<|assistant|>",
    end_token="<|end|>",
    role_sep_token="\n",
    user_sep_token="\n",
    assistant_sep_token="\n",
    system_sep_token="\n",
)

# OpenAI template with no system messages.
no_system_template = DialogueTemplate(
    system="",
    system_token="",
    user_token="user",
    assistant_token="assistant",
)

# Only messages
only_question = DialogueTemplate(
    system="",
    system_token="",
    user_token="",
    assistant_token="",
    role_sep_token="",
    user_sep_token="\n",
    assistant_sep_token="\n",
    system_sep_token="\n",
)

falcon_template = DialogueTemplate(
    system_token="",
    system="The following is a conversation between a highly knowledgeable and intelligent AI assistant, called Falcon, and a human user, called User. In the following interactions, User and Falcon will converse in natural language, and Falcon will answer User's questions. Falcon was built to be respectful, polite and inclusive. Falcon was built by the Technology Innovation Institute in Abu Dhabi. Falcon will never decline to answer a question, and always attempts to give an answer that User would be satisfied with. It knows a lot, and always tells the truth. The conversation begins.",
    user_token="User",
    assistant_token="Assistant",
    role_sep_token=": ",
    user_sep_token="\n\n",
    assistant_sep_token="\n\n",
    system_sep_token="\n\n",
)

vicuna_template = DialogueTemplate(
    system="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
    system_token="",
    user_token="USER",
    assistant_token="ASSISTANT",
    end_token="",
    role_sep_token=": ",
    user_sep_token=" ",
    assistant_sep_token="</s>",
    system_sep_token=" ",
)


stablelm_template = DialogueTemplate(
    system="""# StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
""",
    system_token="<|SYSTEM|>",
    user_token="<|USER|>",
    assistant_token="<|ASSISTANT|>",
    end_token="",
    role_sep_token="",
    user_sep_token="",
    assistant_sep_token="",
    system_sep_token="",
)

dolly_v2_template = DialogueTemplate(
    system="Below is an instruction that describes a task. Write a response that appropriately completes the request.",
    system_token="",
    user_token="### Instruction",
    assistant_token="### Response",
    end_token="",
    role_sep_token=":\n",
    user_sep_token="\n\n",
    assistant_sep_token="### End\n\n",
    system_sep_token="\n\n",
)

llama_template = DialogueTemplate(
    system="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. 
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
If you don't know the answer to a question, please don't share false information.""",
    system_token="",
    user_token="[INST] ",
    assistant_token="",
    end_token="",
    role_sep_token="",
    user_sep_token=" [/INST]\n",
    assistant_sep_token="\n",
    system_sep_token="\n\n",
)


SUPPORTED_DIALOGUE_TEMPLATES = {
    "default": default_template,
    "only_question": only_question,
    "no_system": no_system_template,
    "falcon": falcon_template,
    "starcoder": starcoder_template,
    "vicuna": vicuna_template,
    "stablelm": stablelm_template,
    "dolly_v2": dolly_v2_template,
    "llama": llama_template,
}


'''
## usage example
messages = {'messages': [{"role": "user", "content": "capital of france?"}]}
dialogue_template = get_dialogue_template('starcoder')
prepare_dialogue(messages, dialogue_template, False)
prompt = messages['text']
print(prompt)
#print(dialogue_template.get_dialogue())
#print(dialogue_template.get_special_tokens())
'''
