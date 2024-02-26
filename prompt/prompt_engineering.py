from prompt.dialogues import DialogueTemplate, SUPPORTED_DIALOGUE_TEMPLATES, IGNORE_INDEX
'''
Input example
    {"role": "system", "content": "You're a helpful assistant"},
    {"role": "user", "content": "I ate an apple and a watermelon with my friend."},
    {"role": "assistant", "content": "good."},
    {"role": "user", "content": "what did I eat?"},
'''


def make_chat_history(messages, prompt_style='default'):
    dialogue_template = get_dialogue_template(prompt_style)
    messages_dict = {'messages': messages}
    prepare_dialogue(messages_dict, dialogue_template, False)
    prompt = messages_dict['text']
    return prompt


def get_dialogue_template(template: str) -> DialogueTemplate:
    if template is None:
        template = 'default'
    if template not in SUPPORTED_DIALOGUE_TEMPLATES.keys():
        raise ValueError(f"Template {template} is not supported!")
    return SUPPORTED_DIALOGUE_TEMPLATES[template].copy()


def prepare_dialogue(example, dialogue_template, is_train=True):
    """Format example to single- or multi-turn dialogue."""
    if "messages" in example.keys() and example["messages"] is not None:
        dialogue_template.messages = example["messages"]
        if example["messages"][0]['role'] == 'system':
            dialogue_template.system = example["messages"][0]['content']
    elif all(k in example.keys() for k in ("prompt", "completion")):
        # Construct single-turn dialogue from prompt and completion
        dialogue_template.messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]},
        ]
    elif "prompt" in example.keys():
        # Construct single-turn dialogue from prompt (inference only)
        dialogue_template.messages = [
            {"role": "user", "content": example["prompt"]},
        ]
    else:
        raise ValueError(
            f"Could not format example as dialogue! Require either `messages` or `[prompt, completion]` or `[prompt]` keys but found {list(example.keys())}"
        )
    if is_train:
        example["text"] = dialogue_template.get_training_prompt()
    else:
        example["text"] = dialogue_template.get_inference_prompt()
    return example


def mask_user_labels(tokenizer, dialogue_template, labels):
    """Masks the user turns of a dialogue from the loss"""
    user_token_id = tokenizer.convert_tokens_to_ids(dialogue_template.user_token)
    assistant_token_id = tokenizer.convert_tokens_to_ids(dialogue_template.assistant_token)
    for idx, label_id in enumerate(labels):
        if label_id == user_token_id:
            current_idx = idx
            while labels[current_idx] != assistant_token_id and current_idx < len(labels):
                labels[current_idx] = IGNORE_INDEX
                current_idx += 1

