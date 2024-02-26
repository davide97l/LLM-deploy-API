from prompt import make_chat_history
from dialogues import SUPPORTED_DIALOGUE_TEMPLATES
import copy

# Test case inputs
test_messages = [
    {"role": "system", "content": "You're a helpful assistant"},
    {"role": "user", "content": "I ate an apple and a watermelon with my friend."},
    {"role": "assistant", "content": "good."},
    {"role": "user", "content": "what did I eat?"},
    {"role": "assistant", "content": ""}
]

# Test the function with different prompt styles
test_prompt_styles = SUPPORTED_DIALOGUE_TEMPLATES.keys()
for prompt_style in test_prompt_styles:
    print(f"Testing with prompt_style='{prompt_style}':")
    print("\nPROMPT:\n")
    result = make_chat_history(copy.deepcopy(test_messages), prompt_style=prompt_style)
    print(result)
    print("-" * 50)
