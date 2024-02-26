from datasets import load_dataset

general_questions = [
    ("Solve the following equation: 2x + 3 = -1", -2),
    ("If all oranges are fruits and some fruits are round, does it mean that all oranges are round?", False),
    ("If John is older than Jane, and Jane is younger than Tom, who is the youngest?", "Jane"),
    ("What is the capital city of France?", "Paris"),
    ("Unscramble the letters to form a word to form a meaningful word: ONIORVSE", "Version"),
    ("How many sides does a hexagon have?", 6),
    ("If all elephants are gray and no gray animals can fly, can an elephant fly?", False),
    ("Continue the pattern - 2, 4, 8, 16, __", 32),
    ("Which planet is the largest in our solar system?", "Jupiter"),
    ("Choose the correct word to complete the sentence: Due to bad weather, the flight was ____. (imminent, eminent, immanent)",
     "imminent")
]

coding_questions_easy = [
    "Write a function that takes a positive integer as input and returns the sum of all the numbers from 1 to that integer.",
    "Create a function that takes a string as input and returns the string reversed.",
    "Write a function that generates the first n terms of the Fibonacci sequence where n is a positive integer provided as input.",
    "Create a function that takes a string as input, counts the number of vowels (a, e, i, o, u) in the string, and returns the count.",
    "Write a function that takes two positive integers as input and returns their greatest common divisor using the Euclidean algorithm."
]


conversations = [
    [
        {"role": "system", "content": "You're a helpful assistant"},
        {"role": "user", "content": "I ate an apple and a watermelon with my friend."},
        {"role": "assistant", "content": "good."},
        {"role": "user", "content": "what did I eat?"},
        {"role": "assistant", "content": ""}
    ],
    [
        {"role": "system", "content": "You're a helpful assistant"},
        {"role": "user", "content": "Mergesort is an algorithm to sort arrays"},
        {"role": "assistant", "content": "good."},
        {"role": "user", "content": "Write it in python and modify it to sort an array in reverse order"},
        {"role": "assistant", "content": ""}
    ],
    [
        {"role": "system", "content": "You're a helpful assistant"},
        {"role": "user", "content": "My birthday is on 15 of June"},
        {"role": "assistant", "content": "good."},
        {"role": "user", "content": "Today is 3 of March, how many days until my birthday"},
        {"role": "assistant", "content": ""}
    ],
    [
        {"role": "system", "content": "You're a helpful assistant"},
        {"role": "user", "content": "John is older than Jane, and Jane is younger than Tom"},
        {"role": "assistant", "content": "good."},
        {"role": "user", "content": "who is the youngest?"},
        {"role": "assistant", "content": ""}
    ],
    [
        {"role": "system", "content": "You're a helpful assistant"},
        {"role": "user", "content": "All oranges are fruits and some fruits are round"},
        {"role": "assistant", "content": "good."},
        {"role": "user", "content": "Does it mean that all oranges are round?"},
        {"role": "assistant", "content": ""}
    ],
]


def get_dataset(dataset_name):
    if dataset_name == "general_questions":
        dataset = general_questions
    elif dataset_name == "coding_questions_easy":
        dataset = coding_questions_easy
    elif dataset_name == "hf_code_benchmark":
        d = load_dataset("HuggingFaceH4/code_evaluation_prompts")['train']
        dataset = [example['prompt'] for example in d][:50]
    elif dataset_name == "self_instruct_benchmark":
        # https://huggingface.co/datasets/HuggingFaceH4/testing_self_instruct_small
        d = load_dataset("HuggingFaceH4/testing_self_instruct_small")['test']
        dataset = [example['prompt'] for example in d][:100]
    else:
        print("Invalid dataset name!")
        return

    return dataset
