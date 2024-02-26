import requests
import json
import warnings
import os
import datetime
from test_data import get_dataset
import openai
from dotenv import load_dotenv
load_dotenv()  # load .env file
warnings.filterwarnings("ignore")

'''
This script evaluates a language model (possibly GPT-4) on a dataset of questions.
It sends requests to the model API or uses OpenAI's ChatCompletion model to generate responses.
If enabled, the responses and questions are saved to files.
The model's performance on the dataset is printed to the console.
'''

# user defined params
# ------------------------------------------------------------------------
url = 'http://localhost:5030/ai/llm'
eval_dir = ''  # gpt4_self_instruct_benchmark'
dataset_name = 'general_questions'  # ['general_questions', 'coding_questions_easy', 'hf_code_benchmark', 'self_instruct_benchmark']
save_to_file = True
test_gpt = False
model = 'vicuna-13b'
# ------------------------------------------------------------------------

if __name__ == "__main__":
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "temperature": 0.1,
        "stream": False,
        "max_new_tokens": 1024,
        'min_new_tokens': 1,
        "messages": [
            {"role": "system", "content": "You're a helpful assistant."},
            {"role": "user", "content": "{}"},
        ],
        "stop": []
    }

    question_idx = -1
    if test_gpt:
        print('Attention: you are evaluating GPT4!')
    questions = get_dataset(dataset_name)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if save_to_file:
        evaluation_dir = f'evaluation_{timestamp}' if eval_dir is None else eval_dir
        os.makedirs(evaluation_dir, exist_ok=True)
    for i, qa in enumerate(questions):
        if save_to_file:
            response_file_path = f'{evaluation_dir}/response_{i+1}.txt'
            prompt_file_path = f'{evaluation_dir}/prompt_{i+1}.txt'
            if os.path.exists(response_file_path):
                print(f"Question {i+1} has already been answered, skipping...")
                continue

        if len(qa) == 2:
            q, a = qa
        else:
            q, a = qa, None
        a = str(a)
        temp = data["messages"][question_idx]["content"]
        data["messages"][question_idx]["content"] = data["messages"][question_idx]["content"].format(q)
        if not test_gpt:
            response = requests.post(url, headers=headers, json=data, stream=False, verify=False)
            response = json.loads(response.text)
        else:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=data["messages"],
                temperature=0.,
                max_tokens=1000,
                frequency_penalty=0.0
            )
        response = response['choices'][0]['message']['content']
        print(i+1, q)
        print(response)

        if save_to_file:
            with open(response_file_path, 'w') as file:
                file.write(response)
            with open(prompt_file_path, 'w') as file:
                file.write(q)

        if a is not None:
            print('Correct response:', a)
        print('----------------')
        data["messages"][question_idx]["content"] = temp
