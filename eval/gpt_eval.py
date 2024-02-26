import os
import openai
import re
from utils.utils import exists_in_folder, delete_files
from eval_prompts import coding_system_prompt, instruction_system_prompt, user_prompt

'''
The code evaluates responses to prompts using the OpenAI GPT-4 model.
It scores each response based on certain criteria.
It iterates through prompt files and generates responses using the GPT-4 model.
The generated responses are assigned scores and saved in evaluation files.
At the end, it calculates and prints the average score for all responses.
'''

# user defined params
# ------------------------------------------------------------------------
os.environ['openai.api_key'] = 'your-api-key'
assistant = '../test/result'  # directory to store results
task = 'instruction'  # ['coding', 'instruction']
resume_eval = False  # skip already evaluated files
# ------------------------------------------------------------------------

if __name__ == "__main__":
    if task == 'instruction':
        system_prompt = instruction_system_prompt
    elif task == 'coding':
        system_prompt = coding_system_prompt
    else:
        raise Exception(f'Task {task} not found')
    total_score = 0
    prompt_files = sorted([f for f in os.listdir(assistant) if 'prompt' in f])
    response_files = sorted([f for f in os.listdir(assistant) if 'response' in f])
    if not resume_eval:
        delete_files('gpt_eval_', assistant)
    for i in range(len(prompt_files)):
        eval_n = prompt_files[i].split('.txt')[0].split('_')[-1]
        if resume_eval and exists_in_folder(f'gpt_eval_{eval_n}_', assistant):
            print(f"The question '{prompt_files[i]}' has already been scored. Skipping this.")
            continue
        with open(f'{assistant}/{prompt_files[i]}', 'r') as f:
            prompt = f.read()
        with open(f'{assistant}/{response_files[i]}', 'r') as f:
            response = f.read()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(prompt, response)}]
        gpt_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.,
            max_tokens=1000,
            frequency_penalty=0.0
        )
        gpt_response = gpt_response.choices[0].message.content.strip()
        score = gpt_response.split()[0]
        if len(score) > 1:
            score = score[0]
        score = int(float(score))
        score = min(max(1, score), 8)
        with open(f'{assistant}/gpt_eval_{eval_n}_score{score}.txt', 'w') as f:
            f.write(gpt_response)
        gpt_response = gpt_response.split('\n', 1)[1]
        print(f"The score of question '{prompt_files[i]}' is: {score}\n"
              f"{gpt_response}\n"
              "------------------------------------------------------------------------")

    # calculate average score
    files_in_directory = os.listdir(assistant)
    total_score = 0
    file_count = 0
    for filename in files_in_directory:
        # Check if the file matches the pattern 'scoreN' anywhere in its name
        match = re.search(r"score(\d+)", filename)
        if match is not None:
            score = int(match.group(1))
            total_score += score
            file_count += 1
    # Calculate and print the average score
    average_score = total_score / file_count if file_count > 0 else 0
    print(f"\nThe average score for the responses assessed is: {average_score}/8")
    # Create a new file with the average score as the name
    new_file_path = os.path.join(assistant, f"Avg_{average_score}.txt")
    with open(new_file_path, 'w') as f:
        f.write(f"Avg_{average_score}")
