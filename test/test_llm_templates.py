import requests
import json
import time
import warnings
from prompt.dialogues import SUPPORTED_DIALOGUE_TEMPLATES
from test_data import conversations
warnings.filterwarnings("ignore")

'''
This code conducts testing and evaluation of a language model by sending conversations as prompts and collecting
model-generated responses. It supports testing different prompt styles, and the responses are either printed in chunks
(streaming mode) or as a complete response. The code iterates over a list of conversations, sends requests to the model
API, and processes the received responses for analysis. The primary purpose of the code is to assess the model's
performance on various conversations and prompt styles.
'''

# user defined params
# ------------------------------------------------------------------------
url = 'https://localhost:5000/'
test_all_prompt_styles = True
stream = False
# ------------------------------------------------------------------------

if __name__ == "__main__":
    # Test all prompt styles
    if test_all_prompt_styles:
        prompt_styles_list = SUPPORTED_DIALOGUE_TEMPLATES.keys()
    else:
        prompt_styles_list = ['original']

    for conversation_index, conversation in enumerate(conversations):
        print(f"Conversation {conversation_index + 1}: {conversation[-2]['content']}")

        data_base = {
            "model": "",
            "temperature": 0,
            "stream": stream,
            "max_new_tokens": 1024,
            "min_new_tokens": 1,
            "messages": conversation,
            "stop": []
        }

        for prompt_style in prompt_styles_list:
            print(f"\nTesting prompt_style: {prompt_style}")
            data = data_base.copy()
            if prompt_style != 'original':
                data["prompt_style"] = prompt_style

            response = requests.post(url, json=data, stream=stream, verify=False)

            if stream:
                response_list = []
                start = time.time()
                counter = 0

                for chunk in response.iter_content(chunk_size=None):
                    if chunk:
                        counter += 1
                        message = json.loads(chunk)
                        response_list.append(chunk)
                        chunk = chunk.decode('utf-8')

                        print(f'\nChunk {counter}:')
                        print(f'Time elapsed: {time.time() - start:.2f} seconds')
                        print('\nContent:')
                        print(chunk)
                        print('-----------------------------------')

                print(f'\nTotal number of chunks: {len(response_list)}')
            else:
                response = json.loads(response.text)
                print('\nResponse content:')
                print(response['choices'][0]['message']['content'])
                print('-------------------------------------------------')
