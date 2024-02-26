import requests
import json
import time
import warnings
warnings.filterwarnings("ignore")

# user defined params
# ------------------------------------------------------------------------
url = 'http://localhost:5030/ai/llm'
stream = 0
model = 'vicuna-13b'  # used only if testing proxy
return_only_text = True
# ------------------------------------------------------------------------

if __name__ == "__main__":
    data = {
        "model": model,
        "temperature": 0.1,
        "stream": stream,
        "max_new_tokens": 1024,
        'min_new_tokens': 1,
        #'return_only_text': return_only_text,
        "messages": [
            {"role": "system", "content": "You're a helpful assistant"},
            {"role": "user", "content": "I ate an apple and a watermelon with my friend."},
            {"role": "assistant", "content": "good."},
            {"role": "user", "content": "what did i eat?"},
            {"role": "assistant", "content": ""}
        ],
    }

    response = requests.post(url, json=data, stream=stream, verify=False)
    if stream:
        list = []
        start = time.time()
        c = 0
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                c += 1
                #message = json.loads(chunk)
                list.append(chunk)
                #chunk = chunk.decode('utf-8')
                print(f'{c}) {time.time() - start}')
                print(chunk)
                print('-------')
        print('Total chunks:', len(list))
    else:
        response = json.loads(response.text)
        print(response['choices'][0]['message']['content'])
