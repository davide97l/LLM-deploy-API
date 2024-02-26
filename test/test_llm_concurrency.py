import requests
import time
import warnings
from threading import Thread
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
warnings.filterwarnings("ignore")

# user defined params
# ------------------------------------------------------------------------
url = 'http://localhost:5000/ai/llm'
n_requests = 10
parallel = True
model = 'vicuna-13b'
# ------------------------------------------------------------------------

if __name__ == "__main__":
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "temperature": 0.1,
        "stream": True,
        "max_new_tokens": 2048,
        'min_new_tokens': 1,
        "messages": [
            {"role": "system", "content": "You're a helpful assistant."},
            {"role": "user", "content": "Who is Michael Jordan?"},
        ],
        "stop": []
    }

    start = time.time()
    # Store individual response times
    response_times = []
    # Configure retries with backoff
    retries = Retry(
        total=100,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        method_whitelist=["HEAD", "GET", "PUT", "POST", "PATCH", "DELETE"]
    )
    # Create an HTTP session
    session = requests.Session()


    def make_request(i):
        print(f'started request: {i+1}')
        request_start_time = time.time()
        response = session.post(url, headers=headers, json=data, stream=True, verify=False, timeout=600)
        l = []
        for chunk in response.iter_content(chunk_size=None):  # Use the original method with iter_content
            if chunk:
                l.append(chunk)
        request_end_time = time.time()
        response_time = request_end_time - request_start_time
        response_times.append(response_time)
        print(f'{i+1})', 'streamed chunks:', len(l), ' / response time:', response_time)


    if parallel:
        # Create and start threads
        threads = [Thread(target=make_request, args=(i,)) for i in range(n_requests)]
        for thread in threads:
            thread.start()
        # Wait for all threads to finish
        for thread in threads:
            thread.join()
    else:
        [make_request(i) for i in range(n_requests)]

    end = time.time()
    # Calculate the total time and average response time
    total_time = end - start
    average_response_time = sum(response_times) / n_requests

    print('Num requests:', n_requests)
    print('Total time:', total_time)
    print('Average response time:', average_response_time)