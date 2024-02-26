import argparse
import json
import time
import logging
from tornado.ioloop import IOLoop
from tornado.web import Application, RequestHandler
from tornado.httpserver import HTTPServer
from models import config_dict
from prompt import make_chat_history
import tornado
import traceback
import asyncio


# generation params
gen_params = {
    "early_stopping": False,
    "no_repeat_ngram_size": 0,
    "do_sample": False,
    "min_new_tokens": 0,
    "num_beams": 1,
    "top_k": 50,
    "diversity_penalty": 0.,
    "repetition_penalty": 1.,
    "max_new_tokens": 1024,
    "temperature": 1.,
    "top_p": 1.,
    #"num_return_sequences": 1,
}


class PredictionHandler(RequestHandler):
    def initialize(self, llm):
        self.llm = llm

    async def predict_fn(self, data):
        prompt = data.pop("prompt", data)
        gen_parameters = data.pop("gen_params", None)
        stop = data.pop("stop", None)
        if isinstance(stop, str):
            stop = [stop]

        if 'replicate' in self.llm.model_author:
            kwargs = dict(
                prompt=prompt,
            )
            if gen_parameters is not None:
                kwargs.update(gen_parameters)
            kwargs = self.llm.preprocess_gen_kwargs(kwargs)
            streamer = self.llm.generate(**kwargs)
        else:  # huggingface
            input_ids = self.llm.tokenize([prompt])
            streamer = self.llm.create_stream(timeout=600)
            kwargs = dict(
                input_ids,
                streamer=streamer,
            )
            if gen_parameters is not None:
                kwargs.update(gen_parameters)
            kwargs = self.llm.preprocess_gen_kwargs(kwargs)
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, lambda: self.llm.generate(**kwargs))

        is_stop = False
        gen_text = ''
        c = 0
        add_space = False
        for new_text in streamer:
            if len(new_text) == 0:
                continue
            if c == 0 and new_text[-1] != ' ':
                new_text += ' '
                add_space = True
            if add_space and new_text[0] == ' ':
                new_text = new_text[1:]
                add_space = False
            gen_text += new_text
            if stop:
                for stop_word in stop:
                    if stop_word in gen_text:
                        new_text = new_text.split(stop_word)[0]
                        is_stop = True
                        break
            if is_stop:
                if new_text and len(new_text) > 0:
                    yield new_text
                break
            c += 1
            yield new_text

    async def post(self):
        logging.info(f'Invoked {self.llm.model_name}')
        try:
            inputs = json.loads(self.request.body.decode('utf-8'))
            prompt_style = self.llm.prompt_style
            if inputs.get('prompt_style', None):  # override prompt_style if present in input
                prompt_style = inputs['prompt_style']
            prompt = make_chat_history(inputs['messages'], prompt_style)
            prompt_tokens = self.llm.n_tokens([prompt])
            last_question_tokens = 0
            for message in reversed(inputs['messages']):
                if len(message["content"]) > 0:
                    last_question_tokens = self.llm.n_tokens([message["content"]])
                    break
            stream = inputs.get('stream', False)
            return_only_text = inputs.get('return_only_text', False)  # return only chunks of text, don't return 'event', 'model', 'role'
            update_keys = gen_params.keys()  # take default params
            # update default params with model params
            gen_params.update(self.llm.gen_params)
            # update default params with user defined params
            gen_params.update({key: inputs[key] for key in inputs if key in update_keys})
            # stop words
            stop = []
            stop += inputs.get('stop', [])
            stop += self.llm.gen_params.get('stop', [])
            logging.info(f'Generation parameters: {gen_params}')
            logging.info(f'Prompt:\n{prompt}')

            input_data = {
                "prompt": prompt,
                "gen_params": gen_params,
                "stop": stop,
            }
            output_data = {
                'model': inputs['model'],
                'usage': {
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': 0,
                    'total_tokens': 0,
                }
            }

            async def generate():
                if stream and not return_only_text:
                    data = {
                        "choices": [
                            {
                                "delta": {
                                    "role": "assistant"
                                },
                                "finish_reason": None,
                                "index": 0
                            }
                        ],
                        "created": int(time.time()),
                        "id": "",
                        "model": inputs['model'],
                        "object": "chat.completion.chunk"
                    }
                    #yield json.dumps(data)
                    yield f"data: {json.dumps(data)}\n\n"
                generated_text = ''
                async for chunk in self.predict_fn(input_data):
                    generated_text += chunk
                    if stream:
                        data = {
                            "choices": [
                                {
                                    "delta": {
                                        "content": chunk
                                    },
                                    "finish_reason": None,
                                    "index": 0
                                }
                            ],
                            "created": int(time.time()),
                            "id": "",
                            "model": inputs['model'],
                            "object": "chat.completion.chunk"
                        }
                        #yield json.dumps(data)
                        yield f"data: {json.dumps(data)}\n\n"
                if stream and not return_only_text:
                    data = {
                        "choices": [
                            {
                                "delta": {},
                                "finish_reason": "stop",
                                "index": 0
                            }
                        ],
                        "created": int(time.time()),
                        "id": "",
                        "model": inputs['model'],
                        "object": "chat.completion.chunk"
                    }
                    #yield json.dumps(data)
                    yield f"data: {json.dumps(data)}\n\n"
                    #yield json.dumps({'event': '[DONE]'})
                    yield f"data: {json.dumps({'event': '[DONE]'})}\n\n"
                output_data['usage']['completion_tokens'] = self.llm.n_tokens([generated_text])
                output_data['usage']['last_question_tokens'] = last_question_tokens
                output_data['usage']['total_tokens'] = output_data['usage']['completion_tokens'] + \
                                                       output_data['usage']['prompt_tokens']
                if stream:
                    if not return_only_text:
                        #yield json.dumps(output_data)
                        yield f"data: {json.dumps(output_data)}\n\n"
                else:
                    output_data["id"] = ""
                    output_data["object"] = "chat.completion"
                    output_data["created"] = int(time.time())
                    output_data["choices"] = [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": generated_text,
                        },
                        "finish_reason": "stop"
                    }]
                    yield json.dumps(output_data)
                logging.info(f'Generated response: {generated_text}')

            self.set_header('Content-Type', 'text/event-stream')
            #self.set_header('Content-Type', 'application/json')
            self.set_header('Cache-Control', 'no-cache')
            self.set_header('Connection', 'keep-alive')

            async for chunk in generate():
                self.write(chunk)
                try:
                    await self.flush()
                except tornado.iostream.StreamClosedError as e:
                    error_message = f"Connection closed before response could be sent: {str(e)}"
                    logging.error(error_message)
                    logging.error(traceback.format_exc())
                    break
            logging.info(f'Returned output: {output_data}')

        except Exception as e:
            error_message = f"An error occurred while processing the request: {str(e)}"
            logging.error(error_message)
            logging.error(traceback.format_exc())
            self.set_status(500)
        await self.finish()


def parse_args():
    parser = argparse.ArgumentParser(description='AI model server')
    parser.add_argument('--model_id', type=str, default='tiiuae/falcon-7b-instruct', help='Model ID to use')
    parser.add_argument('--port', type=int, default=5000, help='Port for the server')
    parser.add_argument('--ssl', action='store_true', help='Enable SSL')
    args = parser.parse_args()
    return args


# python llm_deploy.py --model_id google/flan-t5-small
# python llm_deploy.py --model_id replicate/vicuna-13b --port 5001
# python llm_deploy.py --model_id replicate/stablelm-tuned-alpha-7b --port 5002
# python llm_deploy.py --model_id replicate/dolly-v2-12b --port 5003
# python llm_deploy.py --model_id tiiuae/falcon-7b-instruct --port 5000
# python llm_deploy.py --model_id bigcode/starcoder --port 5000
# python llm_deploy.py --model_id HuggingFaceH4/starchat-alpha --port 5000
if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(format='%(asctime)s [%(levelname)s] - %(message)s', level=logging.INFO)

    model_id = args.model_id
    logging.info(f'Loading model from: {model_id}')
    llm = config_dict[model_id]()
    logging.info(f'Loaded model {llm.model_name}')
    if llm.is_cuda():
        logging.info("Model is on a CUDA device (GPU)")
    else:
        logging.info("Model is on a CPU device")
    max_tokens = llm.max_tokens
    logging.info(f'Max tokens: {max_tokens}')
    logging.info(f"Memory footprint: {llm.memory_footprint():.2f} MB")

    port = args.port
    ssl_options = None
    if args.ssl:
        ssl_options = {
            "certfile": "cert.pem",
            "keyfile": "key.pem",
        }
    logging.info(f'Application started on port {port}')
    app = Application([(r"/", PredictionHandler, {'llm': llm})])
    server = HTTPServer(app, ssl_options=ssl_options)
    server.listen(port)
    IOLoop.current().start()
