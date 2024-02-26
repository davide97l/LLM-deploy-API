import argparse
import json
import time
import logging
import asyncio
from fastapi import FastAPI, Request, Response, Depends, HTTPException
from models import config_dict
from prompt import make_chat_history
import uvicorn
import traceback
from fastapi.responses import StreamingResponse
from config import GLOBAL_THREAD_POOL, gen_params, ErrorResponse, PredictOutput, PredictInput


app = FastAPI()
llm = None


async def predict_fn(data, llm):
    prompt = data.pop("prompt", data)
    gen_parameters = data.pop("gen_params", None)
    stop = data.pop("stop", None)
    if isinstance(stop, str):
        stop = [stop]

    if llm.model_author in ['replicate', 'test']:
        kwargs = dict(
            prompt=prompt,
        )
        if gen_parameters is not None:
            kwargs.update(gen_parameters)
        kwargs = llm.preprocess_gen_kwargs(kwargs)
        streamer = llm.generate(**kwargs)
    else:  # huggingface
        input_ids = llm.tokenize([prompt])
        streamer = llm.create_stream(timeout=600)
        kwargs = dict(
            input_ids,
            streamer=streamer,
        )
        if gen_parameters is not None:
            kwargs.update(gen_parameters)
        kwargs = llm.preprocess_gen_kwargs(kwargs)
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, lambda: llm.generate(**kwargs))

    is_stop = False
    gen_text = ''
    c = 0
    add_space = False
    async for new_text in streamer:
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


@app.post("/",
    responses={
        400: {"model": ErrorResponse, "description": "Input validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="LLM API",
    response_model=PredictOutput,
    description="Chat with different open-source LLMs"
)
async def predict(request: Request):
    try:
        inputs = await request.json()
        prompt_style = llm.prompt_style
        if inputs.get('prompt_style', None):  # override prompt_style if present in input
            prompt_style = inputs['prompt_style']
        prompt = make_chat_history(inputs['messages'], prompt_style)
        prompt_tokens = llm.n_tokens([prompt])
        last_question_tokens = 0
        for message in reversed(inputs['messages']):
            if len(message["content"]) > 0:
                last_question_tokens = llm.n_tokens([message["content"]])
                break
        stream = inputs.get('stream', False)
        # return only chunks of text, don't return 'event', 'model', 'role'
        return_only_text = inputs.get('return_only_text', False)
        update_keys = gen_params.keys()  # take default params
        # update default params with model params
        gen_params.update(llm.gen_params)
        # update default params with user defined params
        gen_params.update({key: inputs[key] for key in inputs if key in update_keys})
        # stop words
        stop = []
        stop += inputs.get('stop', [])
        stop += llm.gen_params.get('stop', [])
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
    except Exception as e:
        error_message = f"An error occurred while processing the request: {str(e)}"
        logging.error(error_message)
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail="Input validation error")

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
            yield f"data: {json.dumps(data)}\n\n"
        generated_text = ''
        async for chunk in predict_fn(input_data, llm):
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
            yield f"data: {json.dumps(data)}\n\n"
            yield f"data: {json.dumps({'event': '[DONE]'})}\n\n"
        output_data['usage']['completion_tokens'] = llm.n_tokens([generated_text])
        output_data['usage']['last_question_tokens'] = last_question_tokens
        output_data['usage']['total_tokens'] = output_data['usage']['completion_tokens'] + \
                                               output_data['usage']['prompt_tokens']
        if stream:
            if not return_only_text:
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

    try:
        response = StreamingResponse(generate(), media_type='text/event-stream')
        response.headers["Content-Type"] = "text/event-stream"
        response.headers["Connection"] = "keep-alive"
        response.headers["Cache-Control"] = "no-cache"
        return response

    except Exception as e:
        error_message = f"An error occurred while processing the request: {str(e)}"
        logging.error(error_message)
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error")


def parse_args():
    parser = argparse.ArgumentParser(description='AI model server')
    parser.add_argument('--model_id', type=str, default='tiiuae/falcon-7b-instruct', help='Model ID to use')
    parser.add_argument('--port', type=int, default=5000, help='Port for the server')
    parser.add_argument('--ssl', action='store_true', help='Enable SSL')
    args = parser.parse_args()
    return args


# python llm_deploy_fastapi.py --port 5000 --model_id replicate/vicuna-13b
# python llm_deploy_fastapi.py --port 6000 --model_id test/fake-model
# python llm_deploy_fastapi.py --port 6001 --model_id google/flan-t5-small
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
    ssl_options = {}
    if args.ssl:
        ssl_options = {
            "certfile": "cert.pem",
            "keyfile": "key.pem",
        }
    logging.info(f'Application started on port {port}')
    asyncio.run(uvicorn.run(app, host="0.0.0.0", port=port,
                ssl_keyfile=ssl_options.get("keyfile"), ssl_certfile=ssl_options.get("certfile")))
