from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import httpx
import asyncio
import logging
from config import model_dict, default_args_dict, ALLOWED_ORIGINS
from utils.utils import convert_to_https
import json
import argparse
import uvicorn


app = FastAPI()
path = '/ai/llm'


@app.post(path)
async def api_proxy(request: dict, use_ssl=False):
    try:
        data = request
        if data.get('model', None) is None:
            logging.error(f"'model' not found in request! Using default model 'vicuna-13b'")
            data['model'] = 'vicuna-13b'
        logging.info(f"Received data: {data}")
    except ValueError as e:
        logging.error(f"JSON decoding error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Could not decode JSON request.")

    try:
        model = data['model']
        logging.info(f"Model: {model}")
        if model not in model_dict.keys():
            raise HTTPException(status_code=400, detail=f"Invalid model '{model}'. Valid models are {list(model_dict.keys())}.")

        stream = data.get('stream', False)
        logging.info(f"Stream mode: {stream}")

        for key, value in default_args_dict.items():
            if key not in data:
                data[key] = value

        destination_ip = model_dict[model]['ip']
        if use_ssl:
            destination_ip = convert_to_https(destination_ip)
    except ValueError as e:
        logging.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Input validation error.")

    async def stream_response():
        logging.info(f"Destination IP: {destination_ip}")
        async with httpx.AsyncClient() as client:
            async with client.stream(
                    method=request.get("method", "POST"),
                    url=destination_ip,
                    headers=request.get("headers", {}),
                    params=request.get("params", {}),
                    json=data,
                    timeout=600,
            ) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk
        logging.info(f"Request completed")

    try:
        return StreamingResponse(stream_response())
    except ValueError as e:
        logging.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Input validation error.")


# python3 model_proxy_fastapi.py --port 5000 --path /ai/llm
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to start the server on.')
    parser.add_argument('--ssl', action='store_true', help='Enable SSL')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    port = args.port
    ssl_options = {}
    if args.ssl:
        ssl_options = {
            "certfile": "cert.pem",
            "keyfile": "key.pem",
        }
    logging.info(f"Server started on http{'s' if args.ssl else ''}://localhost:{args.port}{path}")
    asyncio.run(uvicorn.run(app, host="0.0.0.0", port=port,
                ssl_keyfile=ssl_options.get("keyfile"), ssl_certfile=ssl_options.get("certfile")))
