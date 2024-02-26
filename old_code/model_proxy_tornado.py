import tornado.ioloop
import tornado.web
import tornado.httpclient
from tornado.escape import json_decode, json_encode
from tornado.iostream import StreamClosedError
from tornado.ioloop import IOLoop
from tornado.httpserver import HTTPServer
import logging
import argparse
from tornado.locks import Semaphore
from config import model_dict, default_args_dict, ALLOWED_ORIGINS
from utils.utils import convert_to_https


class MainHandler(tornado.web.RequestHandler):
    use_ssl = False
    model_semaphore_dict = {model: Semaphore(info['limit']) for model, info in model_dict.items()}

    @classmethod
    def update(cls, use_ssl=False, requests_limit=None):
        cls.use_ssl = use_ssl
        if requests_limit:
            cls.model_semaphore_dict = {model: Semaphore(requests_limit) for model, info in model_dict.items()}
            logging.info(f"Semaphore limit updated for all models to: {requests_limit}")

    def set_default_headers(self):
        origin = self.request.headers.get('Origin', '')

        if any(pattern.search(origin) for pattern in ALLOWED_ORIGINS):
            self.set_header("Access-Control-Allow-Origin", origin)
            logging.info(f"Set Access-Control-Allow-Origin: {origin}")
        else:
            self.set_header("Access-Control-Allow-Origin", "none")
            logging.info(f"Set Access-Control-Allow-Origin: none")

        self.set_header("Access-Control-Allow-Headers", "Content-Type, x-requested-with, Authorization")
        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.set_header("Access-Control-Allow-Credentials", "true")

    async def options(self, *args):
        # no body
        # `*args` is for route with `path arguments` supports
        self.set_status(204)
        await self.finish()

    async def post(self):
        try:
            try:
                data = json_decode(self.request.body)
                if data.get('model', None) is None:
                    logging.error(f"'model' not found in request! Using default model 'vicuna-13b'")
                    data['model'] = 'vicuna-13b'
                logging.info(f"Received data: {data}")
            except ValueError as e:
                logging.error(f"JSON decoding error: {e}", exc_info=True)
                self.set_status(400)  # Bad Request
                self.write(json_encode({'error': 'Could not decode JSON request.'}))
                return

            model = data['model']
            logging.info(f"Model: {model}")
            if model not in model_dict.keys():
                self.set_status(400)  # Bad Request
                self.write(json_encode({'error': f"Invalid model '{model}'. Valid models are {list(model_dict.keys())}."}))
                return

            stream = data.get('stream', False)
            logging.info(f"Stream mode: {stream}")

            for key, value in default_args_dict.items():
                if key not in data:
                    data[key] = value

            data = json_encode(data).encode()

            await MainHandler.model_semaphore_dict[model].acquire()  # Acquire the semaphore per model

            destination_ip = model_dict[model]['ip']
            if MainHandler.use_ssl:
                destination_ip = convert_to_https(destination_ip)
            logging.info(f"Destination IP: {destination_ip}")
            tornado.httpclient.AsyncHTTPClient.configure(None, defaults=dict(request_queue_timeout=300))
            http_client = tornado.httpclient.AsyncHTTPClient()

            try:
                if not stream:
                    self.set_header('Content-Type', 'application/json')
                else:
                    self.set_header('Content-Type', 'text/event-stream')
                self.set_header('Cache-Control', 'no-cache')
                self.set_header('Connection', 'keep-alive')
                response = await http_client.fetch(
                    destination_ip,
                    method="POST",
                    body=data,
                    request_timeout=600,
                    connect_timeout=300,
                    streaming_callback=self.handle_stream,
                    validate_cert=not MainHandler.use_ssl,
                )
            except tornado.httpclient.HTTPError as e:
                logging.error(f"Error forwarding request: {e}", exc_info=True)
                self.set_status(500)
                self.write(json_encode({'error': f'Error forwarding request: {e}.'}))
                return
        finally:
            # Release the semaphore
            if model in model_dict.keys():
                MainHandler.model_semaphore_dict[model].release()

    def handle_stream(self, data):
        self.write(data)
        try:
            self.flush()
        except StreamClosedError as e:
            logging.error(f"Stream closed prematurely: {e}", exc_info=True)


def make_app(path=r"/"):
    return tornado.web.Application([
        (path, MainHandler),
    ])


# python3 model_proxy.py --port 5000 --requests_limit 10 --path /ai/llm
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to start the server on.')
    parser.add_argument('--ssl', action='store_true', help='Enable SSL')
    parser.add_argument('--requests_limit', type=int, default=None, help='Concurrent requests limit')
    parser.add_argument('--path', type=str, default='/ai/llm', help='Deployment path')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    ssl_options = None
    if args.ssl:
        ssl_options = {
            "certfile": "cert.pem",
            "keyfile": "key.pem",
        }
    MainHandler.update(args.ssl, args.requests_limit)
    path = r"{}".format(args.path)
    app = make_app(path=path)
    server = HTTPServer(app, ssl_options=ssl_options)
    server.listen(args.port)
    logging.info(f"Server started on http{'s' if args.ssl else ''}://localhost:{args.port}{path}")
    IOLoop.current().start()
