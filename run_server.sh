#!/bin/bash

nohup python3 model_proxy_fastapi.py --port 5030 &
nohup python3 llm_deploy_fastapi.py --model_id replicate/vicuna-13b --port 5031 &
nohup python3 llm_deploy_fastapi.py --model_id replicate/stablelm-tuned-alpha-7b --port 5032 &
nohup python3 llm_deploy_fastapi.py --model_id replicate/dolly-v2-12b --port 5033 &
nohup python3 llm_deploy_fastapi.py --model_id replicate/llama-2-70b-chat --port 5034 &
nohup python3 llm_deploy_fastapi.py --model_id replicate/codellama-13b-instruct --port 5035 &