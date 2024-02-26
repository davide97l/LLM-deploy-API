# LLM deploy API

This project implements an API designed to efficiently deploy Language Model (LLM) applications using Flask API. The framework is built with modular code that allows easy integration of new open source models from a range of platforms including HuggingFace and Replicate.

## Installation and Deployment

Follow these steps to deploy the LLM application:

3. Install the necessary requirements:

```
pip install -r requirements.txt
```

4. Run the file `run_server.sh` to deploy the models.
```
sh run_server.sh
```

## Shutting Down

To stop the application, run the following command:

```
kill $(lsof -t -i :IP)
```
