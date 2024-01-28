# exllamav2-openai-server
An implementation of the OpenAI API using the exllama2 backend. It supports batching and streaming.

This wouldn't be possible without exllamav2 or EricLLM. I saw [EricLLM](https://github.com/epolewski/EricLLM) and thought it was already doing what this package does, and my disappointment kickstarted me into writing this code.

supports:
* OpenAI API `/v1/models`, `/v1/chat/completions`
* continuous batching
* streaming responses
* dynamic model loading
* import of ollama configs (bring your own exl2)
* works with continue.dev, ollama webui, openai python module
* can't vouch for multi-gpu because my other gpu hasn't arrived

## Installation


## Usage

* create_model.py
* arg hierarchy: server args > create args > ollama defaults > model defaults

Example usage for one GPU:
```
python ericLLM.py --model ./models/NeuralHermes-2.5-Mistral-7B-5.0bpw-h6-exl2 --max_prompts 8 --num_workers 2
```
In a dual-GPU setup:
```
python ericLLM.py --model ./models/NeuralHermes-2.5-Mistral-7B-5.0bpw-h6-exl2 --gpu_split 24,24 --max_prompts 8 --num_workers 4 --gpu_balance
```
These will both launch the API with multiple workers. In the second example, performance is increased with the --gpu_balance switch that keeps the small models from splitting over GPUs. There's still work to be done on this and I think it gets CPU-bound right now when using 2 GPUs.

Test the API:

```
...use the openai python lib...
```

## Options

Help is available via `--help`` or `-h``

## About


## Throughput
