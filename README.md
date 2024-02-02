# ExLlamaV2-OpenAI-Server

An implementation of the OpenAI API using the ExLlamaV2 backend.
This project is not affiliated with ExLlamaV2 or OpenAI.

## Features

* Continuous batching.
* Streamed responses.
* OpenAI compatibility for `/v1/models` and `/v1/chat/completions` endpoints
* Uses Ollama model metadata information to set default prompting and parameters.
* Remembers your settings per model.
* Loads models on demand.
* Status endpoint with graphs! (and nothing else)

I've been testing against the python openai module, [Ollama Web UI](https://github.com/ollama-webui/ollama-webui) and [continue.dev](https://continue.dev/).

## Origin Story

This wouldn't be possible without [ExLlamaV2](https://github.com/turboderp/exllamav2) or EricLLM. I saw [EricLLM](https://github.com/epolewski/EricLLM) and thought it was close to
doing what I wanted, and by the time I realized what I was doing, I had pretty much completely rewritten it.

My goals are to be able to figure out how to set up a model once (preferably by leveraging work by the Ollama team) and then easily use it in a variety of frontends without thinking about it again. However, I also like to be able to quantize things to meet specific memory goals, and I like the performance of ExLlamaV2. Hence this project.

## Issues

* I have no idea what I'm doing.
* To combat creeping VRAM usage over time it is aggressively calling `torch.cuda.empty_cache()` which definitely has a performance impact, but it's better than running out of VRAM.
* It's currently streaming everything internally, which is almost certainly slowing down non-streaming requests.
* The ExLlamaV2 class `ExLlamaV2StreamingGenerator` has too much important stuff in it to avoid using it, but it also wasn't meant to be used this way.
* Model loading is synchronous, prompt parsing is synchronous, token decode is serialized with model inference, ...

## Installation

```
git clone https://github.com/bjj/exllamav2-openai-server
cd exllamav2-openai-server
pip install -r requirements.txt
```

Notes:
* Tested on python 3.11.7. 3.12+ seems to have version conflicts.
* First start will take a long time to compile `exllamav2_ext`.

## Adding Models

To add a new model:

1. Download a ExLlamaV2 compatible model (EXL2 or GPTQ)
2. Browse the [Ollama Library](https://ollama.ai/library) to find the matching repository. This is where we'll get prompt information and other default settings.
3. Run `python create_model.py --model-dir <path_to_exl2> <repository[:tag]>`
4. Repeat for as many models as you want to use

Note that tags are optional and often have the same metadata in the Ollama library. You can use them for yourself to give models unique names, for example `deepseek-coder:6.7b` and `deepseek-coder:33b`. This never downloads the GGUF files used by Ollama, so it doesn't matter what their default quantization is or what quantization tag you choose. **The quantization is determined by what EXL2 or GPTQ you download for yourself.**

You can also pass options to `create_model.py` to override options provided by Ollama. For example, to add Mixtral-8x7B-Instruct, with a model in `E:\...`, prompting from Ollama, but with batching limited to 1 and context limited to 16k (for memory):

```
python .\create_model.py --model-dir E:\exl2-llm-models\turboderp\Mixtral-8x7B-instruct-3.5bpw-exl2\ --max-batch-size 1 --max-seq-len 16384 mixtral
```

You can add models while the server is running. It reads the `models.json` file again whenever it needs to know about the models.

## Running the Server

You can run the server with no arguments. It will listen on `0.0.0.0:8000` by default:

```
python server.py
```

The server takes several optional arguments. The options used are selected with the following priority:

1. The options provided in the API request (if they don't exceed limits)
2. The `server.py` command line arguments
3. The `create_model.py` command line arguments
4. The Ollama repository configuration data
5. The model's `config.json`

For example, you pass `--max-batch-size 8` to the server. You get a batch size of 8 even though the model (see example above) was limited to `--max-batch-size 1`.

You can do a quick test with `curl http://localhost:8000/v1/models`

## If you get "Out of Memory"

When loading the model, the automatic GPU splitting in ExLlamaV2 allocates all the memory it could possibly need to satisfy the batching and context requirements. If you run out of memory (or your model loads onto two GPUs when you are sure it should fit on one):

* Use `--cache_8bit` to reduce the memory footprint of the cache. This has a significant effect without sacrificing much.
* Reduce the maximum batch size per-model or on the `server.py` command line with `--max-batch-size`. Maximum throughput can be achieved with fairly low batch sizes. Going higher than that just lets more users see progress happening on streaming requests at once.
* Reduce the maximum context size per-model or on the `server.py` command line with `--max-seq-len`. This will catch you out on small models with huge context like `dolphin-mistral:7b`.
* There is a very small effect from reducing `--max-input-len`.
* If you use manual `--gpu_split` you will load the model without accounting for the memory needed to actually handle requests. This will work fine if you don't get many concurrent requests and/or don't use much context, but you risk running out of memory unexpectedly later.

## Monitoring

There is a simple webpage at `http://localhost:8000/`

![screenshot](batchplot.png)

## Windows

You can run on native Windows if you can set up a working python+CUDA environment. I recommend using the nvidia control panel to change "CUDA sysmem fallback policy" to "prefer no sysmem fallback" because I would rather OOM than do inference at the terrible speeds you get if you overflow into "shared" GPU memory.

You can run on WSL2 by setting up CUDA using [nvidia's guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html). The server will translate Windows paths in your models.json so you can experiment with it. WSL2 access to Windows filesystems is unbelieveably slow, so models will take forever to load. If WSL2 is your long-term strategy you'll want your models in a native filesystem. On WSL2, fast models are a little faster (maybe 10%) and slow models are about the same.

## Multi GPU

You can manually specify the memory split with `--gpu_split`, but it's very finicky to get right. Otherwise it will use ExLlamaV2's automatic splitting. Note that the auto splitting works by allocating as much memory as it will ever need for maximum context length and batch size. See "If you get 'Out of Memory'" above.
