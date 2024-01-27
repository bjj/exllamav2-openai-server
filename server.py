import sys, os, time, torch, random, asyncio, json, argparse, pathlib, gc
import typing
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.websockets import WebSocket
from fastapi.exceptions import WebSocketException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.encoders import jsonable_encoder
from openai_types import *
from fastapi_helpers import StreamingJSONResponse
import ollama_template
from create_model import read_registry

# Run exllamav2 from a git checkout in a sibling dir
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/exllamav2"
)
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Lora,
)
from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler


def parse_args():
    parser = argparse.ArgumentParser(description="OpenAI compatible server for exllamav2.")
    parser.add_argument("--verbose", action="store_true", default=False, help="Sets verbose")
    parser.add_argument("--model", metavar="REPOSITORY", type=str, help="Sets ollama-style model repository")
    parser.add_argument("--host", metavar="HOST", type=str, default="0.0.0.0", help="Sets host")
    parser.add_argument("--port", metavar="PORT", type=int, default=8000, help="Sets port")
    parser.add_argument("--timeout", metavar="TIMEOUT", type=float, default=120.0, help="Sets HTTP timeout")
    parser.add_argument("--max-seq-len", metavar="NUM_TOKENS", type=int, help="Sets context length")
    parser.add_argument("--max-input-len", metavar="NUM_TOKENS", type=int, help="Sets input length")
    parser.add_argument("--max-batch-size", metavar="N", type=int, help="Max prompt batch size")
    parser.add_argument("--gpu_split", metavar="GPU_SPLIT", type=str, default="",
                        help="Sets array gpu_split and accepts input like 16,24",)
    parser.add_argument(
        "--gpu_balance",
        action="store_true",
        default=False,
        help="Balance workers on GPUs to maximize throughput. Make sure --gpu_split is set to the full memory of all cards.",
    )
    parser.add_argument("--rope_alpha", metavar="rope_alpha", type=float, default=1.0, help="Sets rope_alpha", )
    parser.add_argument("--rope_scale", metavar="rope_scale", type=float, help="Sets rope_scale", )
    parser.add_argument(
        "--embiggen",
        metavar="embiggen",
        type=int,
        default=0,
        help="Duplicates some attention layers this many times to make larger frankenmodels dynamically. May increase cromulence on benchmarks.",
    )
    parser.add_argument(
        "--cache_8bit",
        metavar="CACHE_8BIT",
        type=bool,
        default=False,
        help="Use 8 bit cache (not implemented)",
    )
    parser.add_argument("--num_workers", metavar="NUM_WORKERS", type=int,
                        default=1, help="Number of worker processes to use", )

    return parser.parse_args()


args = parse_args()

def load_modelfile(repository):
    return ollama_template.ModelFile(repository)

modelfile = None
if args.model:
    try:
        modelfile = load_modelfile(args.model)
    except FileNotFoundError:
        print(f"Could not load model {repository}. Try python create_model.py...")
        sys.exit(1)
        
app = FastAPI()

request_index = 0
def next_request_index():
    global request_index
    request_index += 1
    return request_index


class ServerStatus:
    work_items: list[int] = [0]
    queue_depths: list[int] = [0]
    times: list[float] = [time.time()]

    def update_work_items(self, n):
        if n != self.work_items[-1]:
            self.times.append(time.time())
            self.work_items.append(n)
            self.queue_depths.append(self.queue_depths[-1])

    def update_queue_depths(self, n):
        if n != self.queue_depths[-1]:
            self.times.append(time.time())
            self.queue_depths.append(n)
            self.work_items.append(self.work_items[-1])

status = ServerStatus()

class QueueRequestModelChange(BaseModel):
    modelfile: typing.Any
    
class QueueRequest(BaseModel):
    request_id: str = f"exllamav2-{next_request_index()}"
    messages: list[ChatCompletions.Message]
    completion_queue: typing.Any  # asyncio.Queue
    max_tokens: int | None = None
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.8
    token_repetition_penalty: float = 1.05
    token_presence_penalty: float = 0.0
    token_frequency_penalty: float = 0.0
    stop: list[str]
    stream: bool = False


class QueueResponse(BaseModel):
    content: str
    finish_reason: str | None = None
    completion_tokens: int = 0
    prompt_tokens: int = 0


prompts_queue = asyncio.Queue()

processing_started = False
model = None
tokenizer = None
loras = []
settings_proto = ExLlamaV2Sampler.Settings()


class WorkItem:
    input_ids: list
    output_ids: list
    cache: any
    settings: any
    completion_queue: any
    request: QueueRequest
    completion_tokens: int = 0
    prompt_tokens: int = 0
    first_content: bool = True
    output_offset: int = 0

async def inference_loop():
    global prompts_queue, processing_started, status, settings_proto, modelfile, tokenizer, model
    processing_started = True

    # throttle streaming to 10/s instead of making big JSON HTTP responses for every token
    chunk_interval = 0.1
    next_stream_time = asyncio.get_event_loop().time() + chunk_interval
    
    pending_model_request = None

    work: list[WorkItem] = []

    while processing_started:
        if pending_model_request and not work:
            modelfile = pending_model_request.modelfile
            await load_model()
            pending_model_request = None

        # enter this (possibly blocking) loop if there's nothing to do (ok to block)
        # or if we could accept more work and the queue isn't empty (no blocking)
        added = False
        while pending_model_request is None and (len(work) == 0 or (len(work) < MAX_PROMPTS and prompts_queue.qsize() != 0)):
            try:
                request: QueueRequest|QueueRequestModelChange = await asyncio.wait_for(prompts_queue.get(), 0.5)
            except TimeoutError:
                break
            status.update_queue_depths(prompts_queue.qsize())
            
            if isinstance(request, QueueRequestModelChange):
                pending_model_request = request
                break
            
            item = WorkItem()
            
            chat = ollama_template.Prompt(modelfile).chatString(request.messages)
            print(chat)

            item.input_ids = tokenizer.encode(chat)
            item.prompt_tokens = item.input_ids.shape[-1] - 1
            batch_size = 1
            max_tokens = request.max_tokens or config.max_seq_len
            item.cache = ExLlamaV2Cache(
                model,
                max_seq_len=(item.input_ids.size(1) + max_tokens),
                batch_size=batch_size,
            )
            model.forward(item.input_ids[:, :-1], item.cache, preprocess_only=True)
            item.settings = settings_proto.clone()
            item.settings.temperature = request.temperature
            item.settings.top_p = request.top_p
            item.settings.top_k = request.top_k
            item.settings.token_repetition_penalty = request.token_repetition_penalty
            item.settings.token_presence_penalty = request.token_presence_penalty
            item.settings.token_frequency_penalty = request.token_frequency_penalty
            item.completion_queue = request.completion_queue
            item.request = request
            item.output_ids = torch.empty((1, 0), dtype=torch.long)
            work.append(item)
            added = True
        if added:
            status.update_work_items(len(work))
            print(f"workitems {len(work)}")

        # process as long as there are incomplete requests
        if work:
            send_chunk = False
            now = asyncio.get_event_loop().time()
            if now >= next_stream_time:
                next_stream_time = now + chunk_interval
                send_chunk = True

            inputs = torch.cat([w.input_ids[:, -1:] for w in work], dim=0)
            caches = [w.cache for w in work]
            logits = model.forward(inputs, caches, input_mask=None, loras=loras).float().cpu()

            eos = []
            for i in range(len(work)):
                item = work[i]
                r = random.random()
                token, _, _ = ExLlamaV2Sampler.sample(
                    logits[i: i + 1, :, :], item.settings, item.input_ids, r, tokenizer
                )
                item.output_ids = torch.cat([item.output_ids, token], dim=1)
                item.input_ids = torch.cat([item.input_ids, token], dim=1)
                item.completion_tokens += 1

                stopped = token.item() == tokenizer.eos_token_id
                limited = item.cache.current_seq_len == item.cache.max_seq_len
                final = stopped or limited
                finish_reason = None
                if final:
                    finish_reason = "stop" if stopped and not limited else "length"

                if final or (item.request.stream and send_chunk):
                    try:
                        content = tokenizer.decode(item.output_ids)[0]
                        
                        # this bullshit is because sentencepiece drops leading spaces,
                        # so simply clearing item.output_ids fails here with missing spaces.
                        # instead, we have to decode everything and trim off the already-returned
                        # bits. Fancier (!?) would be to use a random token to stuff on the front
                        pos = len(content)
                        content = content[item.output_offset:]
                        item.output_offset = pos
                        
                        if item.first_content:
                            content = content.lstrip()
                            item.first_content = False
                        if final:
                            content = content.rstrip()
                        response = QueueResponse(content=content, finish_reason=finish_reason,
                                                 prompt_tokens=item.prompt_tokens, completion_tokens=item.completion_tokens)
                        await item.completion_queue.put(response)
                    except Exception as e:
                        print(f"Error processing completed prompt: {e}")
                        final = True
                    if final:
                        eos.insert(0, i)  # Indices of completed prompts
                    else:
                        # reset after sending stream delta
                        pass
                        # see above re: sentencepiece
                        #item.output_ids = torch.empty((1, 0), dtype=torch.long)

            # Remove completed prompts from the list
            for i in eos:
                work.pop(i)
            if eos and (prompts_queue.qsize() == 0 and not pending_model_request):
                print(f"workitems {len(work)}")
                status.update_work_items(len(work))

            # yield to HTTP threads or we can't stream (and batched responses are all as slow as the last one)
            await asyncio.sleep(0)


@app.get("/", response_class=typing.Union[HTMLResponse, FileResponse])
def status_page():
    file_path = pathlib.Path('status.html')

    if file_path.exists():
        return FileResponse(file_path.resolve(), media_type='text/html')
    else:
        return HTMLResponse(f"Server is running model {modelfile.repository} but status.html is missing")

@app.websocket("/ws/status")
async def websocket_status(websocket: WebSocket):
    global status
    
    await websocket.accept()
    while True:
        await asyncio.sleep(1)
        data = [
            { "x": status.times, "y": status.work_items, "name": "run" },
            { "x": status.times, "y": status.queue_depths, "name": "wait" },
        ]
        try:
            await websocket.send_json(data)
        except Exception as e:
            break



@app.post(
    "/v1/chat/completions",
    response_class=typing.Union[StreamingJSONResponse, JSONResponse],
)
async def chat_completions(prompt: ChatCompletions):
    global modelfile, prompts_queue, config, status, model_change_lock

    if not modelfile or prompt.model != modelfile.repository:
        try:
            newmodelfile = load_modelfile(prompt.model)
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail=f"Model \"{prompt.model}\" is not available. Try adding it with create_model.py")
        await prompts_queue.put(QueueRequestModelChange(modelfile=newmodelfile))
    
    # Listify stop
    stop = prompt.stop
    if prompt.stop is None:
        stop = []
    else:
        stop = [stop] if isinstance(stop, str) else stop

    request = QueueRequest(
        messages=prompt.messages,
        completion_queue=asyncio.Queue(0),
        max_tokens=prompt.max_tokens,
        temperature=prompt.temperature,
        top_p=prompt.top_p,
        token_repetition_penalty=1.05,
        token_presence_penalty=prompt.presence_penalty,
        token_frequency_penalty=prompt.frequency_penalty,
        stop=stop,
        stream=prompt.stream,
    )

    await prompts_queue.put(request)
    status.update_queue_depths(prompts_queue.qsize())

    created = int(time.time())  # constant for all chunks according to api docs

    async def gen():
        finish_reason = None
        while finish_reason is None:
            try:
                qresponse: QueueResponse = await asyncio.wait_for(request.completion_queue.get(), timeout=args.timeout)
                finish_reason = qresponse.finish_reason
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Processing the prompt timed out.")
            if request.stream:
                delta = ChatCompletionsChunkResponse.Choice.Delta(content=qresponse.content, role="assistant")
                choice = ChatCompletionsChunkResponse.Choice(finish_reason=finish_reason, index=1, delta=delta)
                response = ChatCompletionsChunkResponse(
                    id=request.request_id,
                    choices=[choice],
                    created=created,
                    model=prompt.model,
                )
                # print(".", end="\n" if finish_reason is not None else "")
                print(qresponse.content, end="\n" if finish_reason is not None else "")
                sys.stdout.flush()
                # print(repr(response))
                yield response
            else:
                if finish_reason is None:
                    raise HTTPException(status_code=505, detail="Tried to stream non-streaming request")
                message = ChatCompletionsResponse.Choice.Message(content=qresponse.content, role="assistant")
                choice = ChatCompletionsResponse.Choice(finish_reason=finish_reason, index=1, message=message)
                usage = ChatCompletionsResponse.Usage(
                    prompt_tokens=qresponse.prompt_tokens,
                    completion_tokens=qresponse.completion_tokens,
                    total_tokens=qresponse.prompt_tokens + qresponse.completion_tokens
                )
                response = ChatCompletionsResponse(
                    id=request.request_id,
                    choices=[choice],
                    created=created,
                    model=prompt.model,
                    usage=usage,
                )
                #print(repr(response))
                yield response

    if request.stream:
        return StreamingJSONResponse(gen())
    else:
        response = await gen().__anext__()
        return JSONResponse(jsonable_encoder(response))


@app.get("/v1/models", response_class=JSONResponse)
async def api_models():
    registry = read_registry()
    models = []
    # make active model first
    if modelfile:
        models.append(ModelsResponse.Model(id=modelfile.repository, created=modelfile.created))
    for k, v in registry.items():
        if modelfile and modelfile.repository == k:
            continue
        models.append(ModelsResponse.Model(id=k, created=v["created"]))
    response = ModelsResponse(data=models)
    return response
    

async def setup_gpu_split():
    global gpu_split
    if not args.gpu_balance:
        gpu_split = list(map(int, args.gpu_split.split(",")))
        return
    
    print("Reading gpu_split...")
    while os.path.exists("gpu_assign.lock"):
        await asyncio.sleep(0.3)
    with open("gpu_assign.lock", "w", encoding="utf-8") as file:
        file.write("")
    # Read the first line, remove it, and write the rest back to the file
    with open("gpu_assign", "r+", encoding="utf-8") as file:
        # Read the first line
        first_line = file.readline().replace("\n", "")

        # Read the rest of the file
        rest_of_content = file.read()

        # Move the cursor to the beginning of the file
        file.seek(0)

        # Write the rest of the content back to the file
        file.write(rest_of_content)

        # Truncate the file to remove any remaining characters from the old content
        file.truncate()
        print(first_line)
    try:
        os.remove("gpu_assign.lock")
    except OSError as e:
        print(f"Error removing lock: {e}")

    gpu_split = list(map(int, first_line.split(",")))


async def load_model():
    global args, model, modelfile, tokenizer, loras, config, MAX_PROMPTS
    
    unload_model()
    
    print("Loading model: " + modelfile.repository)
    print("From: " + modelfile.model_dir)
    
    MAX_PROMPTS = args.max_batch_size or modelfile.max_batch_size or 8
    
    config = ExLlamaV2Config()
    config.model_dir = modelfile.model_dir
    config.prepare()
    
    if args.rope_scale is not None:
        config.scale_pos_emb = args.rope_scale
    elif hasattr(modelfile, 'rope_scale'):
        config.scale_pos_emb = modelfile.rope_scale
        
    if args.rope_alpha is not None:
        config.scale_rope_alpha = args.rope_alpha
    elif hasattr(modelfile, 'rope_alpha'):
        config.scale_rope_alpha = modelfile.rope_alpha
    
    if modelfile.max_seq_len:
        config.max_seq_len = modelfile.max_seq_len
    if args.max_seq_len:
        config.max_seq_len = args.max_seq_len
        
    if modelfile.max_input_len:
        config.max_input_len = modelfile.max_input_len
    if args.max_input_len:
        config.max_input_len = args.max_input_len
        
    config.max_batch_size = MAX_PROMPTS

    model = ExLlamaV2(config)
    if args.gpu_split:
        global gpu_split
        if first and args.num_workers > 1:
            sleep_time = random.uniform(0.1, 3)
            time.sleep(sleep_time)
        model.load(gpu_split=gpu_split)
    else:
        model.load()
    tokenizer = ExLlamaV2Tokenizer(config)
    if modelfile.lora:
        lora = ExLlamaV2Lora.from_directory(model, args.lora)
        loras.append(lora)

    # Embiggen the model x times without increasing memory usage
    for i in range(args.embiggen):
        # mix layers here
        layer_arrangement = list(range(0, 14)) + list(range(4, 22))
        # list(range(8, 18)) +
        # modules arangement: [embedding, [...layers], rms-norm, head]
        # where each layer is [attention, mlp]
        old_modules = model.modules
        model.modules = old_modules[:1]
        for idx in layer_arrangement:
            model.modules += old_modules[idx * 2 + 1: idx * 2 + 3]
        model.modules += old_modules[-2:]
        model.head_layer_idx = len(model.modules) - 1
        model.config.num_hidden_layers = len(layer_arrangement)
        model.last_kv_layer_idx = len(model.modules) - 4
        
    print("Model is loaded.")


def unload_model():
    global model, config, tokenizer, loras
    if model:
        model.unload()
        model = None
        config = None
        tokenizer = None
        for lora in loras:
            lora.unload()
        loras = []
        gc.collect()
        torch.cuda.empty_cache()

@app.on_event("startup")
async def startup_event():
    global args, modelfile
    
    print("Starting up...")
    if args.gpu_split:
        await setup_gpu_split()
    if modelfile:
        if args.gpu_split and args.num_workers > 1:
            await asyncio.sleep(random.uniform(0.1, 3))
        await load_model()
    asyncio.create_task(inference_loop())


@app.on_event("shutdown")
async def shutdown_event():
    global processing_started
    print("Shutting down...")
    processing_started = False
    unload_model()


if __name__ == "__main__":
    import uvicorn

    # Clean up any previous file locks
    if os.path.exists("gpu_assign"):
        print(f"Deleting old gpu assignment file")
        os.remove("gpu_assign")
    if os.path.exists("gpu_assign.lock"):
        print(f"Deleting old gpu lock file")
        os.remove("gpu_assign.lock")

    # global worker_assignments
    # worker_assignments = []
    # Load balance workers across GPUs
    if args.gpu_balance:
        gpus = list(map(int, args.gpu_split.split(",")))
        average_workers = int(args.num_workers / len(gpus))
        content = ""
        for i in range(args.num_workers):
            gpu_mapping = []
            for j in range(len(gpus)):
                # If the number of workers doesn't fit evenly on the cards, distribute the odd ones out. Since exllamav2 doesn't
                # distribute perfectly with --gpu_split, I'm going to just guess at it now with a formula. There's probably a more
                # clever way to split them up perfectly, I just haven't come up with it yet.
                if (
                    i + 1 + args.num_workers % len(gpus) > args.num_workers
                ) and args.num_workers % len(gpus) != 0:
                    # if i % len(gpus) != j:
                    gpu_mapping.append(int(gpus[j] / len(gpus) + 2))
                    # else:
                    # gpu_mapping.append(gpus[j])
                else:
                    if i % len(gpus) == j:
                        gpu_mapping.append(gpus[j])
                    else:
                        gpu_mapping.append(0)
            text_mapping = ",".join(map(str, gpu_mapping))
            content += text_mapping + "\n"
        with open("gpu_assign", "w", encoding="utf-8") as file:
            file.write(content)

    print(f"Starting a server at {args.host} on port {args.port}...")
    uvicorn.run(
        "__main__:app",
        host=args.host,
        port=args.port,
        workers=args.num_workers,
        http="h11",
    )
