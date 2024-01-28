import sys, os, time, torch, random, asyncio, json, argparse, pathlib, gc
import typing
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.websockets import WebSocket
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
from exllamav2.generator import ExLlamaV2Sampler, ExLlamaV2StreamingGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="OpenAI compatible server for exllamav2.")
    parser.add_argument("--verbose", action="store_true", default=False, help="Sets verbose")
    parser.add_argument("--model", metavar="REPOSITORY", type=str, help="Initial model to load")
    parser.add_argument("--host", metavar="HOST", type=str, default="0.0.0.0", help="Sets host")
    parser.add_argument("--port", metavar="PORT", type=int, default=8000, help="Sets port")
    parser.add_argument("--timeout", metavar="TIMEOUT", type=float, default=120.0, help="Sets HTTP timeout")
    parser.add_argument("--max-seq-len", metavar="NUM_TOKENS", type=int, help="Sets context length")
    parser.add_argument("--max-input-len", metavar="NUM_TOKENS", type=int, help="Sets input length")
    parser.add_argument("--max-batch-size", metavar="N", type=int, help="Max prompt batch size")
    parser.add_argument("--gpu_split", metavar="GPU_SPLIT", type=str, default="",
                        help="Sets array gpu_split and accepts input like 16,24")
    parser.add_argument(
        "--gpu_balance",
        action="store_true",
        default=False,
        help="Balance workers on GPUs to maximize throughput. Make sure --gpu_split is set to the full memory of all cards."
    )
    parser.add_argument("--rope_alpha", metavar="rope_alpha", type=float, default=1.0, help="Sets rope_alpha")
    parser.add_argument("--rope_scale", metavar="rope_scale", type=float, help="Sets rope_scale")
    parser.add_argument("--cache_8bit", action="store_true", help="Use 8 bit cache")
    parser.add_argument("--num_workers", metavar="NUM_WORKERS", type=int,
                        default=1, help="Number of worker processes to use")

    return parser.parse_args()


args = parse_args()

what = torch.inference_mode()

def load_modelfile(repository):
    return ollama_template.ModelFile(repository)

modelfile = None
if args.model:
    try:
        modelfile = load_modelfile(args.model)
    except FileNotFoundError:
        print(f"Could not load model {args.model}. Try python create_model.py...")
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

class QueueRequest(BaseModel):
    request_id: str = f"exllamav2-{next_request_index()}"
    modelfile: typing.Any
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
    finish_reason: str | None = None

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


# We need the power of ExLlamaV2StreamingGenerator but we want to
# batch, so we replace the actual inference in this inner function.
# Ideally refactor exllamav2 so this is not necessary
def patch_gen_single_token(sampler):
    def _gen_single_token(self, gen_settings, prefix_token = None):
        if self.draft_model is not None:
            raise NotImplementedError
        
        logits = self.logits_queue.pop(0)
        token, _, eos = ExLlamaV2Sampler.sample(logits, gen_settings, self.sequence_ids[:1, :], random.random(), self.tokenizer, prefix_token)

        if self.sequence_ids.shape[0] > 1 and token.shape[0] == 1:
            self.sequence_ids = torch.cat([self.sequence_ids, token.repeat(self.sequence_ids.shape[0], 1)], dim = 1)
        else:
            self.sequence_ids = torch.cat([self.sequence_ids, token], dim = 1)

        gen_settings.feed_filters(token)
        return token, eos
    
    sampler.logits_queue = []
    sampler._gen_single_token = _gen_single_token.__get__(sampler, ExLlamaV2StreamingGenerator)

    
class WorkItem:
    generator: any
    output_str: str = ""
    cache: any
    settings: any
    completion_queue: any
    request: QueueRequest
    completion_tokens: int = 0
    prompt_tokens: int = 0
    first_content: bool = True

async def inference_loop():
    global prompts_queue, processing_started, status, settings_proto, modelfile, tokenizer, model
    processing_started = True

    # throttle streaming to 10/s instead of making big JSON HTTP responses for every token
    chunk_interval = 0.1
    next_stream_time = asyncio.get_event_loop().time() + chunk_interval
    
    pending_model_request = None

    work: list[WorkItem] = []

    def add_workitem(request):
        if request.finish_reason:
            return
        
        item = WorkItem()
        
        chat = ollama_template.Prompt(modelfile).chatString(request.messages)
        #print(chat)

        input_ids = tokenizer.encode(chat)
        item.prompt_tokens = input_ids.shape[-1] - 1
        batch_size = 1
        max_tokens = request.max_tokens or config.max_seq_len
        CacheClass = ExLlamaV2Cache_8bit if args.cache_8bit else ExLlamaV2Cache
        item.cache = CacheClass(model, max_seq_len=(input_ids.size(1) + max_tokens), batch_size=batch_size)
        item.generator = ExLlamaV2StreamingGenerator(model, item.cache, tokenizer)
        item.generator.set_stop_conditions([tokenizer.eos_token_id, *request.stop])
        patch_gen_single_token(item.generator)
        item.settings = settings_proto.clone()
        item.settings.temperature = request.temperature
        item.settings.top_p = request.top_p
        item.settings.top_k = request.top_k
        item.settings.token_repetition_penalty = request.token_repetition_penalty
        item.settings.token_presence_penalty = request.token_presence_penalty
        item.settings.token_frequency_penalty = request.token_frequency_penalty
        item.completion_queue = request.completion_queue
        item.request = request
        token_healing_must_be_false = False # see below
        item.generator.begin_stream(input_ids, item.settings, loras=loras, token_healing=token_healing_must_be_false)
        work.append(item)
        
    while processing_started:
        added = False
        
        # If we need a new model, handle that when the work queue drains
        if pending_model_request and not work:
            modelfile = pending_model_request.modelfile
            await load_model()
            add_workitem(pending_model_request)
            added = True
            pending_model_request = None

        # If pending model request, do not add more work items.
        # Else enter this (possibly blocking) loop if there's nothing to do (ok to block)
        #      or if we could accept more work and the queue isn't empty (no blocking)
        while pending_model_request is None and (len(work) == 0 or (len(work) < MAX_PROMPTS and prompts_queue.qsize() != 0)):
            try:
                request: QueueRequest = await asyncio.wait_for(prompts_queue.get(), 0.5)
            except TimeoutError:
                break
            status.update_queue_depths(prompts_queue.qsize())
            
            if request.finish_reason:
                continue
            
            if modelfile is None or request.modelfile.repository != modelfile.repository:
                pending_model_request = request
                break
            
            add_workitem(request)
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

            inputs = torch.cat([w.generator.sequence_ids[:, -1:] for w in work], dim=0)
            caches = [w.cache for w in work]
            logits = model.forward(inputs, caches, input_mask=None, loras=loras).float().cpu()

            eos = []
            for i in range(len(work)):
                item = work[i]
                
                item.completion_tokens += 1
                item.generator.logits_queue.append(logits[i: i + 1, :, :])
                # with token_healing off, this queue only needs depth 1.
                # Continuing here can't work because we must update item.generator.sequence_ids before
                # generating the next batch. So more invasive changes to ExLlamaV2StreamingGenerator
                # would be required.
                #if len(item.generator.logits_queue) < 2: # from inspection, most .stream() will consume
                #    continue                

                chunk, stopped, tokens = item.generator.stream()
                item.output_str += chunk

                limited = item.cache.current_seq_len >= item.cache.max_seq_len
                final = stopped or limited
                finish_reason = None
                if final:
                    finish_reason = "stop" if stopped and not limited else "length"

                if item.request.finish_reason:
                    final = True
                    finish_reason = item.request.finish_reason
                    
                if final or (item.request.stream and send_chunk and item.output_str):
                    try:
                        content = item.output_str
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
                        item.output_str = ""

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
    file_path = pathlib.Path('static/status.html')

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
        data = {
            "model": modelfile.repository if modelfile else None,
            "queues": [
                { "x": status.times, "y": status.work_items, "name": "run" },
                { "x": status.times, "y": status.queue_depths, "name": "wait" },                
            ],
        }
        try:
            await websocket.send_json(data)
        except Exception as e:
            break



last_queued_modelfile = None

@app.post(
    "/v1/chat/completions",
    response_class=typing.Union[StreamingJSONResponse, JSONResponse],
)
async def chat_completions(fastapi_request: Request, prompt: ChatCompletions):
    global modelfile, prompts_queue, config, status, last_queued_modelfile

    # if idle, initialize
    if prompts_queue.qsize() == 0:
        last_queued_modelfile = modelfile
        
    if not last_queued_modelfile or prompt.model != last_queued_modelfile.repository:
        try:
            newmodelfile = load_modelfile(prompt.model)
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail=f"Model \"{prompt.model}\" is not available. Try adding it with create_model.py")
        last_queued_modelfile = newmodelfile
        
    # Listify stop
    stop = prompt.stop
    if prompt.stop is None:
        stop = []
    else:
        stop = [stop] if isinstance(stop, str) else stop
    
    # what a terrible interface Request.is_disconnected() is
    async def poll_is_disconnected(fastapi_request, request):
        try:
            while not request.finish_reason and not await fastapi_request.is_disconnected():
                await asyncio.sleep(0.5)
            if not request.finish_reason:
                print(">> Client disconnected!")
                request.finish_reason = "disconnected"
        except asyncio.CancelledError:
            pass
            
    request = QueueRequest(
        modelfile=last_queued_modelfile,
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
    asyncio.create_task(poll_is_disconnected(fastapi_request, request))

    async def gen():
        while request.finish_reason is None:
            try:
                qresponse: QueueResponse = await asyncio.wait_for(request.completion_queue.get(), timeout=args.timeout)
                request.finish_reason = qresponse.finish_reason
            except asyncio.TimeoutError:
                request.finish_reason = "timeout"
                raise HTTPException(status_code=504, detail="Processing the prompt timed out.")
            if request.stream:
                delta = ChatCompletionsChunkResponse.Choice.Delta(content=qresponse.content, role="assistant")
                choice = ChatCompletionsChunkResponse.Choice(finish_reason=request.finish_reason, index=1, delta=delta)
                response = ChatCompletionsChunkResponse(
                    id=request.request_id,
                    choices=[choice],
                    created=created,
                    model=prompt.model,
                )
                # print(".", end="\n" if finish_reason is not None else "")
                #print(qresponse.content, end="\n" if request.finish_reason is not None else "")
                sys.stdout.flush()
                # print(repr(response))
                yield response
            else:
                if request.finish_reason is None:
                    raise HTTPException(status_code=505, detail="Tried to stream non-streaming request")
                message = ChatCompletionsResponse.Choice.Message(content=qresponse.content, role="assistant")
                choice = ChatCompletionsResponse.Choice(finish_reason=request.finish_reason, index=1, message=message)
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
        model.load(gpu_split=gpu_split)
    else:
        model.load()
    tokenizer = ExLlamaV2Tokenizer(config)
    if modelfile.lora:
        lora = ExLlamaV2Lora.from_directory(model, args.lora)
        loras.append(lora)
        
    print(f"Model is loaded. {torch.cuda.max_memory_allocated()} CUDA bytes allocated")


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
        gc.collect()
        print(f"After unload, {torch.cuda.max_memory_allocated()} CUDA bytes allocated")

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
