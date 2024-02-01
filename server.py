import sys, os, time, torch, random, asyncio, json, argparse, pathlib, gc
import uuid
import typing
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.websockets import WebSocket
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.encoders import jsonable_encoder
from openai_types import *
from fastapi_helpers import StreamingJSONResponse
import ollama_template
from create_model import read_registry

# Run exllamav2 from a git checkout in a sibling dir
#sys.path.append(
#    os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/exllamav2"
#)

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
                        help="Sets array gpu_split and accepts input like 16,24. Default is automatic")
    parser.add_argument("--rope_alpha", metavar="rope_alpha", type=float, default=1.0, help="Sets rope_alpha")
    parser.add_argument("--rope_scale", metavar="rope_scale", type=float, help="Sets rope_scale")
    parser.add_argument("--cache_8bit", action="store_true", help="Use 8 bit cache")

    return parser.parse_args()


args = parse_args()

what = torch.inference_mode()

def load_modelfile(repository):
    return ollama_template.ModelFile(repository)

loaded_pct = None
modelfile = None
if args.model:
    try:
        modelfile = load_modelfile(args.model)
    except FileNotFoundError:
        print(f"Could not load model {args.model}. Try python create_model.py...")
        sys.exit(1)
        
app = FastAPI()

class ServerStatus:
    work_item_times: list[float] = [time.time()]
    work_items: list[int] = [0]
    
    queue_depth_times: list[float] = [time.time()]
    queue_depths: list[int] = [0]

    token_rate_times: list[float] = [time.time()]
    token_rates: list[float] = [0.0]

    def update_work_items(self, n):
        if n != self.work_items[-1]:
            self.work_item_times.append(time.time())
            self.work_items.append(n)

    def update_queue_depths(self, n):
        if n != self.queue_depths[-1]:
            self.queue_depth_times.append(time.time())
            self.queue_depths.append(n)
    def increment_queue_depths(self):
        self.update_queue_depths(self.queue_depths[-1] + 1)
            
    def update_token_rates(self, n, offset=0, force=False):
        if n != self.token_rates[-1] or force:
            self.token_rate_times.append(time.time() + offset)
            self.token_rates.append(n)

status = ServerStatus()

class QueueRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
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
    status_code: int = 200
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
        token, prob, eos = ExLlamaV2Sampler.sample(logits, gen_settings, self.sequence_ids[:1, :], random.random(), self.tokenizer, prefix_token)

        if self.sequence_ids.shape[0] > 1 and token.shape[0] == 1:
            self.sequence_ids = torch.cat([self.sequence_ids, token.repeat(self.sequence_ids.shape[0], 1)], dim = 1)
        else:
            self.sequence_ids = torch.cat([self.sequence_ids, token], dim = 1)

        gen_settings.feed_filters(token)
        if hasattr(self, "no_probs"):  # hacks upon hacks: try to support 0.12
            return token, prob, eos
        else:
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
    global prompts_queue, processing_started, status, settings_proto, modelfile, tokenizer, model, config
    processing_started = True

    # throttle streaming to 10/s instead of making big JSON HTTP responses for every token
    chunk_interval = 0.1
    next_stream_time = asyncio.get_event_loop().time() + chunk_interval
    
    pending_model_request = None

    work: list[WorkItem] = []

    async def add_workitem(request):
        if request.finish_reason:
            return False
        
        chat = ollama_template.Prompt(modelfile).chatString(request.messages)
        #print(chat)

        input_ids = tokenizer.encode(chat)
        n_input_tokens = input_ids.shape[-1]
        print(f"num input tokens {n_input_tokens}")
        if n_input_tokens >= config.max_seq_len:
            response = QueueResponse(status_code=403, finish_reason="tokens_exceeded_error",
                                     content=f"Input tokens exceeded. Model limit: {config.max_seq_len}.")
            await request.completion_queue.put(response)
            return False
        
        item = WorkItem()
        item.prompt_tokens = n_input_tokens
        batch_size = 1
        max_tokens = request.max_tokens or config.max_seq_len
        CacheClass = ExLlamaV2Cache_8bit if args.cache_8bit else ExLlamaV2Cache
        item.cache = CacheClass(model, max_seq_len=min(config.max_seq_len, (n_input_tokens + max_tokens)), batch_size=batch_size)
        item.generator = ExLlamaV2StreamingGenerator(model, item.cache, tokenizer)
        item.generator.set_stop_conditions([tokenizer.eos_token_id, *request.stop, *modelfile.stop])
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
        return True
    
    token_rate_start_time = asyncio.get_event_loop().time()
    token_rate_count = 0
    def update_token_rates(force=False):
        nonlocal token_rate_start_time, token_rate_count
        now = asyncio.get_event_loop().time()
        duration = now - token_rate_start_time
        if duration < 1.0 and not force:
            return
        if duration > 0:
            status.update_token_rates(token_rate_count / duration, -duration / 2, force)
        token_rate_start_time = now
        token_rate_count = 0
    
    def update_queue_depths():
        global prompts_queue
        nonlocal pending_model_request
        status.update_queue_depths(prompts_queue.qsize() + (1 if pending_model_request is not None else 0))
        
    while processing_started:
        added = False
        
        # If we need a new model, handle that when the work queue drains
        if pending_model_request and not work:
            update_token_rates(True)
            try:
                modelfile = pending_model_request.modelfile
                await load_model()
                if await add_workitem(pending_model_request):
                    added = True
            except:
                response = QueueResponse(status_code=500, finish_reason="server_error",
                                         content=f"Unable to load requested model {modelfile.repository}.")
                await pending_model_request.completion_queue.put(response)
                modelfile = None
            pending_model_request = None
            update_queue_depths()
            update_token_rates(True)
            status.update_token_rates(0, force=True) # need a better way to handle this boundary

        # If pending model request, do not add more work items.
        # Else enter this (possibly blocking) loop if there's nothing to do (ok to block)
        #      or if we could accept more work and the queue isn't empty (no blocking)
        while pending_model_request is None and (len(work) == 0 or (len(work) < MAX_PROMPTS and prompts_queue.qsize() != 0)):
            try:
                request: QueueRequest = await asyncio.wait_for(prompts_queue.get(), 0.5)
            except TimeoutError:
                update_token_rates()
                break
            
            if request.finish_reason:
                update_queue_depths()
                continue
            
            if modelfile is None or request.modelfile.repository != modelfile.repository:
                pending_model_request = request
                break
            
            if await add_workitem(request):
                added = True
        
        update_queue_depths()
        if added:
            status.update_work_items(len(work))
            update_token_rates(len(work) == 1)

        # process as long as there are incomplete requests
        if work:
            send_chunk = False
            now = asyncio.get_event_loop().time()
            if now >= next_stream_time:
                next_stream_time = now + chunk_interval
                send_chunk = True
                update_token_rates()

            inputs = torch.cat([w.generator.sequence_ids[:, -1:] for w in work], dim=0)
            caches = [w.cache for w in work]
            # NOTE: can run out of memory here. Need to handle that. torch.cuda.OutOfMemoryError
            logits = model.forward(inputs, caches, input_mask=None, loras=loras).float()
            event = torch.cuda.Event()
            event.record(torch.cuda.default_stream())
            token_rate_count += len(work)
            
            # yield to HTTP threads or we can't stream (and batched responses are all as slow as the last one).
            # Without the loop, other things will not get enough time to run (if you have a stack of functions
            # yielding values, only one will run each sleep(0) while making the next runnable).
            # Sleeping for nonzero time here is almost guaranteed to wait too long due to sleep granularity.
            while not event.query():
                await asyncio.sleep(0)

            # sync with GPU
            logits = logits.cpu()

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
            if eos:
                gc.collect()
                torch.cuda.empty_cache()
            if not work and prompts_queue.qsize() == 0:
                update_token_rates(True)
            if eos and (prompts_queue.qsize() == 0 and not pending_model_request):
                status.update_work_items(len(work))


@app.get("/", response_class=typing.Union[HTMLResponse, FileResponse])
def status_page():
    file_path = pathlib.Path('static/status.html')

    if file_path.exists():
        return FileResponse(file_path.resolve(), media_type='text/html')
    else:
        return HTMLResponse(f"Server is running model {modelfile.repository} but status.html is missing")

@app.websocket("/ws/status")
async def websocket_status(websocket: WebSocket):
    global status, loaded_pct
    
    await websocket.accept()
    while True:
        await asyncio.sleep(1)
        model_name = None
        if modelfile:
            model_name = modelfile.repository
            if loaded_pct is not None and loaded_pct < 100:
                model_name = f"{model_name} {loaded_pct:.1f}%"
        data = {
            "model": model_name,
            "queues": [
                { "x": status.work_item_times, "y": status.work_items, "name": "run" },
                { "x": status.queue_depth_times, "y": status.queue_depths, "name": "wait" },                
            ],
            "rates": [
                { "x": status.token_rate_times, "y": status.token_rates, "name": "tok/s" },
            ],
        }
        try:
            await websocket.send_json(data)
        except Exception as e:
            break


# This is meant to be returned. If raised, catch yourself and return
# Maybe there's a cleaner way to do this by inheriting from HTTPException
class ApiErrorResponse(JSONResponse, Exception):
    def __init__(
        self, *,
        status_code: int,
        message: str,
        type: str,
        param: typing.Any = None,
        code: typing.Any = None
    ):
        error = ErrorResponse.Error(message=message, type=type, param=param, code=code)
        response = ErrorResponse(error=error)
        super().__init__(status_code=status_code, content=jsonable_encoder(response))

last_queued_modelfile = None

@app.post(
    "/v1/chat/completions",
    response_class=typing.Union[StreamingJSONResponse, JSONResponse, ApiErrorResponse],
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
            return ApiErrorResponse(status_code=400, type="invalid_request_error",
                                    message=f"Model \"{prompt.model}\" is not available. Try adding it with create_model.py")
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
    status.increment_queue_depths()

    created = int(time.time())  # constant for all chunks according to api docs
    asyncio.create_task(poll_is_disconnected(fastapi_request, request))

    async def gen():
        while request.finish_reason is None:
            try:
                qresponse: QueueResponse = await asyncio.wait_for(request.completion_queue.get(), timeout=args.timeout)
                request.finish_reason = qresponse.finish_reason
            except asyncio.TimeoutError:
                request.finish_reason = "timeout" # abort inference
                raise ApiErrorResponse(status_code=408, type="timeout", message=f"Processing did not complete within {args.timeout} seconds.")
            if qresponse.status_code >= 300:
                raise ApiErrorResponse(status_code=qresponse.status_code, type=qresponse.finish_reason,
                                       message=qresponse.content)

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
                #sys.stdout.flush()
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

    try:
        # catch error from first gen. I don't love this
        response = await gen().__anext__()
        if request.stream:
            async def concat(first, rest):
                yield first
                async for next in rest:
                    yield next
            return StreamingJSONResponse(concat(response, gen()))
        else:
            return JSONResponse(jsonable_encoder(response))
    except ApiErrorResponse as e:
        return e

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
    gpu_split = None
    if args.gpu_split:
        gpu_split = list(map(float, args.gpu_split.split(",")))

async def load_model():
    global args, model, modelfile, tokenizer, loras, config, MAX_PROMPTS, gpu_split, loaded_pct
    
    unload_model()
    
    print("Loading model: " + modelfile.repository)
    print("From: " + modelfile.model_dir)
    
    MAX_PROMPTS = args.max_batch_size or modelfile.max_batch_size or 8
    
    try:
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

        # use loading status callback to yield to web thread
        def callback_gen(idx, total):
            yield 100.0 * idx / total
        model = ExLlamaV2(config)
        if gpu_split is not None:
            loader = model.load_gen(gpu_split=gpu_split, callback_gen=callback_gen)
        else:
            CacheClass = ExLlamaV2Cache_8bit if args.cache_8bit else ExLlamaV2Cache
            scratch_cache = CacheClass(model, max_seq_len=config.max_seq_len, batch_size=config.max_batch_size, lazy = True)
            loader = model.load_autosplit_gen(scratch_cache, callback_gen=callback_gen)
        for pct in loader:
            loaded_pct = pct
            await asyncio.sleep(0)
            
        tokenizer = ExLlamaV2Tokenizer(config)
        if modelfile.lora:
            lora = ExLlamaV2Lora.from_directory(model, args.lora)
            loras.append(lora)
    except Exception as e:
        import traceback
        traceback.print_exception(e);
        print(f"Exception loading {modelfile.repository}: {str(e)}")
        unload_model()
        raise
        
    loaded_pct = None
    print(f"Model is loaded. {torch.cuda.max_memory_allocated()} CUDA bytes allocated")


def unload_model():
    global model, config, tokenizer, loras
    do_print = bool(model or loras)
    if model:
        model.unload()
    model = None
    config = None
    tokenizer = None
    loaded_pct = None
    for lora in loras:
        lora.unload()
    loras = []
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    if do_print:
        print(f"After unload, {torch.cuda.max_memory_allocated()} CUDA bytes allocated")

@app.on_event("startup")
async def startup_event():
    global args, modelfile
    
    print("Starting up...")
    await setup_gpu_split()
    if modelfile:
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

    print(f"Starting a server at {args.host} on port {args.port}...")
    uvicorn.run(
        "__main__:app",
        host=args.host,
        port=args.port,
        http="h11",
    )
