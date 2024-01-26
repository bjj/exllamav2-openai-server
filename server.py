import sys, os, time, torch, random, asyncio, json, argparse
import typing
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from openai_types import *
from fastapi_helpers import StreamingJSONResponse
import ollama_template

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
    parser = argparse.ArgumentParser(description="Command line arguments for a Python script.")
    parser.add_argument("--verbose", action="store_true", default=False, help="Sets verbose")
    parser.add_argument("--model", metavar="MODEL_DIRECTORY", type=str, help="Sets model_directory")
    parser.add_argument("--lora", metavar="LORA_DIRECTORY", type=str, help="Sets lora_directory")
    parser.add_argument("--host", metavar="HOST", type=str, default="0.0.0.0", help="Sets host")
    parser.add_argument("--port", metavar="PORT", type=int, default=8000, help="Sets port")
    parser.add_argument("--max-model-len", metavar="MAX_SEQ_LEN", type=int, help="Sets max_seq_len")
    parser.add_argument("--max-input-len", metavar="MAX_INPUT_LEN", type=int, help="Sets max_input_len")
    parser.add_argument("--gpu_split", metavar="GPU_SPLIT", type=str, default="",
                        help="Sets array gpu_split and accepts input like 16,24",)
    parser.add_argument(
        "--gpu_balance",
        action="store_true",
        default=False,
        help="Balance workers on GPUs to maximize throughput. Make sure --gpu_split is set to the full memory of all cards.",
    )
    parser.add_argument("--max_prompts", metavar="MAX_PROMPTS", type=int,
                        default=16, help="Max prompts to process at once", )
    parser.add_argument("--timeout", metavar="TIMEOUT", type=float, default=120.0, help="Sets timeout")
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
print(f"Model Directory: {args.model}")
# Maximum number of generations to hold in memory before forcing a wait on new requests.
MAX_PROMPTS = args.max_prompts

app = FastAPI()


request_index = 0


def next_request_index():
    global request_index
    request_index += 1
    return request_index


# Globals to store states
class QueueRequest(BaseModel):
    request_id: str = f"exllamav2-{next_request_index()}"
    ids: typing.Any
    completion_queue: typing.Any  # asyncio.Queue
    max_tokens: int
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
modelfile = None
tokenizer = None
loras = []


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


async def inference_loop():
    global prompts_queue, processing_started
    processing_started = True

    settings_proto = ExLlamaV2Sampler.Settings()

    # throttle streaming to 10/s instead of making big JSON HTTP responses for every token
    chunk_interval = 0.1
    next_stream_time = asyncio.get_event_loop().time() + chunk_interval

    work: list[WorkItem] = []

    while processing_started:
        # enter this (possibly blocking) loop if there's nothing to do (ok to block)
        # or if we could accept more work and the queue isn't empty (no blocking)
        added = False
        while len(work) == 0 or (len(work) < MAX_PROMPTS and prompts_queue.qsize() != 0):
            try:
                request: QueueRequest = await asyncio.wait_for(prompts_queue.get(), 0.5)
            except TimeoutError:
                break
            item = WorkItem()
            item.input_ids = request.ids
            item.prompt_tokens = request.ids.shape[-1] - 1
            batch_size = 1
            item.cache = ExLlamaV2Cache(
                model,
                max_seq_len=(request.ids.size(1) + request.max_tokens),
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

                if final or (request.stream and send_chunk):
                    try:
                        content = tokenizer.decode(item.output_ids)[0]
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
                        item.output_ids = torch.empty((1, 0), dtype=torch.long)

            # Remove completed prompts from the list
            for i in eos:
                work.pop(i)
            if eos and prompts_queue.qsize() == 0:
                print(f"workitems {len(work)}")

            # yield to HTTP threads or we can't stream (and batched responses are all as slow as the last one)
            await asyncio.sleep(0)

        #current_time = time.time()
        #time_elapsed_seconds = current_time - token_processing_start_time
        #total_processing_time += time_elapsed_seconds
        #read_speed = token_count["read_tokens"] / time_elapsed_seconds
        #generation_speed = token_count["gen_tokens"] / time_elapsed_seconds
        #average_gen_speed = token_count["total_tokens"] / total_processing_time
#
        # Log stats to the console
        # print(
        #    f"Batch process done. Read {token_count['read_tokens']} tokens at {read_speed:.2f} tokens/s. "
        #    f"Generated {token_count['gen_tokens']} tokens at {generation_speed:.2f} tokens/s.\n"
        #    f"This thread generated a total of {token_count['total_tokens']} tokens at {average_gen_speed:.2f} tokens/s."
        # )
        # token_processing_start_time = None  # Reset the start time


@app.get("/")
def read_root():
    return {"message": "ExLlamaV2 Language Model API is running."}


@app.post(
    "/v1/chat/completions",
    response_class=typing.Union[StreamingJSONResponse, JSONResponse],
)
async def chat_completions(prompt: ChatCompletions):
    global modelfile, prompts_queue, token_count, config

    # Listify stop
    stop = prompt.stop
    if prompt.stop is None:
        stop = []
    else:
        stop = [stop] if isinstance(stop, str) else stop

    chat = ollama_template.Prompt(modelfile).chatString(prompt.messages)
    print(chat)

    request = QueueRequest(
        ids=tokenizer.encode(chat),
        completion_queue=asyncio.Queue(0),
        max_tokens=prompt.max_tokens or config.max_seq_len,
        temperature=prompt.temperature,
        top_p=prompt.top_p,
        token_repetition_penalty=1.05,
        token_presence_penalty=prompt.presence_penalty,
        token_frequency_penalty=prompt.frequency_penalty,
        stop=stop,
        stream=prompt.stream,
    )

    await prompts_queue.put(request)

    created = int(time.time())  # constant for all chunks according to api docs

    async def gen():
        finish_reason = None
        while finish_reason is None:
            try:
                qresponse: QueueResponse = await asyncio.wait_for(request.completion_queue.get(), timeout=args.timeout) finish_reason = qresponse.finish_reason
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
                    total_tokens=qresponse.prompt_tokens +
                    qresponse.completion_tokens)
                response = ChatCompletionsResponse(
                    id=request.request_id,
                    choices=[choice],
                    created=created,
                    model=prompt.model,
                    usage=usage,
                )
                print(repr(response))
                yield response

    if request.stream:
        return StreamingJSONResponse(gen())
    else:
        response = await gen().__anext__()
        return JSONResponse(jsonable_encoder(response))


def setup_model():
    global model, modelfile, tokenizer, loras, config
    model_directory = args.model
    config = ExLlamaV2Config()
    config.model_dir = model_directory
    config.prepare()
    if args.rope_scale is not None:
        config.scale_pos_emb = args.rope_scale
    if args.rope_alpha is not None:
        config.scale_rope_alpha = args.rope_alpha
    if args.max_model_len is not None:
        config.max_seq_len = args.max_model_len
    if args.max_input_len is not None:
        config.max_input_len = args.max_input_len
    config.max_batch_size = args.max_prompts

    print("Loading model: " + model_directory)
    modelfile = ollama_template.ModelFile()  # XXX specific
    model = ExLlamaV2(config)
    if args.gpu_split:
        sleep_time = random.uniform(0.1, 3)
        time.sleep(sleep_time)
        if args.gpu_balance:
            while os.path.exists("gpu_assign.lock"):
                time.sleep(0.3)
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

            gpus = list(map(int, first_line.split(",")))

        else:
            gpus = list(map(int, args.gpu_split.split(",")))
        model.load(gpu_split=gpus)
    else:
        model.load()
    tokenizer = ExLlamaV2Tokenizer(config)
    print("Model is loaded.")
    if args.lora:
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


@app.on_event("startup")
async def startup_event():
    print("Starting up...")
    setup_model()
    asyncio.create_task(inference_loop())


@app.on_event("shutdown")
async def shutdown_event():
    global processing_started
    processing_started = False
    print("Shutting down...")


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
