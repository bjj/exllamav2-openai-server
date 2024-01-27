
# Given a model directory and the ollama repository path for the same model,
# generate a configuration for that model.

import json
import sys
import os
import tempfile
import argparse
import asyncio
import time
from ollama_registry import get_ollama_model_descriptor

registry_path = "models.json"

# Run exllamav2 from a git checkout in a sibling dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/exllamav2")
from exllamav2 import ExLlamaV2Config


def parse_args():
    parser = argparse.ArgumentParser(description="Add a model description.")
    parser.add_argument("--model-dir", metavar="MODEL_DIRECTORY", type=str, help="Sets model_directory", required=True)
    parser.add_argument("--lora", metavar="LORA_DIRECTORY", type=str, help="Sets lora_directory")
    parser.add_argument("--max-seq-len", metavar="NUM_TOKENS", type=int, help="Sets context length")
    parser.add_argument("--max-input-len", metavar="NUM_TOKENS", type=int, help="Sets input length")
    parser.add_argument("--max-batch-size", metavar="N", type=int, help="Max prompts to process at once")
    parser.add_argument("--rope_alpha", metavar="rope_alpha", type=float, help="Sets rope_alpha")
    parser.add_argument("--rope_scale", metavar="rope_scale", type=float, help="Sets rope_scale")
    parser.add_argument("--system_prompt", metavar="prompt", type=str, help="Override system_prompt")
    parser.add_argument("--cache_8bit", type=bool, help="Use 8 bit kv cache")
    parser.add_argument("repository")
    return parser.parse_args(), parser

def read_registry():
    global registry_path
    try:
        with open(registry_path, 'r') as file:
            registry = json.load(file)
    except FileNotFoundError:
        print(f"Creating new registry {registry_path}")
        registry = {}
    return registry

def write_registry(registry):
    global registry_path
    
    # Try to atomically update the JSON
    temp_fd, temp_path = tempfile.mkstemp(dir='.')
    with os.fdopen(temp_fd, 'w') as temp_file:
        json.dump(registry, temp_file, indent=4)
    os.replace(temp_path, registry_path)
    
async def main():
    args, parser = parse_args()

    registry = read_registry()
    
    # Sanity check the model
    config = ExLlamaV2Config()
    config.model_dir = args.model_dir
    config.prepare()

    # Get ollama's description of the model
    ollama_descr = await get_ollama_model_descriptor(args.repository, debug=True)

    record = {
        "model_dir": args.model_dir,
        "settings": {},
        "ollama": ollama_descr,
        "created": int(time.time()),
    }
    for setting in [x.dest for x in parser._actions if isinstance(x, argparse._StoreAction) and not x.required]:
        if getattr(args, setting) is not None:
            record["settings"][setting] = getattr(args, setting)

    if args.repository in registry:
        print(f"Replacing model {args.repository}, was:\n{json.dumps(registry[args.repository], indent=4)}")
    else:
        print(f"Adding new model {args.repository}")
    registry[args.repository] = record

    write_registry(registry)


if __name__ == "__main__":
    asyncio.run(main())
