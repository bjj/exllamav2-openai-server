
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
from model_settings import ModelSettings

registry_path = "models.json"

# Run exllamav2 from a git checkout in a sibling dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/exllamav2")
from exllamav2 import ExLlamaV2Config


def parse_args():
    parser = argparse.ArgumentParser(description="Add a model description.")
    parser.add_argument("--model-dir", metavar="MODEL_DIRECTORY", type=str, help="Sets model_directory", required=True)
    parser.add_argument("--no-ollama", action='store_true', help="Make a model without ollama data")
    ModelSettings.add_arguments(parser)
    parser.add_argument("repository")
    args = parser.parse_args()
    return args, ModelSettings.from_args(args)

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
    args, settings = parse_args()

    registry = read_registry()
    
    # Sanity check the model
    config = ExLlamaV2Config()
    config.model_dir = args.model_dir
    config.prepare()

    # Get ollama's description of the model
    if not args.no_ollama:
        ollama_descr = await get_ollama_model_descriptor(args.repository, debug=True)
    else:
        ollama_descr = {}

    record = {
        "model_dir": args.model_dir,
        "settings": settings.dict(),
        "ollama": ollama_descr,
        "created": int(time.time()),
    }

    if args.repository in registry:
        print(f"Replacing model {args.repository}, was:\n{json.dumps(registry[args.repository], indent=4)}")
    else:
        print(f"Adding new model {args.repository}")
    registry[args.repository] = record

    write_registry(registry)


if __name__ == "__main__":
    asyncio.run(main())
