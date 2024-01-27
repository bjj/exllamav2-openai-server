# Use ollama style templates to generate chat prompts.
# NOTE: ollama is dependent on Go's text/template and it's not easy to emulate or wrap
# this is just dealing with basic substitutions, not {{if}} shenanigans and so on

import re
import typing
from pydantic import BaseModel
from openai_types import ChatCompletions
from create_model import read_registry


class ModelFile:
    repository: str
    created: int
    template: str
    system_prompt: str = ""
    model_dir: str
    stop: list[str] = []
    lora: str = None
    max_seq_len: int = None
    max_input_len: int = None
    max_batch_size: int = None

    def __init__(self, repository):
        self.repository = repository
        registry = read_registry()
        try:
            record = registry[repository]
        except KeyError:
            raise FileNotFoundError()
        self.model_dir = record["model_dir"]
        self.created = record["created"]
        
        # defaults from ollama
        ollama = record.get("ollama", {})
        ollama_params = ollama.get("params", {})
        self.template = ollama["template"]
        self.system_prompt = ollama.get("system", "")
        self.max_seq_len = ollama_params.get("num_ctx", None)
        self.stop = ollama_params.get("stop", [])

        # override with command line settings from create_model.py
        for k, v in record["settings"].items():
            setattr(self, k, v)
        
class Prompt:
    first: bool = True
    system_prompt: str = ""
    prompt: str = ""
    response: str = ""
    template: str

    result: str = ""

    def __init__(self, model: ModelFile):
        if model.system_prompt is not None:
            self.system_prompt = model.system_prompt
        self.template = model.template.strip(" ")
        self.template = re.sub(r"{{\s+", "{{", self.template)
        self.template = re.sub(r"\s+}}", "}}", self.template)

    def flush(self, template=None):
        if template is None:
            template = self.template
        subbed = template.replace("{{.System}}", self.system_prompt)
        subbed = subbed.replace("{{.Prompt}}", self.prompt)
        if "{{.Response}}" in subbed:
            subbed = subbed.replace("{{.Response}}", self.response)
        else:
            subbed = subbed + self.response

        # we're not fully text/template compatible by a long shot
        if '{{' in subbed:
            raise(f'Incomplete template substitution {template}')

        self.first = False
        self.system_prompt = ""
        self.prompt = ""
        self.response = ""

        self.result += subbed

    def chatString(self, messages: list[ChatCompletions.Message]):
        if self.result:
            raise ("Do not re-use this object")
        for m in messages:
            if m.role == "system":
                # and not self.First => does not match ollama. It
                # would add a whole empty exchange with the model system prompt
                # and then the user system prompt.
                # This does replacement. Concatenation also makes sense
                if self.system_prompt and not self.first:
                    self.flush()
                self.system_prompt = m.content
            elif m.role == "user":
                if self.prompt:
                    self.flush()
                self.prompt = m.content
            elif m.role == "assistant":
                self.response = m.content
                self.flush()
            else:
                pass

        if self.prompt or self.system_prompt:
            pre = self.template.split("{{.Response}}")
            self.flush(pre[0])

        return self.result


def main():
    model = ModelFile()
    p = Prompt(model)
    messages = [
        ChatCompletions.Message(content="MySystemMessage", role="system"),
        ChatCompletions.Message(content="What is my name?", role="user"),
        ChatCompletions.Message(content="King Arthur", role="assistant"),
        ChatCompletions.Message(content="What is my quest?", role="user"),
    ]
    print(p.chatString(messages))

    p = Prompt(model)
    messages = [
        ChatCompletions.Message(content="What is my name?", role="user"),
        ChatCompletions.Message(content="King Arthur", role="assistant"),
        ChatCompletions.Message(content="What is my quest?", role="user"),
    ]
    print(p.chatString(messages))


if __name__ == "__main__":
    main()
