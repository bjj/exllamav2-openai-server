# Use ollama style templates to generate chat prompts.
# NOTE: ollama is dependent on Go's text/template and it's not easy to emulate or wrap
# this is just dealing with basic substitutions, not {{if}} shenanigans and so on

import re, platform
import typing
from pydantic import BaseModel
from openai_types import ChatCompletions
from create_model import read_registry
from model_settings import ModelSettings

def _windows_to_wsl2_path(windows_path):
    # Convert backslashes to forward slashes
    wsl_path = windows_path.replace('\\', '/')

    # Replace the drive letter and colon (e.g., "C:") with "/mnt/c"
    if wsl_path[1:3] == ':/':
        wsl_path = '/mnt/' + wsl_path[0].lower() + wsl_path[2:]

    return wsl_path

class ModelFile:
    repository: str
    model_dir: str
    created: int
    settings: ModelSettings
    our_settings: ModelSettings
    ollama_settings: ModelSettings

    def __init__(self, repository):
        self.repository = repository
        registry = read_registry()
        try:
            record = registry[repository]
        except KeyError:
            raise FileNotFoundError()

        self.model_dir = record["model_dir"]
        if platform.system() != "Windows":
            self.model_dir = _windows_to_wsl2_path(self.model_dir)
        self.created = record["created"]

        # defaults from ollama
        ollama = record.get("ollama", {})
        ollama_params = ollama.get("params", {})
        self.ollama_settings = ModelSettings(
            template=ollama.get("template"),
            system_prompt=ollama.get("system"),
            max_seq_len=ollama_params.get("num_ctx"),
            stop=ollama_params.get("stop", []),
        )

        self.our_settings = ModelSettings(**record.get("settings", {}))

        self.settings = self.our_settings.copy(deep=True)
        self.settings.inherit_from(self.ollama_settings)

class Prompt:
    first: bool = True
    system_prompt: str = ""
    prompt: str = ""
    response: str = ""
    template: str

    result: str = ""

    def __init__(self, settings: ModelSettings):
        if settings.system_prompt is not None:
            self.system_prompt = settings.system_prompt
        self.template = settings.template.strip(" ")
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
    model = ModelFile(repository="hello")
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
