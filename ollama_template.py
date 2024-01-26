# Use ollama style templates to generate chat prompts.
# NOTE: ollama is dependent on Go's text/template and it's not easy to emulate or wrap
# this is just dealing with basic substitutions, not {{if}} shenanigans and so on

import re
import typing
from pydantic import BaseModel
from openai_types import ChatCompletionsMessage

deepseek_coder_template = (
    "{{ .System }}\n### Instruction:\n{{ .Prompt }}\n### Response:\n"
)
deepseek_coder_system_prompt = """
You are an AI programming assistant, utilizing the Deepseek 
Coder model, developed by Deepseek Company, and you only answer 
questions related to computer science. For politically 
sensitive questions, security and privacy issues, and other 
non-computer science questions, you will refuse to answer.
"""


class ModelFile:
    template: str = deepseek_coder_template
    system_prompt: str = deepseek_coder_system_prompt


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

        self.first = False
        self.system_prompt = ""
        self.prompt = ""
        self.response = ""

        self.result += subbed

    def chatString(self, messages: list[ChatCompletionsMessage]):
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
        ChatCompletionsMessage(content="MySystemMessage", role="system"),
        ChatCompletionsMessage(content="What is my name?", role="user"),
        ChatCompletionsMessage(content="King Arthur", role="assistant"),
        ChatCompletionsMessage(content="What is my quest?", role="user"),
    ]
    print(p.chatString(messages))

    p = Prompt(model)
    messages = [
        ChatCompletionsMessage(content="What is my name?", role="user"),
        ChatCompletionsMessage(content="King Arthur", role="assistant"),
        ChatCompletionsMessage(content="What is my quest?", role="user"),
    ]
    print(p.chatString(messages))


if __name__ == "__main__":
    main()
