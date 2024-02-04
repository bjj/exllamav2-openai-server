import argparse, re
from pydantic import BaseModel, Field
import typing

# transforms -> middle-out from openrouter.ai

class ModelSettings(BaseModel):
    system_prompt: str | None
    template: str | None
    stop: list[str] | None
    lora: str | None
    max_seq_len: int | None
    max_input_len: int | None
    max_batch_size: int | None
    rope_alpha: float | None
    rope_scale: float | None
    cache_8bit: bool | None
    temperature: float | None
    top_k: int | None
    top_p: float | None
    presence_penalty: float | None
    frequency_penalty: float | None
    repetition_penalty: float | None
    min_p: float | None
    top_a: float | None
    logit_bias: dict[str, float] | None

    def apply_to_exllamav2_settings(self, settings):
        """
        Apply to ExLlamaV2Sampler.Settings
        """
        settings.temperature = self.temperature
        settings.top_k = self.top_k
        settings.top_p = self.top_p
        settings.min_p = self.min_p
        settings.top_a = self.top_a
        settings.token_presence_penalty = self.presence_penalty
        settings.token_frequency_penalty = self.frequency_penalty
        settings.token_repetition_penalty = self.repetition_penalty
        settings.token_bias = self.logit_bias
        return settings

    def apply_to_config(self, config):
        """
        Apply to ExLlamaV2Config
        """
        config.max_batch_size = self.max_batch_size

        if self.max_seq_len:
            config.max_seq_len = self.max_seq_len

        if self.max_input_len:
            config.max_input_len = self.max_input_len

        if self.rope_scale is not None:
            config.scale_pos_emb = self.rope_scale

        if self.rope_alpha is not None:
            config.scale_rope_alpha = self.rope_alpha

        return config

    def inherit_from(self, *sets):
        """
        Merges multiple sets together on top of this one, first is highest priority
        """
        for s in sets:
            for name, field in ModelSettings.__fields__.items():
                if getattr(self, name, None) is None:
                    setattr(self, name, getattr(s, name, None))

    def dict(self):
        result = super().dict()
        return {k: v for k, v in result.items() if v is not None}

    @staticmethod
    def add_arguments(parser):
        """
        Adds command line arguments to a given parser based on the fields in the ModelSettings class.
        """
        for name, field in ModelSettings.__fields__.items():
            arg_name = (field.alias or name).replace('_', '-')
            add = {
                # "required": field.required
            }
            if field.type_ == bool:
                # store_true will default False rather than None and override
                add["action"] = 'store_const'
                add["const"] = True
            elif field.key_field is not None: # logit_bias
                add["action"] = _StoreDictStrFloat
            else:
                add["type"] = field.type_
                if field.is_complex():
                    add["action"] = 'append'
            parser.add_argument(f"--{arg_name}", **add)

    @staticmethod
    def from_args(args):
        """
        Constructs a ModelSettings from the result of argparse.parse_args()
        """
        kv = {k: v for k, v in vars(args).items() if v is not None and k in ModelSettings.__fields__}
        return ModelSettings(**kv)

    @staticmethod
    def defaults():
        """
        Defaults suitable to merge with. Anything not defaulted here either gets
        defaults from exllamav2 or can't be defaulted without more context.
        """
        return ModelSettings(
            max_batch_size=4,
            system_prompt="",
            template="{{ .System }}{{ .Prompt }}", # response is implicit. this is "raw"
            stop=[],
            cache_8bit=False,
            temperature=0.8,
            top_k=0, # 0 means "none" in exllamav2
            top_p=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty= 1.0,
            min_p=0.0,
            top_a=0.0,
        )

class _StoreDictStrFloat(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = getattr(namespace, self.dest) or {}
        for kv in values.split(","):
            try:
                k, v = re.split('[:=]', kv)
            except ValueError:
                parser.error('Expected k:v or k=v')
            try:
                v = float(v)
            except ValueError:
                parser.error('Expected key=<float>')
            my_dict[k] = v

        setattr(namespace, self.dest, my_dict)


def _main():
    parser = argparse.ArgumentParser()
    ModelSettings.add_arguments(parser)
    args = parser.parse_args()
    settings = ModelSettings.from_args(args)
    print(repr(settings))
    print(repr(settings.dict()))

if __name__ == "__main__":
    _main()
