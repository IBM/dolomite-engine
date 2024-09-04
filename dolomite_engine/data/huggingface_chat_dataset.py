import copy
import glob
import logging
import os

from datasets import load_dataset
from transformers import AutoTokenizer

from ..enums import DatasetSplit, Mode
from ..utils import log_rank_0
from .base import BaseDataset


_DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|system|>\n' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|user|>\n' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|assistant|>\n' + message['content'] + eos_token }}"
    "{% if not loop.last %}\n{% endif %}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|assistant|>\n' }}"
    "{% endif %}"
)


_TAG_TO_TEMPLATE = {
    "default": _DEFAULT_CHAT_TEMPLATE,
}


class HuggingFaceChatDataset(BaseDataset):
    """A dataset class to load local json files which are in messages format"""

    def __init__(
        self,
        class_args: dict,
        split: DatasetSplit,
        mode: Mode,
        tokenizer: AutoTokenizer,
        is_encoder_decoder: bool,
        data_name: str,
        input_format: str,
        output_format: str,
        max_input_tokens: int,
        max_output_tokens: int,
        num_virtual_tokens: int = 0,
    ) -> None:
        super().__init__(
            class_args=class_args,
            split=split,
            mode=mode,
            tokenizer=tokenizer,
            is_encoder_decoder=is_encoder_decoder,
            data_name=data_name,
            input_format=input_format,
            output_format=output_format,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            num_virtual_tokens=num_virtual_tokens,
        )

        chat_template_tag = self.class_args.get("chat_template_tag", None)
        if chat_template_tag:
            chat_template = _TAG_TO_TEMPLATE[chat_template_tag]
            self.tokenizer.chat_template = chat_template

        # The prompt before the assistant/generation turn
        self.generation_prompt: str = self.class_args.get("generation_prompt", "<|assistant|>\n")
        self.generation_prompt_length: int = len(self.tokenizer.encode(self.generation_prompt))

        self.examples = self.prepare_examples()

    def construct_input_output_from_chat_format(self, example):

        messages = example["messages"]
        if len(messages) == 0:
            raise ValueError("messages field is empty.")

        input = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, max_length=self.max_output_tokens, truncation=True
        )
        output = copy.deepcopy(input)

        for message_idx, message in enumerate(messages):
            preceding_conversation = self.tokenizer.apply_chat_template(
                [messages[:message_idx]],
                add_generation_prompt=False,
                max_length=self.max_output_tokens,
                truncation=True,
            )[0]
            current_conversation = self.tokenizer.apply_chat_template(
                [messages[message_idx]],
                add_generation_prompt=False,
                max_length=self.max_output_tokens,
                truncation=True,
            )

            start_idx = len(preceding_conversation)

            if message["role"] != "assistant":
                # For non-assistant messages, all tokens are masked
                end_idx = start_idx + len(current_conversation)
            else:
                # For assistant messages, only the generation_prompt tokens are masked
                end_idx = start_idx + self.generation_prompt_length

            end_idx = min(end_idx, self.max_output_tokens)
            output[start_idx:end_idx] = [-100] * (end_idx - start_idx)

            # Avoid redundant masking
            if end_idx >= self.max_output_tokens:
                break

        # If the system and user sequences take up to max_length, we drop this example
        all_ignore = output.count(-100) == len(output)
        if all_ignore:
            return

        return {"input": input, "output": output}

    def prepare_examples(self) -> list[dict]:
        assert "data_path" in self.class_args, "`data_path` is not specified"

        data_files = glob.glob(os.path.join(self.class_args["data_path"], self.split.value, "*.jsonl"))

        split = "validation" if self.split == DatasetSplit.val else self.split.value

        examples = []
        num_discarded = 0

        for filename in data_files:
            # split is 'train' as we are loading a jsonl file where the default train split is created by the datasets library.
            json_dataset = load_dataset("json", data_files=filename, split="train")
            log_rank_0(logging.INFO, f"num examples in dataset = {len(json_dataset)}")

            for raw_example in json_dataset:
                example = self.construct_input_output_from_chat_format(raw_example)
                if not example:
                    num_discarded += 1
                    continue
                examples.append(example)
            log_rank_0(
                logging.INFO,
                f"num examples discarded for not containing any labels within the context length = {num_discarded}",
            )

        return examples

