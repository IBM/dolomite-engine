import os
from argparse import Namespace
from typing import List

import torch
from deepspeed import DeepSpeedEngine
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from peft import PromptTuningConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoTokenizer
from transformers.deepspeed import HfDeepSpeedConfig

from src.constants import Mode, TrainingInferenceType
from src.utils import get_deepspeed_config, get_local_rank, register_profiler, register_timer, run_rank_n, warn_rank_0


def pad(arrays: list, padding: int, max_length: int = None, side: str = "left"):
    if max_length is None:
        max_length = max(list(map(len, arrays)))

    if side == "left":
        inputs = [[padding] * (max_length - len(array)) + array for array in arrays]
        masks = [[0] * (max_length - len(array)) + [1] * len(array) for array in arrays]
    else:
        inputs = [array + [padding] * (max_length - len(array)) for array in arrays]
        masks = [[1] * len(array) + [0] * (max_length - len(array)) for array in arrays]

    return inputs, masks


class Model(torch.nn.Module):
    @register_profiler("initialize_model")
    @register_timer("initialize_model")
    def __init__(self, args: Namespace, mode: Mode):
        super().__init__()

        self.mode = mode
        self.model_name = args.model_name
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.is_encoder_decoder = self.config.is_encoder_decoder
        self.training_inference_type = args.training_inference_type
        self.dtype = args.dtype

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.training_inference_type == TrainingInferenceType.full_finetuning:
            if mode == Mode.training:
                deepspeed_config = get_deepspeed_config(args)
                # this tells from_pretrained to instantiate directly on gpus
                # this only instantiates a single instance of the model across the ranks
                dschf = HfDeepSpeedConfig(deepspeed_config)

                self.model = args.model_class.from_pretrained(self.model_name)
            else:
                self.model = args.model_class.from_pretrained(self.model_name, torch_dtype=self.dtype)
        elif args.training_inference_type == TrainingInferenceType.prompt_tuning:
            self.peft_config = PromptTuningConfig(
                task_type=TaskType.SEQ_2_SEQ_LM if self.is_encoder_decoder else TaskType.CAUSAL_LM,
                prompt_tuning_init=args.prompt_tuning_init,
                num_virtual_tokens=args.num_virtual_tokens,
                prompt_tuning_init_text=args.prompt_tuning_init_text,
                tokenizer_name_or_path=args.model_name,
            )

            self.model = args.model_class.from_pretrained(self.model_name, torch_dtype=self.dtype)
            self.model = get_peft_model(self.model, self.peft_config)

        if mode == Mode.training:
            # if using deepspeed
            self.input_device = get_local_rank()
        else:
            self.input_device = 0
            if not torch.cuda.is_available():
                warn_rank_0("no CUDA device found, running on CPU")
                self.input_device = "cpu"

            self.to(self.input_device)

        self.prepare_input_output_for_forward = lambda *args, **kwargs: None
        self.prepare_input_output_for_generate = lambda *args, **kwargs: None

    @register_profiler("forward_pass")
    @register_timer("forward_pass")
    def forward(self, batch: dict) -> torch.Tensor:
        inputs, outputs = self.prepare_input_output_for_forward(batch)
        batch = self.get_input_ids(inputs, outputs)

        for i in batch:
            batch[i] = batch[i].to(self.input_device)

        model_outputs = self.model(**batch)

        return model_outputs.loss

    @register_profiler("generate")
    @register_timer("generate")
    def generate(self, batch: dict, generate_kwargs: dict) -> List[str]:
        inputs = self.prepare_input_output_for_generate(batch)
        batch = self.get_input_ids(inputs)

        for i in batch:
            batch[i] = batch[i].to(self.input_device)

        generated = self.model.generate(**batch, **generate_kwargs)

        if not self.is_encoder_decoder:
            generated = generated[:, batch["input_ids"].shape[1] :]

        generated_text = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        return generated_text

    @register_profiler("load_ds_checkpoint")
    @register_timer("load_ds_checkpoint")
    def load_ds_checkpoint(self, path: str) -> None:
        checkpoint_dir = os.path.dirname(path)
        tag = os.path.basename(path)
        state = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, tag)

        if self.training_inference_type == TrainingInferenceType.prompt_tuning:
            self.load_state_dict(state, strict=False)
        elif self.training_inference_type == TrainingInferenceType.full_finetuning:
            for key in state:
                state[key] = state[key].to(self.dtype)

            # tied weights are not loaded by DeepSpeed using the above method https://github.com/microsoft/DeepSpeed/issues/1896
            if self.model_name.startswith("bigscience/bloom"):
                state["model.lm_head.weight"] = state["model.transformer.word_embeddings.weight"]
            elif self.model_name.startswith("google/flan"):
                state["model.encoder.embed_tokens.weight"] = state["model.shared.weight"]
                state["model.decoder.embed_tokens.weight"] = state["model.shared.weight"]

            self.load_state_dict(state)

    @register_profiler("get_input_ids")
    @register_timer("get_input_ids")
    def get_input_ids(self, inputs: List[int], outputs: List[int] = None) -> dict:
        result = {}

        if self.mode == Mode.training:
            assert outputs is not None, "outputs can't be None during training"

            max_length = None
            if not self.is_encoder_decoder:
                max_length = max(list(map(len, inputs)))

            input_ids, attention_mask = pad(
                inputs, padding=self.tokenizer.pad_token_id, max_length=max_length, side=self.tokenizer.padding_side
            )
            labels, _ = pad(outputs, padding=-100, max_length=max_length, side=self.tokenizer.padding_side)

            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)
            labels = torch.tensor(labels)

            result["labels"] = labels
        elif self.mode == Mode.inference:
            input_ids, attention_mask = pad(
                inputs, padding=self.tokenizer.pad_token_id, side=self.tokenizer.padding_side
            )

            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)

        result["input_ids"] = input_ids
        result["attention_mask"] = attention_mask
        return result


class ModelCheckpointer:
    @classmethod
    @register_profiler("save_checkpoint")
    @register_timer("save_checkpoint")
    def save_checkpoint(cls, model: DeepSpeedEngine, path: str) -> None:
        model.save_checkpoint(path)

        # its a bit complicated to unwrap in fp32 during training, so we
        # recover from the saved sharded deepspeed checkpoint
        if model.training_inference_type == TrainingInferenceType.prompt_tuning:
            tag = f"global_step{model.global_steps}"
            state_dict: dict = run_rank_n(get_fp32_state_dict_from_zero_checkpoint)(path, tag)

            # effectively equivalent to run_rank_n
            if state_dict is not None:
                tensor = state_dict["model.prompt_encoder.embedding.weight"]

                if model.is_encoder_decoder:
                    encoder_tensor, decoder_tensor = torch.split(tensor, model.peft_config.num_virtual_tokens)
                    torch.save(encoder_tensor, os.path.join(path, tag, "encoder.pt"))
                    torch.save(decoder_tensor, os.path.join(path, tag, "decoder.pt"))
                else:
                    torch.save(tensor, os.path.join(path, tag, "decoder.pt"))

    @classmethod
    def ds_to_hf(cls, model: Model, save_path: str) -> None:
        model.tokenizer.save_pretrained(save_path)
        model.model.save_pretrained(save_path)
