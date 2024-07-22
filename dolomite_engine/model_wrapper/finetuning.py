import torch
import torch.distributed
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

from dolomite_engine.enums import AttentionImplementation, DistributedBackend, Mode

from ..communication import Communication
from ..hf_models import convert_padding_free_lists_to_tensors
from ..utils import ProcessGroupManager
from .base import ModelWrapper


class ModelWrapperForFinetuning(ModelWrapper):
    def __init__(
        self,
        mode: Mode,
        model_name: str | None,
        pretrained_config: dict | None,
        model_class: AutoModelForCausalLM | AutoModelForSeq2SeqLM,
        dtype: torch.dtype,
        efficient_initialization: bool,
        attention_implementation: AttentionImplementation,
        use_padding_free_transformer: bool,
        tensor_parallel_word_embeddings: bool,
        sequence_parallel: bool,
        distributed_backend: DistributedBackend,
        random_seed: int,
        neft_alpha: float | None = None,
        trust_remote_code: bool = False,
        tokenizer_name: str | None = None,
        additional_special_tokens: list[str] | None = None,
        upcast_logits_for_loss: bool = False,
    ) -> None:
        super().__init__(
            mode,
            model_name,
            pretrained_config,
            model_class,
            dtype,
            efficient_initialization,
            attention_implementation,
            use_padding_free_transformer,
            tensor_parallel_word_embeddings,
            sequence_parallel,
            distributed_backend,
            random_seed,
            neft_alpha,
            trust_remote_code,
            tokenizer_name,
            additional_special_tokens,
            upcast_logits_for_loss,
        )

        assert self.pp_world_size == 1, "pipeline parallel is not supported for finetuning"

    def forward(self, batch: dict) -> torch.Tensor:
        """forward function for a batch

        Args:
            batch (dict): a dict of key, value pairs for a batch

        Returns:
            torch.Tensor: loss tensor
        """

        batch = self._prepare_model_inputs(batch)

        model_outputs = self.model(**batch)
        loss = model_outputs[0] if isinstance(model_outputs, tuple) else model_outputs.loss

        return loss

    def _prepare_model_inputs(self, batch: dict) -> dict:
        device = torch.cuda.current_device()

        if self.tp_world_size > 1:
            tp_source_rank = ProcessGroupManager.get_tensor_parallel_first_rank()
            tp_group = ProcessGroupManager.get_tensor_parallel_group()

            if self.use_padding_free_transformer:
                keys = ["input_ids", "position_ids", "labels", "cu_seqlens", "max_seqlen"]

                if self.tp_rank == 0:
                    batch_size_total_elements = torch.tensor(
                        [len(batch["input_ids"]), sum([len(i) for i in batch["input_ids"]])], device=device
                    )
                else:
                    batch_size_total_elements = torch.empty(2, dtype=torch.long, device=device)

                torch.distributed.broadcast(batch_size_total_elements, src=tp_source_rank, group=tp_group)
                batch_size, total_elements = batch_size_total_elements

                if self.tp_rank == 0:
                    input_ids, position_ids, _, labels, cu_seqlens, max_seqlen = convert_padding_free_lists_to_tensors(
                        **batch
                    )

                    batch = {
                        "input_ids": input_ids,
                        "position_ids": position_ids,
                        "labels": labels,
                        "cu_seqlens": cu_seqlens,
                        "max_seqlen": max_seqlen,
                    }
                else:
                    batch = {
                        "input_ids": torch.empty(total_elements, dtype=torch.long, device=device),
                        "position_ids": torch.empty(total_elements, dtype=torch.long, device=device),
                        "labels": torch.empty(total_elements, dtype=torch.long, device=device),
                        "cu_seqlens": torch.empty(batch_size + 1, dtype=torch.int32, device=device),
                        "max_seqlen": torch.empty(1, dtype=torch.long, device=device),
                    }
            else:
                keys = ["input_ids", "attention_mask", "labels"]

                batch_shape = batch["input_ids"].shape if self.tp_rank == 0 else None
                batch_shape = Communication.broadcast_object(batch_shape, src=tp_source_rank, group=tp_group)

                if self.tp_rank == 0:
                    for key in keys:
                        batch[key] = batch[key].to(device)
                else:
                    batch = {key: torch.empty(batch_shape, dtype=torch.long, device=device) for key in keys}

            for key in keys:
                torch.distributed.broadcast(batch[key], src=tp_source_rank, group=tp_group)
        else:
            if self.use_padding_free_transformer:
                input_ids, position_ids, _, labels, cu_seqlens, max_seqlen = convert_padding_free_lists_to_tensors(
                    **batch
                )

                batch = {
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                    "labels": labels,
                    "cu_seqlens": cu_seqlens,
                    "max_seqlen": max_seqlen,
                }
            else:
                for key in batch:
                    batch[key] = batch[key].to(device)

        return batch
