import json
import os
import tempfile
from itertools import product
from typing import Any, Callable
from unittest import TestCase, skipUnless

import torch
from torch.testing import assert_close
from transformers import AutoConfig, AutoModelForCausalLM

from dolomite_engine import SafeTensorsWeightsManager
from dolomite_engine.hf_models import (
    AttentionHeadType,
    GPTDolomiteConfig,
    MoEDolomiteConfig,
    PositionEmbeddingType,
    export_to_huggingface,
    import_from_huggingface,
)
from dolomite_engine.hf_models.config import CommonConfig


_RUN_SLOW = True if os.getenv("RUN_SLOW", "False").lower() in ["1", "true"] else False


class TestCommons(TestCase):
    @staticmethod
    def get_all_devices() -> list[torch.device]:
        return [torch.device("cpu"), torch.device("cuda")]

    @staticmethod
    def get_attention_head_types() -> list[AttentionHeadType]:
        return [AttentionHeadType.mha, AttentionHeadType.mqa, AttentionHeadType.gqa]

    @staticmethod
    def get_attention_implementations() -> list[str]:
        return ["eager", "sdpa", "flash_attention_2"]

    @staticmethod
    def get_position_embedding_types() -> list[PositionEmbeddingType]:
        return [PositionEmbeddingType.learned_absolute, PositionEmbeddingType.alibi, PositionEmbeddingType.rope]

    @staticmethod
    def get_dtypes() -> list[torch.dtype]:
        return [torch.float32, torch.float16, torch.bfloat16]

    def make_args_matrix(*args_lists) -> list[Any]:
        return [p for p in product(*args_lists)]

    def skip_test_if_device_unavailable(self, device: torch.device) -> None:
        # convert to str
        if isinstance(device, torch.device):
            device = device.type

        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("skipping test because CUDA is unavailable")

    def skip_test_if_layernorm_kernel_unavailable(self, device: torch.device, dtype: torch.dtype) -> None:
        # convert to str
        if isinstance(device, torch.device):
            device = device.type

        if device == "cpu" and dtype == torch.float16:
            self.skipTest("LayerNormKernelImpl not implemented for Half")

    @staticmethod
    def get_dense_test_config(
        attention_head_type: AttentionHeadType,
        position_embedding_type: PositionEmbeddingType,
        num_layers: int = 8,
        add_bias: bool = True,
        activation_function: str = "gelu_pytorch_tanh",
        normalization_function: str = "layernorm",
        m_emb: float = None,
        m_width: float = None,
        m_residual: float = None,
        attention_multiplier: float = None,
    ) -> GPTDolomiteConfig:
        return GPTDolomiteConfig(
            vocab_size=2048,
            n_positions=1024,
            n_embd=32,
            n_layer=num_layers,
            n_head=4,
            num_key_value_heads=2 if attention_head_type == AttentionHeadType.gqa else None,
            attention_head_type=attention_head_type.value,
            position_embedding_type=position_embedding_type.value,
            add_bias=add_bias,
            activation_function=activation_function,
            normalization_function=normalization_function,
            tie_word_embeddings=False,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
            m_emb=m_emb,
            m_width=m_width,
            m_residual=m_residual,
            attention_multiplier=attention_multiplier,
        )

    @staticmethod
    def get_moe_test_config(
        attention_head_type: AttentionHeadType,
        position_embedding_type: PositionEmbeddingType,
        num_layers: int = 8,
        num_experts: int = 8,
        num_experts_per_tok: int = 8,
        add_bias: bool = True,
        activation_function: str = "gelu_pytorch_tanh",
        normalization_function: str = "layernorm",
        m_emb: float = None,
        m_width: float = None,
        m_residual: float = None,
        attention_multiplier: float = None,
    ) -> MoEDolomiteConfig:
        return MoEDolomiteConfig(
            vocab_size=2048,
            n_positions=1024,
            n_embd=32,
            n_layer=num_layers,
            n_head=4,
            num_experts_per_tok=num_experts_per_tok,
            num_key_value_heads=2 if attention_head_type == AttentionHeadType.gqa else None,
            attention_head_type=attention_head_type.value,
            position_embedding_type=position_embedding_type.value,
            add_bias=add_bias,
            activation_function=activation_function,
            normalization_function=normalization_function,
            tie_word_embeddings=False,
            num_experts=num_experts,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
            m_emb=m_emb,
            m_width=m_width,
            m_residual=m_residual,
            attention_multiplier=attention_multiplier,
        )

    def get_dummy_inputs(self, device: torch.device, return_list: bool = False) -> tuple[torch.Tensor | list[int]]:
        if return_list:
            # needed for flash attention
            input_ids = [list(range(5, 15)), list(range(10, 15))]
            attention_mask = None
            labels = [[-100] * 6 + list(range(11, 15)), [-100] * 2 + list(range(12, 15))]
        else:
            input_ids = torch.tensor([list(range(5, 15)), [0] * 5 + list(range(10, 15))], device=device)
            attention_mask = torch.tensor([[1] * 10, [0] * 5 + [1] * 5], device=device)
            labels = torch.tensor([[-100] * 6 + list(range(11, 15)), [-100] * 7 + list(range(12, 15))], device=device)

        return input_ids, attention_mask, labels

    def model_conversion_test(
        self,
        dolomite_config: CommonConfig,
        model_type: str,
        device: torch.device,
        exact_match: bool = True,
        compare_loss: bool = True,
    ) -> None:
        self.skip_test_if_device_unavailable(device)

        dolomite_model = self.from_config(dolomite_config).to(device)
        dolomite_model.eval()

        with tempfile.TemporaryDirectory() as tmp_path:
            save_path = os.path.join(tmp_path, "save")
            export_path = os.path.join(tmp_path, "export")
            import_path = os.path.join(tmp_path, "import")

            dolomite_model.save_pretrained(save_path, safe_serialization=True)

            export_to_huggingface(save_path, export_path, model_type=model_type)
            import_from_huggingface(export_path, import_path)

            assert self.compare_saved_models(save_path, import_path)

            hf_model = AutoModelForCausalLM.from_pretrained(export_path).to(device)
            hf_model.eval()

        input_ids, attention_mask, labels = self.get_dummy_inputs(device)

        hf_output = hf_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
        hf_logits = hf_output.logits
        hf_loss = hf_output.loss

        dolomite_output = dolomite_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True
        )
        dolomite_logits = dolomite_output.logits
        dolomite_loss = dolomite_output.loss

        # we don't care about what happens on masked values (they don't match btw)
        hf_logits[attention_mask == 0] = 0
        dolomite_logits[attention_mask == 0] = 0

        self.assert_equal_tensors(
            dolomite_logits,
            hf_logits,
            exact_match,
            rtol_float32=0,
            atol_float32=3e-7,
            rtol_float16=0,
            atol_float16=3e-7,
            rtol_bfloat16=0,
            atol_bfloat16=3e-7,
        )

        if compare_loss:
            self.assert_equal_tensors(
                dolomite_loss,
                hf_loss,
                exact_match,
                rtol_float32=0,
                atol_float32=1e-5,
                rtol_float16=0,
                atol_float16=1e-5,
                rtol_bfloat16=0,
                atol_bfloat16=1e-5,
            )

    @staticmethod
    def compare_saved_models(path1: str, path2: str) -> bool:
        config1 = json.load(open(os.path.join(path1, "config.json"), "r"))
        config2 = json.load(open(os.path.join(path2, "config.json"), "r"))

        for key in ["architectures", "torch_dtype"]:
            config1.pop(key, None)
            config2.pop(key, None)

        if config1 == config2:
            weights1 = SafeTensorsWeightsManager(path1)
            weights2 = SafeTensorsWeightsManager(path2)

            return weights1 == weights2

        return False

    def from_config(self, config: AutoConfig, **kwargs) -> AutoModelForCausalLM:
        model = AutoModelForCausalLM.from_config(config, **kwargs)

        attention_implementation = kwargs.pop("attn_implementation", None)
        use_padding_free_transformer = kwargs.pop("use_padding_free_transformer", False)
        moe_implementation = kwargs.pop("moe_implementation", None)

        if use_padding_free_transformer:
            assert model._use_padding_free_transformer

        if attention_implementation == "eager":
            assert "Attention" in str(model)
            assert "FlashAttention2" not in str(model)
            assert "PaddingFreeAttention" not in str(model)
        elif attention_implementation == "sdpa":
            assert "SDPA" in str(model)
        elif attention_implementation == "flash_attention_2":
            if use_padding_free_transformer:
                assert "PaddingFreeAttention" in str(model)
            else:
                assert "FlashAttention2" in str(model)

        if moe_implementation == "eager":
            assert "SparseMoE" in str(model)
        elif moe_implementation == "scattermoe":
            assert "ScatterMoE" in str(model)

        kwargs.pop("torch_dtype", None)
        assert len(kwargs) == 0

        return model

    def assert_equal_tensors(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        exact_match: bool,
        rtol_float32: float | None = None,
        atol_float32: float | None = None,
        rtol_float16: float | None = None,
        atol_float16: float | None = None,
        rtol_bfloat16: float | None = None,
        atol_bfloat16: float | None = None,
    ) -> None:
        if exact_match:
            assert x.equal(y)
        else:
            assert x.dtype == y.dtype
            dtype = x.dtype

            if dtype == torch.float32:
                assert_close(x, y, rtol=rtol_float32, atol=atol_float32)
            elif dtype == torch.float16:
                assert_close(x, y, rtol=rtol_float16, atol=atol_float16)
            elif dtype == torch.bfloat16:
                assert_close(x, y, rtol=rtol_bfloat16, atol=atol_bfloat16)
            else:
                raise ValueError(f"unexpected dtype ({dtype})")

    @staticmethod
    def slow_test(func: Callable) -> Callable:
        return skipUnless(_RUN_SLOW, "skipping slow test since RUN_SLOW=True is not set in the environment")(func)
