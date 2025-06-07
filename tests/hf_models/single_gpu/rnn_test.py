# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Callable

import torch
from parameterized import parameterized
from transformers import set_seed

from dolomite_engine.hf_models.modeling_utils.sequence_mixer_blocks.rnn import RNN, rnn_cute, rnn_torch
from dolomite_engine.utils import divide_if_divisible

from ..test_common import TestCommons


_SEED = 42


class RNNTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            [torch.device("cuda")],
            [torch.float32, torch.float16],
            [4],  # batch_size
            [1024],  # sequence_length
            [256],  # state_size
            [4, 256],  # num_heads
            [False, True],  # has_input_state
            [rnn_cute, torch.compile(rnn_cute, fullgraph=True)],  # function
        )
    )
    def test_rnn(
        self,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        sequence_length: int,
        state_size: int,
        num_heads: int,
        has_input_state: bool,
        function: Callable,
    ) -> None:
        set_seed(_SEED)

        x_kernel, x_expected, weight_kernel, weight_expected, input_state_kernel, input_state_expected = (
            self._get_packed_tensor_inputs(
                batch_size=batch_size,
                sequence_length=sequence_length,
                total_tokens=None,
                num_heads=num_heads,
                state_size=state_size,
                has_input_state=has_input_state,
                dtype=dtype,
                device=device,
            )
        )

        y_kernel = function(x_kernel, weight_kernel, input_state_kernel)
        y_expected = rnn_torch(x_expected, weight_expected, input_state_expected)

        y_kernel.sum().backward()
        y_expected.sum().backward()

        self.assert_equal_tensors(
            y_kernel, y_expected, False, atol_float32=4e-6, rtol_float32=0, atol_float16=6.5e-5, rtol_float16=0
        )
        self.assert_equal_tensors(
            x_kernel.grad, x_expected.grad, False, atol_float32=6e-3, rtol_float32=0, atol_float16=2e-3, rtol_float16=0
        )
        self.assert_equal_tensors(
            weight_kernel.grad,
            weight_expected.grad,
            False,
            atol_float32=6e-3,
            rtol_float32=0,
            atol_float16=2.3e-2,
            rtol_float16=0,
        )

    @parameterized.expand(
        TestCommons.make_args_matrix(
            [torch.device("cuda")],
            TestCommons.get_dtypes(),
            [[0, 7, 19, 27, 93]],  # cu_seqlens
            [256],  # state_size
            [4],  # num_heads
            [False, True],  # has_input_state
        )
    )
    def test_rnn_varlen_torch(
        self,
        device: torch.device,
        dtype: torch.dtype,
        cu_seqlens: list[int],
        state_size: int,
        num_heads: int,
        has_input_state: bool,
    ) -> None:
        set_seed(_SEED)

        batch_size = len(cu_seqlens) - 1
        cu_seqlens = torch.tensor(cu_seqlens, device=device)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()

        (
            x_packed_kernel,
            x_packed_expected,
            weight_kernel,
            weight_expected,
            input_state_kernel,
            input_state_expected,
        ) = self._get_packed_tensor_inputs(
            batch_size=batch_size,
            sequence_length=None,
            total_tokens=cu_seqlens[-1],
            num_heads=num_heads,
            state_size=state_size,
            has_input_state=has_input_state,
            dtype=dtype,
            device=device,
        )

        y_kernel = rnn_torch(
            x_packed_kernel, weight_kernel, input_state_kernel, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        )

        y_expected = []
        for i in range(batch_size):
            y = rnn_torch(
                x_packed_expected[cu_seqlens[i] : cu_seqlens[i + 1]].unsqueeze(0),
                weight_expected,
                input_state_expected[i].unsqueeze(0) if has_input_state else None,
            ).squeeze(0)
            y_expected.append(y)
        y_expected = torch.cat(y_expected)

        y_kernel.sum().backward()
        y_expected.sum().backward()

        self.assert_equal_tensors(y_kernel, y_expected, False)
        self.assert_equal_tensors(x_packed_kernel.grad, x_packed_expected.grad, False)
        self.assert_equal_tensors(
            weight_kernel.grad,
            weight_expected.grad,
            False,
            atol_float32=1.5e-7,
            rtol_float32=0,
            atol_float16=1.5e-3,
            rtol_float16=0,
            atol_bfloat16=6e-3,
            rtol_bfloat16=0,
        )

    @parameterized.expand(
        TestCommons.make_args_matrix(
            [torch.device("cuda")],
            TestCommons.get_dtypes(),
            [[0, 7, 19, 27, 93]],  # cu_seqlens
            [256],  # state_size
            [4, 256],  # num_heads
            [False, True],  # has_input_state
        )
    )
    def test_rnn_varlen_cute(
        self,
        device: torch.device,
        dtype: torch.dtype,
        cu_seqlens: list[int],
        state_size: int,
        num_heads: int,
        has_input_state: bool,
    ) -> None:
        set_seed(_SEED)

        batch_size = len(cu_seqlens) - 1
        cu_seqlens = torch.tensor(cu_seqlens, device=device)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()

        x_kernel, x_expected, weight_kernel, weight_expected, input_state_kernel, input_state_expected = (
            self._get_packed_tensor_inputs(
                batch_size=batch_size,
                sequence_length=None,
                total_tokens=cu_seqlens[-1],
                num_heads=num_heads,
                state_size=state_size,
                has_input_state=has_input_state,
                dtype=dtype,
                device=device,
            )
        )

        y_kernel = rnn_cute(x_kernel, weight_kernel, input_state_kernel, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        y_expected = rnn_torch(
            x_expected, weight_expected, input_state_expected, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        )

        y_kernel.sum().backward()
        y_expected.sum().backward()

        self.assert_equal_tensors(
            y_kernel,
            y_expected,
            False,
            atol_float32=3e-6,
            rtol_float32=0,
            atol_float16=2e-3,
            rtol_float16=0,
            atol_bfloat16=1.3e-4,
            rtol_bfloat16=0,
        )
        self.assert_equal_tensors(
            x_kernel.grad, x_expected.grad, False, atol_float32=2e-3, rtol_float32=0, atol_float16=2e-3, rtol_float16=0
        )
        self.assert_equal_tensors(
            weight_kernel.grad,
            weight_expected.grad,
            False,
            atol_float32=4e-3,
            rtol_float32=0,
            atol_float16=8e-4,
            rtol_float16=0,
            atol_bfloat16=6e-3,
            rtol_bfloat16=0,
        )

    @parameterized.expand(
        TestCommons.make_args_matrix(
            [torch.device("cuda")],
            TestCommons.get_dtypes(),
            [[0, 7, 19, 27, 93], None],  # cu_seqlens
            [256],  # state_size
            [4],  # num_heads
        )
    )
    def test_rnn_module(
        self,
        device: torch.device,
        dtype: torch.dtype,
        cu_seqlens: list[int] | None,
        state_size: int,
        num_heads: int,
    ) -> None:
        input_size = 79
        output_size = 93

        rnn = RNN(
            input_size=input_size,
            state_size=state_size,
            output_size=output_size,
            num_heads=num_heads,
            add_bias=False,
            gradient_clipping=None,
        ).to(device, dtype)

        batch_size = 4 if cu_seqlens is None else len(cu_seqlens) - 1
        cu_seqlens = None if cu_seqlens is None else torch.tensor(cu_seqlens, device=device)
        max_seqlen = None if cu_seqlens is None else (cu_seqlens[1:] - cu_seqlens[:-1]).max()

        input = (
            torch.randn(batch_size, 1024, input_size, device=device, dtype=dtype)
            if cu_seqlens is None
            else torch.randn(cu_seqlens[-1], input_size, device=device, dtype=dtype)
        )
        input_state = torch.randn(batch_size, state_size, device=device, dtype=dtype)

        output, output_state = rnn(
            input=input, input_state=input_state, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, use_kernel=True
        )

        output.sum().backward()

        assert output.size() == (*input.size()[:-1], output_size)
        assert output_state.size() == input_state.size()

    def _get_packed_tensor_inputs(
        self,
        batch_size: int,
        sequence_length: int | None,
        total_tokens: int | None,
        num_heads: int,
        state_size: int,
        has_input_state: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor | None]:
        head_dim = divide_if_divisible(state_size, num_heads)

        x_kernel, x_expected = self.get_random_duplicated_tensors(
            (
                (batch_size, sequence_length, num_heads, head_dim)
                if total_tokens is None
                else (total_tokens, num_heads, head_dim)
            ),
            device=device,
            dtype=dtype,
            std=0.01,
        )

        weight_kernel, weight_expected = self.get_random_duplicated_tensors(
            (num_heads, head_dim, head_dim), device=device, dtype=dtype, std=0.01
        )

        input_state_kernel = None
        input_state_expected = None
        if has_input_state:
            input_state_kernel, input_state_expected = self.get_random_duplicated_tensors(
                (batch_size, num_heads, head_dim), device=device, dtype=dtype, std=0.01
            )

        return x_kernel, x_expected, weight_kernel, weight_expected, input_state_kernel, input_state_expected
