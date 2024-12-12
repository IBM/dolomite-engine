# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [Note] Getting the 'torchao' package:
# This script requires the 'torchao' package to function correctly.
# Please ensure you have this package installed from the appropriate repository.
# You can obtain it from https://github.com/pytorch/ao by following the
# installation instructions.

# Note: Performance
# Float8 experimental is intended to be ran under `torch.compile`` for competitive performance

import torch
from torchao.float8 import (
    CastConfig,
    Float8LinearConfig,
    ScalingType,
    convert_to_float8_training,
    precompute_float8_dynamic_scale_for_fsdp,
    sync_float8_amax_and_scale_history,
)

from ..containers import ModelContainer


class Float8Handler:
    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims) -> None:
        float8_config = job_config.float8

        enable_fsdp_float8_all_gather = parallel_dims.dp_shard_enabled and float8_config.enable_fsdp_float8_all_gather
        scaling_type_input = ScalingType(float8_config.scaling_type_input)
        scaling_type_weight = ScalingType(float8_config.scaling_type_weight)
        scaling_type_grad_output = ScalingType(float8_config.scaling_type_grad_output)
        self.config = Float8LinearConfig(
            enable_fsdp_float8_all_gather=enable_fsdp_float8_all_gather,
            cast_config_input=CastConfig(scaling_type=scaling_type_input),
            cast_config_weight=CastConfig(scaling_type=scaling_type_weight),
            cast_config_grad_output=CastConfig(scaling_type=scaling_type_grad_output),
        )

        self.precompute_scale = (
            enable_fsdp_float8_all_gather and float8_config.precompute_float8_dynamic_scale_for_fsdp
        )

        self.delayed_scaling = (
            scaling_type_input is ScalingType.DELAYED
            or scaling_type_weight is ScalingType.DELAYED
            or scaling_type_grad_output is ScalingType.DELAYED
        )
        self._sync_float8_amax_and_scale_history = None
        self.compile = job_config.training.compile

    def convert_to_float8_training(self, model_container: ModelContainer) -> None:
        for model in model_container:
            convert_to_float8_training(
                model,
                config=self.config,
                module_filter_fn=lambda mod, fqn: fqn != "output",
            )

    def precompute_float8_dynamic_scale_for_fsdp(self, model_container: ModelContainer) -> None:
        if not self.precompute_scale:
            return

        for model in model_container:
            precompute_float8_dynamic_scale_for_fsdp(model)

    def sync_float8_amax_and_scale_history(self, model_container: ModelContainer) -> None:
        if not self.delayed_scaling:
            return

        if self._sync_float8_amax_and_scale_history is None:
            if self.compile:
                self._sync_float8_amax_and_scale_history = torch.compile(sync_float8_amax_and_scale_history)
            else:
                self._sync_float8_amax_and_scale_history = sync_float8_amax_and_scale_history

        for model in model_container:
            self._sync_float8_amax_and_scale_history(model)
