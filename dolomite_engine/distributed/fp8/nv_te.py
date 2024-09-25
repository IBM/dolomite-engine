import torch
import torch.nn as nn

from ...utils import is_transformer_engine_available


if is_transformer_engine_available():
    import transformer_engine.pytorch as te

    class TranformerEngineFP8Linear(te.Linear): ...

    class TranformerEngineFP8LayerNorm(te.LayerNorm): ...


@torch.no_grad()
def convert_model_to_transformer_engine(
    model: nn.Module, _convert_linear: bool = True, _convert_ln: bool = True
) -> None:
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and _convert_linear:
            if any(p % 16 != 0 for p in module.weight.shape):
                continue

            has_bias = module.bias is not None
            te_module = TranformerEngineFP8Linear(
                module.in_features, module.out_features, bias=has_bias, params_dtype=module.weight.dtype
            )
            te_module.weight.copy_(module.weight)
            if has_bias:
                te_module.bias.copy_(module.bias)

            setattr(model, name, te_module)
        elif isinstance(module, nn.LayerNorm) and _convert_ln:
            te_module = TranformerEngineFP8LayerNorm(
                module.normalized_shape[0], eps=module.eps, params_dtype=module.weight.dtype
            )
            te_module.weight.copy_(module.weight)
            te_module.bias.copy_(module.bias)

            setattr(model, name, te_module)
        else:
            convert_model_to_transformer_engine(module, _convert_linear=_convert_linear, _convert_ln=_convert_ln)


@torch.no_grad()
def convert_model_from_transformer_engine(
    model: nn.Module, _convert_linear: bool = True, _convert_ln: bool = True
) -> None:
    for name, module in model.named_children():
        if isinstance(module, TranformerEngineFP8Linear) and _convert_linear:
            has_bias = module.bias is not None
            new_module = nn.Linear(
                module.in_features, module.out_features, bias=has_bias, params_dtype=module.weight.dtype
            )
            new_module.weight.copy_(module.weight)
            if has_bias:
                new_module.bias.copy_(module.bias)

            setattr(model, name, new_module)
        elif isinstance(module, TranformerEngineFP8LayerNorm) and _convert_ln:
            new_module = nn.LayerNorm(module.normalized_shape[0], eps=module.eps, params_dtype=module.weight.dtype)
            new_module.weight.copy_(module.weight)
            new_module.bias.copy_(module.bias)

            setattr(model, name, new_module)
        else:
            convert_model_from_transformer_engine(module, _convert_linear=_convert_linear, _convert_ln=_convert_ln)


def has_transformer_engine_layers(model: nn.Module) -> bool:
    for m in model.modules():
        if isinstance(m, (te.LayerNorm, te.Linear, te.TransformerLayer)):
            return True
    return False
