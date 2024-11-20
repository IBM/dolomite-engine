from functools import partial, wraps

import torch


def contiguous(fn):
    @wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        return fn(
            ctx,
            *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
            **{k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()},
        )

    return wrapper


autocast_custom_fwd = partial(torch.amp.custom_fwd, device_type="cuda")
autocast_custom_bwd = partial(torch.amp.custom_bwd, device_type="cuda")
