import torch
from torch.distributed._functional_collectives import (
    AsyncCollectiveTensor,
    _are_we_tracing,
    _is_view_op,
    tree_map_only,
    wait_tensor,
)
from torch.utils._cxx_pytree import tree_map_only


class DolomiteAsyncCollectiveTensor(AsyncCollectiveTensor):
    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        assert meta is None
        elem = inner_tensors["elem"]
        return DolomiteAsyncCollectiveTensor(elem)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.trigger_wait()})"

    def _get_acs_underlying_tensor(self):
        """This method enables  _functional_collectives_impl to test if a tensor is an ACS"""
        return self.elem

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.view.default:
            # Fast handle aten.view as a lot of view related op goes to aten.view
            # eventually, this avoids pytree slowdown
            res = func(args[0].elem, args[1])
            wrapper_res = AsyncCollectiveTensor(res)
            return wrapper_res

        is_view_op = _is_view_op(func)

        def unwrap(e: DolomiteAsyncCollectiveTensor):
            # wait_tensor is idepotent and will do stream sync only once
            if not is_view_op:
                return e.trigger_wait()
            return e.elem

        def wrap(e: torch.Tensor):
            # wait_tensor is idepotent and will do stream sync only once
            assert not isinstance(e, DolomiteAsyncCollectiveTensor)
            res = DolomiteAsyncCollectiveTensor(e)
            return res

        unwrapped_args = tree_map_only(DolomiteAsyncCollectiveTensor, unwrap, args)
        unwrapped_kwargs = tree_map_only(DolomiteAsyncCollectiveTensor, unwrap, kwargs)

        # we don't wrap the result as it doesn't need to be waited on.
        out = func(*unwrapped_args, **unwrapped_kwargs)

        # View ops dont require a sync, so we should re-wrap the outputs.
        if is_view_op:
            out = tree_map_only(torch.Tensor, wrap, out)

        return out


def maybe_wrap_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if _are_we_tracing():
        return wait_tensor(tensor)
    return DolomiteAsyncCollectiveTensor(tensor)
