import torch
from torch.distributed._functional_collectives import AsyncCollectiveTensor, wait_tensor


class DolomiteAsyncCollectiveTensor(AsyncCollectiveTensor):
    handle = None

    def tolist(self):
        return self.trigger_wait().tolist()

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        assert meta is None
        elem = inner_tensors["elem"]
        return DolomiteAsyncCollectiveTensor(elem)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.trigger_wait()})"

    def trigger_wait(self):
        if not self.completed:
            out = wait_tensor(self.elem)
            self.completed = True
            return out
        else:
            return self.elem

    def wait(self) -> torch.Tensor:
        return wait_tensor(self.elem)

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

        def unwrap(e: AsyncCollectiveTensor):
            # wait_tensor is idepotent and will do stream sync only once
            if not is_view_op:
                return e.trigger_wait()
            return e.elem

        def wrap(e: torch.Tensor):
            # wait_tensor is idepotent and will do stream sync only once
            assert not isinstance(e, AsyncCollectiveTensor)
            res = AsyncCollectiveTensor(e)
            return res

        unwrapped_args = tree_map_only(AsyncCollectiveTensor, unwrap, args)
        unwrapped_kwargs = tree_map_only(AsyncCollectiveTensor, unwrap, kwargs)

        # we don't wrap the result as it doesn't need to be waited on.
        out = func(*unwrapped_args, **unwrapped_kwargs)

        # View ops dont require a sync, so we should re-wrap the outputs.
        if is_view_op:
            out = tree_map_only(torch.Tensor, wrap, out)

        return out

    def numpy(self):
        return self.wait().numpy()


def wait_tensor(tensor: DolomiteAsyncCollectiveTensor, handle) -> torch.Tensor:
    handle.wait()
    return tensor.elem
