import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed._tensor._collective_utils import (
    mesh_broadcast,
    mesh_scatter,
    pad_tensor,
    shard_dim_alltoall,
    unpad_tensor,
)
from torch.distributed._tensor.placement_types import Partial, Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh

from ..communication import Communication, CommunicationBackend


class DolomitePartialPlacement(Partial):
    def _reduce_value(self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int) -> torch.Tensor:
        # Partial placement contract #1:
        # _reduce_value: reduce the value of the tensor on the mesh dimension
        return Communication.all_reduce(
            tensor=tensor, op=self.reduce_op, group=(mesh, mesh_dim), backend=CommunicationBackend.torch_distributed
        )

    def _reduce_shard_value(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        shard_spec: Shard,
    ) -> torch.Tensor:
        # Partial placement contract #2:
        # _reduce_shard_value: reduce_scatter the value of the tensor over the mesh dimension
        return shard_spec._reduce_shard_tensor(tensor, mesh, self.reduce_op, mesh_dim)


class DolomiteReplicatePlacement(Replicate):
    def _replicate_tensor(self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int) -> torch.Tensor:
        """
        Replicate (broadcast) a torch.Tensor on a mesh dimension (use
        the first coordinate on the mesh dimension as source of truth)
        """
        my_coordinate = mesh.get_coordinate()
        if my_coordinate is None:
            # if rank is not part of mesh, we simply return an empty tensor
            return tensor.new_empty(0, requires_grad=tensor.requires_grad)

        tensor = tensor.contiguous()
        mesh_broadcast(tensor, mesh, mesh_dim=mesh_dim)
        return tensor


class DolomiteShardPlacement(Shard):
    def _shard_tensor(self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int) -> torch.Tensor:
        """
        shard and scatter a tensor on a mesh dimension (use coordinate
        0 on the mesh dimension as source of truth)
        """
        my_coordinate = mesh.get_coordinate()
        num_chunks = mesh.size(mesh_dim=mesh_dim)

        if my_coordinate is None:
            # if rank is not part of mesh, we simply return an empty tensor
            return tensor.new_empty(0, requires_grad=tensor.requires_grad)

        scatter_list, pad_sizes = self._split_tensor(tensor, num_chunks, with_padding=True, contiguous=True)

        mesh_dim_local_rank = my_coordinate[mesh_dim]
        output = torch.empty_like(scatter_list[mesh_dim_local_rank])
        mesh_scatter(output, scatter_list, mesh, mesh_dim=mesh_dim)

        # Only unpad if the local_tensor was padded on the dimension.
        if pad_sizes and pad_sizes[mesh_dim_local_rank] > 0:
            output = unpad_tensor(output, self.dim, pad_sizes[mesh_dim_local_rank])
        return output

    def _reduce_shard_tensor(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        reduce_op: str,
        mesh_dim: int,
    ) -> torch.Tensor:
        """
        reduce and scatter a tensor on a mesh dimension
        """
        my_coordinate = mesh.get_coordinate()
        num_chunks = mesh.size(mesh_dim=mesh_dim)

        if my_coordinate is None:
            # if rank is not part of mesh, we simply return local_tensor,
            # which should be an empty tensor
            return tensor

        is_padded = tensor.size(self.dim) % num_chunks != 0
        if is_padded:
            scattered_list, pad_sizes = self._split_tensor(tensor, num_chunks, with_padding=True, contiguous=True)
            tensor = torch.cat(scattered_list, dim=self.dim)
        elif not tensor.is_contiguous():
            tensor = tensor.contiguous()

        output = funcol.reduce_scatter_tensor(tensor, reduce_op, scatter_dim=self.dim, group=(mesh, mesh_dim))

        if is_padded:
            output = unpad_tensor(output, self.dim, pad_sizes[my_coordinate[mesh_dim]])  # type: ignore[possibly-undefined]
        return output

    def _to_replicate_tensor(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: list[int],
    ) -> torch.Tensor:
        """
        This function all_gather all shards and return a tensor that
        is replicated on the previously sharded mesh dimension
        """
        num_chunks = mesh.size(mesh_dim=mesh_dim)
        # check if it's uneven, so we need to pad input tensor before all_gather
        local_shape = list(local_tensor.size())

        logical_dim_size = current_logical_shape[self.dim]
        is_padded = logical_dim_size % num_chunks != 0

        if is_padded:
            full_chunk_size = (logical_dim_size + num_chunks - 1) // num_chunks
            pad_size = full_chunk_size - local_shape[self.dim]
            local_tensor = pad_tensor(local_tensor, self.dim, pad_size)

        if not local_tensor.is_contiguous():
            local_tensor = local_tensor.contiguous()

        result = funcol.all_gather_tensor(
            local_tensor,
            gather_dim=self.dim,
            group=(mesh, mesh_dim),
        )
        if is_padded:
            unpad_size = full_chunk_size * num_chunks - logical_dim_size  # type: ignore[possibly-undefined]
            result = unpad_tensor(result, self.dim, unpad_size)
        return result

    def _to_new_shard_dim(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: list[int],
        new_shard_dim: int,
    ) -> torch.Tensor:
        """
        transform from existing sharded tensor to a new sharded tensor on
        that shard on a new dimension, which performs an alltoall
        """
        my_coordinate = mesh.get_coordinate()
        if my_coordinate is None:
            # if rank is not part of mesh, we simply return local_tensor,
            # which should be an empty tensor
            return local_tensor

        num_chunks = mesh.size(mesh_dim=mesh_dim)

        old_dim_logical_size = current_logical_shape[self.dim]
        new_dim_logical_size = current_logical_shape[new_shard_dim]
        old_dim_padding = old_dim_logical_size % num_chunks != 0
        new_dim_padding = new_dim_logical_size % num_chunks != 0
        if old_dim_padding:
            old_dim_full_chunk_size = (old_dim_logical_size + num_chunks - 1) // num_chunks
            old_dim_pad_size = old_dim_full_chunk_size - local_tensor.size(self.dim)
            local_tensor = pad_tensor(local_tensor, self.dim, old_dim_pad_size)
        if new_dim_padding:
            new_dim_full_chunk_size = (new_dim_logical_size + num_chunks - 1) // num_chunks
            new_dim_pad_size = new_dim_full_chunk_size * num_chunks - local_tensor.size(new_shard_dim)
            local_tensor = pad_tensor(local_tensor, new_shard_dim, new_dim_pad_size)

        if not local_tensor.is_contiguous():
            local_tensor = local_tensor.contiguous()

        new_tensor = shard_dim_alltoall(local_tensor, self.dim, new_shard_dim, mesh, mesh_dim)

        if old_dim_padding:
            old_dim_unpad_size = (
                old_dim_full_chunk_size * num_chunks - current_logical_shape[self.dim]  # type: ignore[possibly-undefined]
            )
            new_tensor = unpad_tensor(new_tensor, self.dim, old_dim_unpad_size)  # type: ignore[possibly-undefined]

        if new_dim_padding:
            local_shard_size_on_new_dim = self._local_shard_size_on_dim(
                new_dim_logical_size, num_chunks, my_coordinate[mesh_dim]
            )[0]
            new_dim_unpad_size = new_dim_full_chunk_size - local_shard_size_on_new_dim  # type: ignore[possibly-undefined]
            new_tensor = unpad_tensor(new_tensor, new_shard_dim, new_dim_unpad_size)  # type: ignore[possibly-undefined]

        return new_tensor
