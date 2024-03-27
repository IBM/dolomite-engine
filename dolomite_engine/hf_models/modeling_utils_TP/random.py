import contextlib

import torch


class CUDA_RNGStatesTracker:
    def __init__(self) -> None:
        self.states = {}
        self.seeds = set()

    def add(self, name: str = "tensor-parallel-seed", seed: int = 42) -> None:
        self.seeds.add(seed)
        if name in self.states:
            raise Exception("cuda rng state {} already exists".format(name))

        orig_rng_state = torch.cuda.get_rng_state()

        torch.cuda.manual_seed(seed)
        self.states[name] = torch.cuda.get_rng_state()

        torch.cuda.set_rng_state(orig_rng_state)

    @contextlib.contextmanager
    def fork(self, name: str = "tensor-parallel-seed"):
        if name not in self.states:
            raise Exception("cuda rng state {} is not added".format(name))

        orig_rng_state = torch.cuda.get_rng_state()

        torch.cuda.set_rng_state(self.states[name])
        try:
            yield
        finally:
            self.states[name] = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(orig_rng_state)
