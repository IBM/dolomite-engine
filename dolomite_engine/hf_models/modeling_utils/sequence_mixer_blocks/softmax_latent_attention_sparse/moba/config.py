from dataclasses import dataclass


@dataclass
class MoBAConfig:
    moba_chunk_size: int
    moba_topk: int
