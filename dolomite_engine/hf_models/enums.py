from enum import Enum


class InitMethod(Enum):
    normal = "normal"
    mup = "mup"


class PositionEmbeddingType(Enum):
    """
    Enum class for position embeddings
    """

    learned_absolute = "learned_absolute"
    alibi = "alibi"
    rope = "rope"
    nope = "nope"


class AttentionHeadType(Enum):
    """
    Enum class for attention head type
    """

    mha = "mha"
    mqa = "mqa"
    gqa = "gqa"
