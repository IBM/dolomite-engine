from enum import Enum


class PositionEmbeddingType(Enum):
    """
    Enum class for position embeddings
    """

    learned_absolute = "learned_absolute"
    rope = "rope"
    nope = "nope"
