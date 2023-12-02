from engine.data.config import DatasetConfig


class AlpacaConfig(DatasetConfig):
    seed: int = 42
    val_samples: int = 2600
    test_samples: int = 2600


class DollyConfig(AlpacaConfig):
    val_samples: int = 750
    test_samples: int = 750


class VicunaConfig(AlpacaConfig):
    val_samples: int = 1000
    test_samples: int = 1000
    filter_sorry: bool = False
