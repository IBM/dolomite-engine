from torch.utils.data import DataLoader as _DataLoader


class DataLoader(_DataLoader):
    def state_dict(self) -> dict:
        return {"dataset": self.dataset.state_dict(), "sampler": self.sampler.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        self.dataset.load_state_dict(state_dict.get("dataset"))
        self.sampler.load_state_dict(state_dict.get("sampler"))
