from __future__ import annotations


class MetricsTrackingDict:
    def __init__(self, data: dict = {}) -> None:
        self.data = data

    def __add__(self, x: MetricsTrackingDict | dict | float | int) -> MetricsTrackingDict:
        if isinstance(x, (MetricsTrackingDict, dict)):
            if isinstance(x, MetricsTrackingDict):
                x = x.data

            for key, value in x.items():
                self.data[key] = self.data.get(key, 0) + value
        elif isinstance(x, (int, float)):
            for key in self.data:
                self.data[key] += x

        return self

    def __divide__(self, x: MetricsTrackingDict | dict | float | int) -> MetricsTrackingDict:
        if isinstance(x, (MetricsTrackingDict, dict)):
            if isinstance(x, MetricsTrackingDict):
                x = x.data

            for key, value in x.items():
                self.data[key] = self.data.get(key, 0) / value
        elif isinstance(x, (int, float)):
            for key in self.data:
                self.data[key] /= x

        return self

    def get_dict(self) -> dict:
        return self.data

    def __getitem__(self, key: str) -> float:
        return self.data[key]

    def __setitem__(self, key: str, value: float) -> None:
        self.data[key] = value
