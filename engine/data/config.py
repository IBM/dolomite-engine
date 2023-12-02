from typing import Any

from pydantic import BaseModel


class DatasetConfig(BaseModel):
    def __init__(__pydantic_self__, **data: Any) -> None:
        super().__init__(**data)
        __pydantic_self__._post_init()

    def _post_init(self) -> None:
        return
