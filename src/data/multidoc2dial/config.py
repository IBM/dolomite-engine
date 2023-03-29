from pydantic import BaseModel

from src.constants import DatasetSplit


class DineshChitChatConfig(BaseModel):
    files: dict = {
        DatasetSplit.train.value: "chitchat_subdocs_train.json",
        DatasetSplit.val.value: "chitchat_subdocs_val.json",
        DatasetSplit.test.value: "chitchat_subdocs_test.json",
    }


class YatinAnswerabilityConfig(BaseModel):
    filter: bool = True
    files: dict = {
        DatasetSplit.train.value: f"md2d_subdocs_train_pos_neg.json",
        DatasetSplit.val.value: f"md2d_subdocs_val_pos_neg.json",
        DatasetSplit.test.value: f"md2d_document_test_pos_neg.json",
    }
