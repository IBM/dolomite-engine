from enum import Enum

from pydantic import BaseModel

from src.constants import DatasetSplit


class YatinDineshDatasetType(Enum):
    NO_EVIDENCE = "no_evidence"
    RESPONSE_EVIDENCE = "response_evidence"
    EVIDENCE_RESPONSE = "evidence_response"
    TOKEN_GUIDED_EVIDENCE_RESPONSE = "token_guided_evidence_response"


class DineshChitChatConfig(BaseModel):
    max_document_length: int = 700
    files: dict = {
        DatasetSplit.train.value: "chitchat_subdocs_train.json",
        DatasetSplit.val.value: "chitchat_subdocs_val.json",
        DatasetSplit.test.value: "chitchat_subdocs_test.json",
    }
    dataset_type: YatinDineshDatasetType = YatinDineshDatasetType.NO_EVIDENCE


class YatinAnswerabilityConfig(DineshChitChatConfig):
    filter: bool = True
    files: dict = {
        DatasetSplit.train.value: f"md2d_subdocs_train_pos_neg.json",
        DatasetSplit.val.value: f"md2d_subdocs_val_pos_neg.json",
        DatasetSplit.test.value: f"md2d_document_test_pos_neg.json",
    }
