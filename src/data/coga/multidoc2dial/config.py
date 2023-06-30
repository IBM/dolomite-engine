from enum import Enum
from typing import List

from src.constants import DatasetSplit
from src.data.config import DatasetConfig


class YatinDineshDatasetType(Enum):
    # don't use evidence for training
    NO_EVIDENCE = "no_evidence"
    # generate response first and then evidence
    RESPONSE_EVIDENCE = "response_evidence"
    # generate evidence first and then response
    EVIDENCE_RESPONSE = "evidence_response"
    # generate evidence and feed it back to the encoder to generate the response
    TOKEN_GUIDED_EVIDENCE_RESPONSE = "token_guided_evidence_response"
    # generate response only with response tag
    RESPONSE_ONLY = "response_only"
    # generate evidence only with evidence tag
    EVIDENCE_ONLY = "evidence_only"


class YatinAnswerabilityConfig(DatasetConfig):
    # max tokens to use for document
    max_document_length: int = 700
    # max tokens to use for evidence
    max_evidence_length: int = 100
    # files for the dataset
    files: dict = {
        DatasetSplit.train.value: "md2d_subdocs_train_pos_neg.json",
        DatasetSplit.val.value: "md2d_subdocs_val_pos_neg.json",
        DatasetSplit.test.value: "md2d_document_test_pos_neg.json",
    }
    # dataset evidence type
    dataset_type: YatinDineshDatasetType = YatinDineshDatasetType.NO_EVIDENCE
    # whether to turn on some filtering for the dataset
    filter_allowed: bool = True
    # always use this evidence, evidence field is looked up if this is None
    static_evidence: str = None
    # use this evidence for positive class
    static_positive_evidence: str = None
    # if this is True, chit-chat and unanswerable is combined into NA
    combine_no_evidence: bool = False
    # filter by ST or MT
    allowed_data_type: str = None
    # filter by allowed data types
    allowed_da_types: List[str] = None
    # filter functions to use
    filter_functions: List[str] = []
    # select only examples with response length greater than this
    length_threshold: int = None
    # whether to use the control prompt
    control_prompt: str = None
    # threshold for bertscore match between document and response
    bertscore_threshold: float = None
    # If a formatted response is desired, specify the format here
    response_format: str = None

    def _post_init(self) -> None:
        return


class DineshChitChatConfig(YatinAnswerabilityConfig):
    # files for the dataset
    files: dict = {
        DatasetSplit.train.value: "chitchat_subdocs_train.json",
        DatasetSplit.val.value: "chitchat_subdocs_val.json",
        DatasetSplit.test.value: "chitchat_document_test.json",
    }
    # whether to turn on some filtering for the dataset
    filter_allowed: bool = False
    # always use this evidence, evidence field is looked up if this is None
    static_evidence: str = "chit-chat"

    def _post_init(self) -> None:
        if self.combine_no_evidence:
            self.static_evidence = "NA"
