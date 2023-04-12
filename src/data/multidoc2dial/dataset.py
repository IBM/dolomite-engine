import json
import os
from argparse import Namespace
from typing import List, Type

from pydantic import BaseModel
from transformers import AutoTokenizer

from src.constants import DatasetKeys, DatasetSplit, Mode
from src.data.dataset import BaseDataset, check_raw_example, generate_random_id
from src.data.multidoc2dial.config import DineshChitChatConfig, YatinAnswerabilityConfig, YatinDineshDatasetType
from src.utils.logging import warn_rank_0


class YatinAnswerabilityDataset(BaseDataset):
    def __init__(
        self,
        args: Namespace,
        split: DatasetSplit,
        mode: Mode,
        tokenizer: AutoTokenizer,
        is_encoder_decoder: bool,
        config_class: Type[BaseModel] = YatinAnswerabilityConfig,
    ) -> None:
        super().__init__(args, split, mode, tokenizer, is_encoder_decoder)

        self.pre_tokenized = True
        self.data_config["dataset_type"] = YatinDineshDatasetType(self.data_config.get("dataset_type", "no_evidence"))
        self.data_config = config_class(**self.data_config)

        if self.do_format_input:
            raise NotImplementedError(f"{self.__class__.__name__} doesn't support input_format argument")

        if self.do_format_output:
            raise NotImplementedError(f"{self.__class__.__name__} doesn't support output_format argument")

        if self.max_input_tokens is None:
            warn_rank_0("ignoring max_document_length in the config since max_input_tokens was not specified")

        if self.data_config.dataset_type == YatinDineshDatasetType.TOKEN_GUIDED_EVIDENCE_RESPONSE:
            self.evidence_marker_token = "[evidence]"
            self.evidence_task_token = "<evidence>"
            self.response_task_token = "<response>"

            special_tokens = {
                "additional_special_tokens": [
                    self.evidence_marker_token,
                    self.evidence_task_token,
                    self.response_task_token,
                ]
            }
            self.tokenizer.add_special_tokens(special_tokens)

        self.examples = self.prepare_examples()

    def construct_input_from_format(
        self, context: str, document: str, evidence: str = None, special_token: str = None
    ) -> str:
        context = self.tokenizer(context, add_special_tokens=False)["input_ids"]
        document = self.tokenizer(document, add_special_tokens=False)["input_ids"]

        if self.max_input_tokens is not None:
            context = context[-(self.max_input_tokens - self.data_config.max_document_length) :]
            document = document[: self.data_config.max_document_length - 1]

        if (
            self.data_config.dataset_type == YatinDineshDatasetType.TOKEN_GUIDED_EVIDENCE_RESPONSE
            and evidence is not None
        ):
            evidence = self.tokenizer(evidence, add_special_tokens=False)["input_ids"]

            if self.max_input_tokens is not None:
                document = document[: -self.data_config.max_evidence_length - 1]
                evidence = evidence[: self.data_config.max_evidence_length]

            evidence_marker_token_id = self.tokenizer.convert_tokens_to_ids(self.evidence_marker_token)
            input = [evidence_marker_token_id] + evidence
        else:
            input = []

        input = document + context + input

        if special_token is not None:
            special_token_id = self.tokenizer.convert_tokens_to_ids(special_token)
            input = [special_token_id] + input

        return input

    def construct_output_from_format(self, output: str) -> str:
        output = self.tokenizer(output, add_special_tokens=False)["input_ids"]
        output = output[: self.max_output_tokens]
        return output

    def prepare_examples(self) -> List[dict]:
        examples = []
        data_file = os.path.join(self.data_path, self.data_config.files[self.split.value])

        with open(data_file, "r") as f:
            json_file = json.load(f)

            for raw_example in json_file:
                if self.data_config.filter_allowed and (
                    raw_example["neg_subtype"].lower() == "original" or raw_example["last_speaker"].lower() == "agent"
                ):
                    continue

                check_raw_example(raw_example, self.mode)

                result_example = {}

                # construct input
                context = raw_example["context"]
                document = raw_example["document"]

                if self.data_config.dataset_type != YatinDineshDatasetType.NO_EVIDENCE:
                    if self.data_config.static_evidence is None:
                        if raw_example["type"] == "positive":
                            evidence = raw_example["evidence_20_50"]
                        elif raw_example["type"] == "negative":
                            if self.data_config.combine_no_evidence:
                                evidence = "NA"
                            else:
                                evidence = "unanswerable"
                        else:
                            raise ValueError(f"unexpected 'type' {raw_example['type']} found in one of the examples")
                    else:
                        evidence = self.data_config.static_evidence

                task_token = None
                if self.data_config.dataset_type == YatinDineshDatasetType.TOKEN_GUIDED_EVIDENCE_RESPONSE:
                    task_token = self.evidence_task_token

                result_example[DatasetKeys.preprocessed_input.value] = self.construct_input_from_format(
                    context, document, special_token=task_token
                )

                # construct output
                if self.mode == Mode.training:
                    response = raw_example["response"]

                    if self.data_config.dataset_type == YatinDineshDatasetType.NO_EVIDENCE:
                        output = response
                    elif self.data_config.dataset_type == YatinDineshDatasetType.RESPONSE_EVIDENCE:
                        output = f"response: {response}; evidence: {evidence}"
                    elif self.data_config.dataset_type == YatinDineshDatasetType.EVIDENCE_RESPONSE:
                        output = f"evidence: {evidence}; response: {response}"
                    elif self.data_config.dataset_type == YatinDineshDatasetType.TOKEN_GUIDED_EVIDENCE_RESPONSE:
                        output = evidence

                    result_example[DatasetKeys.preprocessed_output.value] = self.construct_output_from_format(output)

                if DatasetKeys.id.value not in raw_example:
                    result_example[DatasetKeys.id.value] = generate_random_id(self.__class__)

                    if self.data_config.dataset_type == YatinDineshDatasetType.TOKEN_GUIDED_EVIDENCE_RESPONSE:
                        result_example[DatasetKeys.id.value] = f"{result_example[DatasetKeys.id.value]}:evidence"

                result_example.update(raw_example)
                examples.append(result_example)

                if self.data_config.dataset_type == YatinDineshDatasetType.TOKEN_GUIDED_EVIDENCE_RESPONSE:
                    result_example = {}

                    result_example[DatasetKeys.preprocessed_input.value] = self.construct_input_from_format(
                        context, document, evidence=evidence, special_token=self.response_task_token
                    )

                    if self.mode == Mode.training:
                        result_example[DatasetKeys.preprocessed_output.value] = self.construct_output_from_format(
                            response
                        )

                    if DatasetKeys.id.value not in raw_example:
                        result_example[DatasetKeys.id.value] = generate_random_id(self.__class__)

                        if self.data_config.dataset_type == YatinDineshDatasetType.TOKEN_GUIDED_EVIDENCE_RESPONSE:
                            result_example[DatasetKeys.id.value] = f"{result_example[DatasetKeys.id.value]}:response"

                    result_example.update(raw_example)
                    examples.append(result_example)

        return examples


class DineshChitChatDataset(YatinAnswerabilityDataset):
    def __init__(
        self, args: Namespace, split: DatasetSplit, mode: Mode, tokenizer: AutoTokenizer, is_encoder_decoder: bool
    ) -> None:
        super().__init__(args, split, mode, tokenizer, is_encoder_decoder, config_class=DineshChitChatConfig)
