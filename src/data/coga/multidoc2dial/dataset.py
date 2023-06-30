import json
import os
from typing import List, Type, Union

from pydantic import BaseModel
from transformers import AutoTokenizer

from src.arguments import InferenceArgs, TrainingArgs
from src.constants import DatasetKeys, DatasetSplit, Mode
from src.data.coga.filters import filter_dataset
from src.data.coga.multidoc2dial.config import DineshChitChatConfig, YatinAnswerabilityConfig, YatinDineshDatasetType
from src.data.dataset import BaseDataset, check_raw_example, generate_random_id
from src.utils.logging import print_rank_0, warn_rank_0


class YatinAnswerabilityDataset(BaseDataset):
    def __init__(
        self,
        args: Union[TrainingArgs, InferenceArgs],
        split: DatasetSplit,
        mode: Mode,
        tokenizer: AutoTokenizer,
        is_encoder_decoder: bool,
        config_class: Type[BaseModel] = YatinAnswerabilityConfig,
    ) -> None:
        super().__init__(args, split, mode, tokenizer, is_encoder_decoder)

        self.data_config["dataset_type"] = YatinDineshDatasetType(self.data_config.get("dataset_type", "no_evidence"))
        self.data_config = config_class(**self.data_config)
        self.control_prompt = self.data_config.control_prompt
        print_rank_0("The data name is: ", self.data_name)
        print_rank_0("The control prompt is: ", self.control_prompt)
        if self.control_prompt is None:
            self.control_token_ids = []
        else:
            self.control_token_ids = self.tokenizer(self.control_prompt, add_special_tokens=False)["input_ids"]
        if self.do_format_input:
            raise ValueError(f"input_format for {self.__class__.__name__} should be '__input__'")

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
        seperator = self.tokenizer("\n", add_special_tokens=False)["input_ids"]
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

        input = document + seperator + context + input

        if special_token is not None:
            special_token_id = self.tokenizer.convert_tokens_to_ids(special_token)
            input = [special_token_id] + input

        if len(self.control_token_ids) != 0:
            input = input + seperator + self.control_token_ids
        return input

    def construct_output_from_format(self, output: str) -> str:
        output = super().construct_output_from_format(output)
        output = self.tokenizer(output, add_special_tokens=False)["input_ids"]
        output = output[: self.max_output_tokens]
        return output

    def prepare_examples(self) -> List[dict]:
        print_rank_0("Preparing examples for dataset {}...".format(self.data_name))
        if self.split.value not in self.data_config.files:
            return []

        data_file = os.path.join(self.data_path, self.data_config.files[self.split.value])

        with open(data_file, "r") as f:
            raw_examples = json.load(f)
            print_rank_0("Total number of examples in json file: {}".format(len(raw_examples)))
            filtered_examples = filter_dataset(raw_examples)
            print_rank_0("Total number of examples after filtering: {}".format(len(filtered_examples)))
            examples = []
            for raw_example in filtered_examples:
                if self.data_config.filter_allowed and (
                    raw_example["neg_subtype"].lower() == "original" or raw_example["last_speaker"].lower() == "agent"
                ):
                    continue

                if (
                    self.data_config.allowed_data_type is not None
                    and self.data_config.allowed_data_type not in raw_example["data_types"]
                ):
                    continue

                check_raw_example(raw_example, self.mode)

                result_example = {}

                # construct input
                context: str = raw_example["context"]
                document: str = raw_example["document"]

                context = context.strip()
                document = document.strip()

                # for decoder only models, we need an explicit marker
                if not self.is_encoder_decoder:
                    context += "\nAgent:"

                if (
                    self.data_config.dataset_type != YatinDineshDatasetType.NO_EVIDENCE
                    and self.data_config.dataset_type != YatinDineshDatasetType.RESPONSE_ONLY
                ):
                    if self.data_config.static_evidence is None:
                        if raw_example["type"] == "positive":
                            if self.data_config.static_positive_evidence is None:
                                evidence = raw_example["evidence"]
                            else:
                                evidence = self.data_config.static_positive_evidence
                        else:
                            if self.data_config.combine_no_evidence:
                                evidence = "NA"
                            else:
                                evidence = "unanswerable"
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
                    if self.data_config.response_format != None:
                        response = self.data_config.response_format
                    if self.data_config.dataset_type == YatinDineshDatasetType.NO_EVIDENCE:
                        output = response
                    elif self.data_config.dataset_type == YatinDineshDatasetType.RESPONSE_EVIDENCE:
                        output = f"response: {response}; evidence: {evidence}"
                    elif self.data_config.dataset_type == YatinDineshDatasetType.EVIDENCE_RESPONSE:
                        output = f"evidence: {evidence}; response: {response}"
                    elif self.data_config.dataset_type == YatinDineshDatasetType.TOKEN_GUIDED_EVIDENCE_RESPONSE:
                        output = evidence
                    elif self.data_config.dataset_type == YatinDineshDatasetType.RESPONSE_ONLY:
                        output = f"response: {response}"

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

        print_rank_0("Prepared {} examples".format(len(examples)))
        return examples


class DineshChitChatDataset(YatinAnswerabilityDataset):
    def __init__(
        self,
        args: Union[TrainingArgs, InferenceArgs],
        split: DatasetSplit,
        mode: Mode,
        tokenizer: AutoTokenizer,
        is_encoder_decoder: bool,
    ) -> None:
        super().__init__(args, split, mode, tokenizer, is_encoder_decoder, config_class=DineshChitChatConfig)
