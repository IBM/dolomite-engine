import json
import os
from copy import deepcopy

from deepspeed import DeepSpeedEngine
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

from engine.arguments import InferenceArgs, TrainingArgs
from engine.constants import DatasetConfigKeys, OptimizerKeys, TrainingInferenceType
from engine.model import Model
from engine.utils import register_timer, run_rank_n


class ModelCheckpointer:
    """class for loading and saving models"""

    @classmethod
    @register_timer("load_checkpoint_for_training")
    def load_checkpoint_for_training(cls, model: DeepSpeedEngine, load_path: str) -> None:
        """loads the deepspeed checkpoint saved for training

        Args:
            model (DeepSpeedEngine): loaded checkpoint is filled into this model
            load_path (str): path to load the deepspeed checkpoint from
        """

        model.load_checkpoint(load_path)

    @classmethod
    @register_timer("save_deepspeed_checkpoint")
    def save_deepspeed_checkpoint(cls, model: DeepSpeedEngine, args: TrainingArgs, save_path: str) -> None:
        """save deepspeed checkpoint during training

        Args:
            model (DeepSpeedEngine): model to save
            args (InferenceArgs): arguments for training
            save_path (str): save location on disk
        """

        model.save_checkpoint(save_path)
        cls.save_training_args(
            args, os.path.join(save_path, f"global_step{model.global_steps}", "training_config.json")
        )

    @classmethod
    @register_timer("load_checkpoint_for_inference")
    def load_checkpoint_for_inference(cls, model: Model, load_path: str) -> None:
        """load deepspeed checkpoint for inference

        Args:
            model (Model): model to save
            load_path (str): path to load the deepspeed checkpoint from
        """

        checkpoint_dir = os.path.dirname(load_path)
        tag = os.path.basename(load_path)
        state = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, tag)

        if model.training_inference_type == TrainingInferenceType.prompt_tuning:
            model.load_state_dict(state, strict=False)
        elif model.training_inference_type == TrainingInferenceType.full_finetuning:
            for key in state:
                state[key] = state[key].to(model.dtype)

            model.load_state_dict(state)

    @classmethod
    @register_timer("convert_deepspeed_to_huggingface_checkpoint")
    def convert_deepspeed_to_huggingface_checkpoint(cls, model: Model, load_path: str, save_path: str) -> None:
        """load the model as a deepspeed checkpoint, converts to huggingface and saves it

        Args:
            model (Model): model to save
            load_path (str): path to load the deepspeed checkpoint from
            save_path (str): save location on disk for huggingface checkpoint
        """

        cls.load_checkpoint_for_inference(model, load_path)

        model.tokenizer.save_pretrained(save_path)
        model.model.save_pretrained(save_path)

        args = json.load(open(os.path.join(load_path, "training_config.json"), "r"))
        json.dump(args, open(os.path.join(save_path, "training_config.json"), "w"), indent=4)

    @classmethod
    @run_rank_n
    def save_training_args(cls, args: TrainingArgs, save_path: str) -> None:
        """saves training args as a json

        Args:
            args (TrainingArgs): arguments for training
            save_path (str): save location on disk
        """

        args = deepcopy(args)

        # model_class
        args.model_class = args.model_class.__name__
        # dtype
        args.dtype = str(args.dtype).split(".")[1]

        # training_inference_type
        args.training_inference_type = args.training_inference_type.value
        # prompt_tuning_init
        if args.prompt_tuning_init is not None:
            args.training_inference_type = args.prompt_tuning_init.value

        # datasets
        for data_config in args.datasets:
            data_config[DatasetConfigKeys.data_class.value] = data_config[DatasetConfigKeys.data_class.value].__name__

        # optimizer
        args.optimizer[OptimizerKeys.optimizer_class.value] = args.optimizer[
            OptimizerKeys.optimizer_class.value
        ].__name__

        json.dump(vars(args), open(save_path, "w"), indent=4)

    @classmethod
    @run_rank_n
    def save_inference_args(cls, args: InferenceArgs, save_path: str) -> None:
        """saves inference args as a json

        Args:
            args (InferenceArgs): arguments for inference
            save_path (str): save location on disk
        """

        args = deepcopy(args)

        # model_class
        args.model_class = args.model_class.__name__
        # dtype
        args.dtype = str(args.dtype).split(".")[1]

        # training_inference_type
        args.training_inference_type = args.training_inference_type.value
        # prompt_tuning_init
        if args.prompt_tuning_init is not None:
            args.training_inference_type = args.prompt_tuning_init.value

        # datasets
        for data_config in args.datasets:
            data_config[DatasetConfigKeys.data_class.value] = data_config[DatasetConfigKeys.data_class.value].__name__

        json.dump(vars(args), open(save_path, "w"), indent=4)
