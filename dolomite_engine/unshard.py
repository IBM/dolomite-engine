from .arguments import get_args
from .checkpointing import load_checkpoint_for_inference
from .enums import Mode
from .utils import run_rank_n


def main() -> None:
    """main program"""

    mode = Mode.unsharding

    args = get_args(mode)

    model, _, state_dict = load_checkpoint_for_inference(args, mode, use_meta=True)
    run_rank_n(model.save_pretrained, barrier=True)(args.unsharded_path, state_dict=state_dict)


if __name__ == "__main__":
    main()
