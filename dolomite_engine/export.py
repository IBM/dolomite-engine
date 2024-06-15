from .arguments import get_args
from .checkpointing import load_checkpoint_for_inference
from .enums import Mode


def main() -> None:
    """main program"""

    mode = Mode.export

    args = get_args(mode)

    model, _ = load_checkpoint_for_inference(args, mode, use_meta=True)
    model.save_pretrained(args.export_path)


if __name__ == "__main__":
    main()
