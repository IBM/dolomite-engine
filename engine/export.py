from .arguments import get_args
from .checkpointing import load_checkpoint_for_inference
from .enums import Mode
from .model import Model


def main() -> None:
    """main program"""

    mode = Mode.export

    args = get_args(mode)

    model = Model(args, mode)

    load_checkpoint_for_inference(model, args.load_args.load_path, args.load_args.iteration)

    model.tokenizer.save_pretrained(args.export_path)
    model.model.save_pretrained(args.export_path)


if __name__ == "__main__":
    main()
