import json
import os
import tempfile
from argparse import ArgumentParser, Namespace

from transformers import AutoTokenizer

from dolomite_engine.data.megatron.preprocess_data import convert_file


def get_args() -> Namespace:
    parser = ArgumentParser()

    group = parser.add_argument_group("data_paths")
    group.add_argument("--input-path", type=str, help="path to input dataset")
    group.add_argument("--tmp-path", type=str, help="temporary path for dataset")
    group.add_argument("--output-path", type=str, help="path to output dataset")
    group.add_argument("--output-suffix", type=str, help="suffix to add to the file")

    group = parser.add_argument_group("dataset")
    group.add_argument("--dataset-list", type=str, help="path to list of datasets")
    group.add_argument("--data-subsets", type=str, nargs="+", help="specific subset to convert")

    group = parser.add_argument_group("tokenizer")
    group.add_argument("--tokenizer", type=str, help="tokenizer")

    group = parser.add_argument_group("tasks")
    group.add_argument("--convert", action="store_true", help="convert")
    group.add_argument("--merge", action="store_true", help="merge")
    group.add_argument("--max-file-size", type=int, help="max file size for merged files in GBs")
    group.add_argument("--ccc-job", action="store_true", help="submit multiple jobs on CCC")
    group.add_argument("--blue-vela-job", action="store_true", help="submit multiple jobs on Blue Vela")
    group.add_argument("--workers", type=int, default=1, help="number of workers for tokenization")

    group = parser.add_argument_group("num_files")
    group.add_argument("--num-files-per-job", type=int, default=200, help="num files per job")
    group.add_argument("--start-index", type=int, help="starting index")
    group.add_argument("--end-index", type=int, help="ending index")

    args = parser.parse_args()
    return args


def get_groups_by_sizes(path: str, max_size: int) -> list[list[str]]:
    fnames = filter(lambda x: x.endswith(".bin"), os.listdir(path))
    fnames = [os.path.join(path, i) for i in fnames]

    max_size *= 1024**3
    groups = []
    current_grp = []
    current_size = 0

    for index, fname in enumerate(fnames):
        current_grp.append(fname)
        current_size += os.path.getsize(fname)

        if current_size > max_size or index == len(fnames) - 1:
            groups.append(current_grp)
            current_grp = []
            current_size = 0

    return groups


def get_arrow_files(input_path: str, data_subset: str) -> list[str]:
    arrow_files = os.listdir(os.path.join(input_path, data_subset))
    arrow_files.sort()
    return arrow_files


def job(args: Namespace, is_blue_vela: bool = False) -> None:
    assert not args.merge, "CCC jobs don't support merge"
    assert args.convert, "CCC jobs are only for conversion"

    os.makedirs("err", exist_ok=True)
    os.makedirs("out", exist_ok=True)

    for data_subset in args.data_subsets:
        num_arrow_files = len(get_arrow_files(args.input_path, data_subset))

        start_index = 0
        while start_index < num_arrow_files:
            end_index = start_index + args.num_files_per_job

            if is_blue_vela:
                prefix = f"bsub -U p1345nodes -hl -M 32G -o out/{data_subset}-{start_index}-{end_index}.log -e err/{data_subset}-{start_index}-{end_index}.log"
            else:
                prefix = f"jbsub -q x86_24h -cores 1x4+0 -mem 32G -err err/{data_subset}-{start_index}-{end_index}.log -out out/{data_subset}-{start_index}-{end_index}.log"

            cmd = f"{prefix} -err err/{data_subset}-{start_index}-{end_index}.log -out out/{data_subset}-{start_index}-{end_index}.log python tools/convert_fms_data_to_megatron.py --input-path {args.input_path} --data-subset {data_subset} --tmp-path {args.tmp_path} --output-path {args.output_path} --tokenizer-path {args.tokenizer_path} --convert --start-index {start_index} --end-index {end_index}"
            os.system(cmd)

            start_index = end_index


def interactive(args: Namespace) -> None:
    args = get_args()

    os.makedirs(args.output_path, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    for data_subset in args.data_subsets:
        if args.convert:
            arrow_files = get_arrow_files(args.input_path, data_subset)
            arrow_files = arrow_files[args.start_index : args.end_index]

            os.makedirs(os.path.join(args.tmp_path, data_subset), exist_ok=True)

            for arrow_file in arrow_files:
                convert_file(
                    tokenizer=tokenizer,
                    input_file=os.path.join(args.input_path, data_subset, arrow_file),
                    output_prefix=os.path.join(args.tmp_path, data_subset, arrow_file.split(".")[0]),
                    workers=args.workers,
                    chunk_size=1000,
                )
        elif args.merge:
            output_suffix = (
                "-" + args.output_suffix if args.output_suffix is not None and len(args.output_suffix) > 0 else ""
            )

            if args.max_file_size is None:
                output_prefix = os.path.join(args.output_path, data_subset + output_suffix)

                cmd = f"python tools/data/merge_data.py --input-prefixes {os.path.join(args.tmp_path, data_subset)} --output-prefix {output_prefix}"
                os.system(cmd)
            else:
                file_groups = get_groups_by_sizes(os.path.join(args.tmp_path, data_subset), args.max_file_size)

                file_map = {}

                for grp_id, group in enumerate(file_groups):
                    file_map[grp_id] = group

                    output_prefix = os.path.join(args.output_path, data_subset + "-" + str(grp_id) + output_suffix)

                    with tempfile.TemporaryDirectory() as tmpdir:
                        for fid, fname in enumerate(group):
                            prefix = fname.split(".bin")[0]

                            for extension in [".idx", ".ndocs", ".bin"]:
                                os.system(
                                    f"ln -s {os.path.join(args.tmp_path, data_subset, prefix + extension)} {os.path.join(tmpdir, str(fid) + extension)}"
                                )

                        cmd = f"python tools/data/merge_datasets.py --input {tmpdir} --output-prefix {output_prefix}"
                        os.system(cmd)

                json.dump(file_map, open(os.path.join(args.output_path, f"files{output_suffix}.json"), "w"), indent=4)
        else:
            raise ValueError("unexpected task")


if __name__ == "__main__":
    args = get_args()

    if args.ccc_job or args.blue_vela_job:
        job(args, is_blue_vela=args.blue_vela_job)
    else:
        interactive(args)
