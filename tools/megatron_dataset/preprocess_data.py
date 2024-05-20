# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import json
import multiprocessing
from argparse import ArgumentParser, Namespace
from typing import List

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from dolomite_engine.data.megatron.indexed_dataset import DType, MMapIndexedDatasetBuilder


class Encoder:
    def __init__(self, tokenizer: AutoTokenizer, json_keys: List[str], append_eod: bool) -> None:
        self.tokenizer = tokenizer
        self.json_keys = json_keys
        self.append_eod = append_eod

    def _encode_data(self, data):
        ids = {}
        for key in self.json_keys:
            text = data[key]
            document_ids = self.tokenizer.encode(text)
            if len(document_ids) > 0:
                if self.append_eod:
                    document_ids.append(self.tokenizer.eos_token_id)
                ids[key] = document_ids
        return ids

    def encode(self, json_line):
        data = json.loads(json_line)
        return self._encode_data(data)

    def encode_jsonl_zstd(self, bytes_obj):
        json_str = bytes_obj.decode("utf-8")
        return self.encode(json_str)

    def encode_hf(self, sample):
        return self._encode_data(sample)


def get_args() -> Namespace:
    parser = ArgumentParser()

    group = parser.add_argument_group(title="input data")
    group.add_argument("--input", type=str, required=True, help="Path to input JSON/Arrow")
    group.add_argument(
        "--subset", type=str, default=None, help="Subset argument when loading input data from a HuggingFace dataset"
    )
    group.add_argument(
        "--json-keys", nargs="+", default=["text"], help="space separate listed of keys to extract from json"
    )

    group = parser.add_argument_group(title="tokenizer")
    group.add_argument("--tokenizer", type=str, required=True, help="Path to the tokenizer")
    group.add_argument("--append-eod", action="store_true", help="Append an <eod> token to the end of a document.")

    group = parser.add_argument_group(title="output data")
    group.add_argument("--output-prefix", type=str, required=True, help="Path to binary output file without suffix")

    group = parser.add_argument_group(title="runtime")
    group.add_argument("--workers", type=int, required=True, help="Number of worker processes to launch")
    group.add_argument("--chunk-size", type=int, required=True, help="Chunk size assigned to each worker process")
    args = parser.parse_args()

    return args


def main() -> None:
    args = get_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    encoder = Encoder(tokenizer, args.json_keys, args.append_eod)

    pool = multiprocessing.Pool(args.workers)
    print("Opening", args.input)

    if args.input.endswith(".jsonl"):
        print("Input is a jsonl file")

        assert args.subset is None, f"subset argument set to: {args.subset}, but loading a jsonl file."

        fin = open(args.input, "r", encoding="utf-8")
        encoded_docs = pool.imap(encoder.encode, fin, args.chunk_size)
    elif args.input.endswith(".jsonl.zst"):
        print("Input is a jsonl zst file")

        assert args.subset is None, f"subset argument set to: {args.subset}, but loading a zst jsonl file."

        import tempfile

        import zstandard

        dctx = zstandard.ZstdDecompressor()
        outfile = tempfile.TemporaryFile(suffix=args.input.rstrip(".zstd"))
        with open(args.input, "rb") as infile:
            dctx.copy_stream(infile, outfile)
        outfile.seek(0)

        encoded_docs = pool.imap(encoder.encode_jsonl_zstd, outfile, args.chunk_size)
    else:
        print("Input is not a jsonl file, will try to load from HF datasets")

        ds = load_dataset(args.input, use_auth_token=True, streaming=True, split="train", data_dir=args.subset)
        encoded_docs = pool.imap(encoder.encode_hf, ds, args.chunk_size)

    builders = {
        key: MMapIndexedDatasetBuilder(
            f"{args.output_prefix}_{key}.bin", dtype=DType.optimal_dtype(tokenizer.vocab_size)
        )
        for key in args.json_keys
    }

    for item in tqdm(encoded_docs):
        for key, document in item.items():
            builders[key].add_item(torch.IntTensor(document))
            builders[key].end_document()

    print("Done! Now finalizing.")

    for key in args.json_keys:
        builders[key].finalize(f"{args.output_prefix}_{key}.idx")


if __name__ == "__main__":
    main()
