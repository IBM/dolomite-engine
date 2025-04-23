TOKENIZER_PATH=bigcode/starcoder
INPUT_PATH=/dataset/bluepile/tmp
OUTPUT_PATH=/dataset/bluepile/megatron

python tools/data/convert_fms_data_to_megatron.py \
    --data-subsets _tokenization2arrow \
    --input-path $INPUT_PATH \
    --output-path $OUTPUT_PATH \
    --tokenizer $TOKENIZER_PATH \
    --max-file-size 100 \
    --merge
