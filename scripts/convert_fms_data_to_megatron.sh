TOKENIZER_PATH=bigcode/starcoder
INPUT_PATH=/dataset/bluepile
TMP_PATH=/dataset/bluepile/tmp
OUTPUT_PATH=/dataset/bluepile/megatron

python tools/data/convert_fms_data_to_megatron.py \
    --input-path $INPUT_PATH \
    --data-subsets _tokenization2arrow \
    --tmp-path $TMP_PATH \
    --output-path $OUTPUT_PATH \
    --tokenizer $TOKENIZER_PATH \
    --convert \
    --num-files-per-job 200 \
    --blue-vela-job
