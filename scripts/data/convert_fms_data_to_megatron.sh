TOKENIZER_PATH=bigcode/starcoder
INPUT_PATH=/dataset/bluepile
OUTPUT_PATH=/dataset/bluepile/megatron
DATA_SUBSETS=commoncrawl
NUM_FILES_PER_JOB=200

python tools/data/convert_fms_data_to_megatron.py \
    --input-path $INPUT_PATH \
    --output-path $OUTPUT_PATH \
    --data-subsets $DATA_SUBSETS \
    --tokenizer $TOKENIZER_PATH \
    --convert \
    --num-files-per-job $NUM_FILES_PER_JOB \
    --blue-vela-job
