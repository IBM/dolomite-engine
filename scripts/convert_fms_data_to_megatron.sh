TOKENIZER_PATH=tokenizers/gpt-neox-20b

DATASET_LIST=sampling_proportions/bluepile-03-granite/proportions.txt

INPUT_PATH=/dataset/bluepile
TMP_PATH=/dataset/bluepile/tmp
OUTPUT_PATH=/dataset/bluepile/megatron

python tools/convert_fms_data_to_megatron.py \
    --input-path $INPUT_PATH \
    --dataset-list $DATASET_LIST \
    --tmp-path $TMP_PATH \
    --output-path $OUTPUT_PATH \
    --tokenizer-path $TOKENIZER_PATH \
    --convert \
    --ccc-job
