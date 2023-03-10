# please note that this is not the optimal batch size for training, you can increase this to a much larger number
python -m src.generate \
    --model_name bigscience/bloom-560m \
    --model_class AutoModelForCausalLM \
    --training_inference_type full_finetuning \
    --data_class JSONLinesDataset \
    --input_format $'Classify the sentiment of the sentence:\n__input__\nSentiment:' \
    --output_format $' __output__' \
    --load_path checkpoints/full_finetuning/global_step4000 \
    --output_file output/full_finetuning.jsonl \
    --batch_size 8 \
    --dtype bfloat16 \
    --data_path data
