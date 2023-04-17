# please note that this is not the optimal batch size for training, you can increase this to a much larger number
python -m src.generate \
    --model_name bigscience/bloom-560m \
    --model_class AutoModelForCausalLM \
    --training_inference_type prompt_tuning \
    --dataset_configs_json configs/jsonlines.json \
    --prompt_tuning_init TEXT \
    --prompt_tuning_init_text $'Classify the sentiment of the sentence:' \
    --num_virtual_tokens 8 \
    --load_path checkpoints/prompt_tuning/global_step4000 \
    --output_file output/sst2-prompt_tuning.jsonl \
    --batch_size 8 \
    --dtype bfloat16
