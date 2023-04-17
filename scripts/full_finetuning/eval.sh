# please note that this is not the optimal batch size for training, you can increase this to a much larger number
python -m src.generate \
    --model_name bigscience/bloom-560m \
    --model_class AutoModelForCausalLM \
    --training_inference_type full_finetuning \
    --dataset_configs_json configs/sst2-full_finetuning.json \
    --load_path checkpoints/full_finetuning/global_step4000 \
    --output_file output/full_finetuning.jsonl \
    --batch_size 8 \
    --dtype bfloat16
