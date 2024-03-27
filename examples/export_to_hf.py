from dolomite_engine.hf_models import export_to_huggingface_llama


load_path = "load/"
save_path = "save/"

# export to HF llama
export_to_huggingface_llama(load_path, save_path)
