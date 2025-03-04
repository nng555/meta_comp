from transformers import AutoModelForCausalLM, AutoTokenizer
import os 
from models.llms import AVAILABLE_MODELS 

def download_models():
   
    print("Fetching all models in repository...", flush=True)
    huggingface_ids= {model_name: model(load_model=False).huggingface_id for model_name, model in AVAILABLE_MODELS.items()}

    print("Starting download...", flush=True)
    for class_name, name in huggingface_ids.items():
        print("Trying to download model: ", name, flush=True)
        if os.path.exists(f"model_weights/{name}"):
            print(f"Model {name} already exists. Skipping...", flush=True)
        else: 
            try:
                model = AutoModelForCausalLM.from_pretrained(name)
                tokenizer = AutoTokenizer.from_pretrained(name)
                model.save_pretrained(f"/scratch/mr7401/projects/meta_comp/model_weights/{name}")
                tokenizer.save_pretrained(f"/scratch/mr7401/projects/meta_comp/model_weights/{name}")
                print(f"Model and Tokenizer {name} downloaded successfully", flush=True)
            except Exception as e:
                print(f"Model {name} download failed", flush=True)
                print(e, flush=True)
                continue

if __name__ == "__main__":
    print("Main Running", flush=True)
    download_models()