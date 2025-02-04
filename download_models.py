import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import os 

def download_models():
   
    # Add your model downloading logic here
    print("Downloading models...")
    ### Downloading models 
    names = [
    "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B" ,
    "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-405B", 
    "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", 
    "facebook/opt-125m", "facebook/opt-350m",  "facebook/opt-1.3B",  "facebook/opt-2.7B","facebook/opt-6.7B", "facebook/opt-13B", "facebook/opt-30B", "facebook/opt-66B"
    ]
    for name in names:
        print("Trying to download model: ", name)
        if os.path.exists(f"model_weights/{name}"):
            print(f"Model {name} already exists. Skipping...")
            continue
        else: 
            try:
                model = AutoModelForCausalLM.from_pretrained(name)
                tokenizer = AutoTokenizer.from_pretrained(name)
                model.save_pretrained(f"/scratch/mr7401/projects/meta_comp/model_weights/{name}")
                tokenizer.save_pretrained(f"/scratch/mr7401/projects/meta_comp/model_weights/{name}")
                print(f"Model and Tokenizer {name} downloaded successfully")
            except Exception as e:
                print(f"Model {name} download failed")
                print(e)
                continue

if __name__ == "__main__":
    print("Main Running")
    download_models()