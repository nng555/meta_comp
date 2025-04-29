from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import torch

def generate_api(n, prompt="", device = "cpu"): 
    huggingface_id = "facebook/opt-125m"
    print(f"\nAPI Run. Num Samples={n}, prompt={prompt}", flush=True)
    # Load
    start_time = time.perf_counter()
    max_length = 512
    model = AutoModelForCausalLM.from_pretrained(huggingface_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(huggingface_id, padding_side="left")
    end_time = time.perf_counter()
    load_time = end_time - start_time
    print(f"    Load Time: {load_time:.4F}", flush=True)

    # Generate
    start_time = time.perf_counter()
    for i in range(n):
        if prompt: 
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            model.generate(input_ids, max_length=max_length, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        else:
            model.generate(max_length=max_length, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    
    end_time = time.perf_counter()
    gen_time = end_time - start_time
    print(f"    Generation Time: {gen_time:.4F}", flush=True)
    return
        

def generate_local(n, prompt = "", device = "cpu"):
    huggingface_id = "facebook/opt-125m"
    print(f"\nLocal Run. Num Samples={n}, prompt={prompt}", flush=True)
    # Load
    start_time = time.perf_counter()
    max_length = 512
    model = AutoModelForCausalLM.from_pretrained("/scratch/mr7401/projects/meta_comp/model_weights/facebook/opt-125m").to(device)
    tokenizer = AutoTokenizer.from_pretrained("/scratch/mr7401/projects/meta_comp/model_weights/facebook/opt-125m", padding_side="left")
    end_time = time.perf_counter()
    load_time = end_time - start_time
    print(f"    Load Time: {load_time:.4F}", flush=True)
    
    # Generate
    start_time = time.perf_counter()
    for i in range(n):
        if prompt: 
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            model.generate(input_ids, max_length=max_length, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        else:
            model.generate(max_length=max_length, do_sample=True, pad_token_id=tokenizer.eos_token_id)

    
    end_time = time.perf_counter()
    gen_time = end_time - start_time
    print(f"    Generation Time: {gen_time:.4F}", flush=True)
    return

def generate_local_batch(n, b, prompt = "", device = "cpu"):
    huggingface_id = "facebook/opt-125m"
    print(f"\nLocal Batch via num_return_sequences. Num Samples={n*b} (batch size={b}, num_gen={n}), prompt = {prompt}", flush=True)
    # Load
    start_time = time.perf_counter()
    max_length = 512
    model = AutoModelForCausalLM.from_pretrained("/scratch/mr7401/projects/meta_comp/model_weights/facebook/opt-125m").to(device)
    tokenizer = AutoTokenizer.from_pretrained("/scratch/mr7401/projects/meta_comp/model_weights/facebook/opt-125m", padding_side="left")
    end_time = time.perf_counter()
    load_time = end_time - start_time
    print(f"    Load Time: {load_time:.4F}", flush=True)
    
    # Generate
    start_time = time.perf_counter()
    for i in range(n):
        if prompt: 
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            model.generate(input_ids, max_length=max_length, do_sample=True, pad_token_id=tokenizer.eos_token_id, num_return_sequences = b)
        else: 
            model.generate(max_length=max_length, do_sample=True, pad_token_id=tokenizer.eos_token_id, num_return_sequences=b)
    
    end_time = time.perf_counter()
    gen_time = end_time - start_time
    print(f"    Generation Time: {gen_time:.4F}", flush=True)
    return

def generate_local_batch_via_prompts(n, b, prompt = "hi", device = "cpu"):
    huggingface_id = "facebook/opt-125m"
    print(f"\nLocal Batch via repeated prompts. Num Samples={n*b} (batch size={b}, num_gen={n}), prompt = {prompt}", flush=True)
    # Load
    start_time = time.perf_counter()
    max_length = 512
    model = AutoModelForCausalLM.from_pretrained("/scratch/mr7401/projects/meta_comp/model_weights/facebook/opt-125m").to(device)
    tokenizer = AutoTokenizer.from_pretrained("/scratch/mr7401/projects/meta_comp/model_weights/facebook/opt-125m", padding_side="left")
    end_time = time.perf_counter()
    load_time = end_time - start_time
    print(f"    Load Time: {load_time:.4F}", flush=True)

    prompts = [prompt for _ in range(b)]
    print(prompts)
    # Generate
    start_time = time.perf_counter()
    for i in range(n):
        input_ids = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding = True).input_ids.to(device)
        model.generate(input_ids, max_length=max_length, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    
    end_time = time.perf_counter()
    gen_time = end_time - start_time
    print(f"    Generation Time: {gen_time:.4F}", flush=True)
    return

def main(n, b):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    n_batches = n //b 
    print("\n---- NO Prompts ----\n")

    print(f"Running opt-125m generation for n_samples = {n}, batch_size = {b}, without prompts", flush=True)
    generate_api(n=n, device = device)
    generate_local(n=n, prompt = None, device = device)
    generate_local_batch(n=n_batches,b=b, prompt =None,device = device)
    
    print("\n---- WITH Prompts ----\n")
    print(f"Running opt-125m generation for n_samples = {n}, batch_size = {b}, with prompt", flush=True)
    generate_api(n=n, prompt = "hi", device = device)
    generate_local(n=n, prompt = "hi", device = device)
    generate_local_batch(n=n_batches,b=b, prompt =None,device = device)
    generate_local_batch_via_prompts(n=n_batches,b=b, prompt ="hi" , device = device)
 

if __name__ == "__main__":    
    main(n=32, b=8)