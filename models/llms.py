
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import os 
import torch

class Model:
    
    def __init__(self, huggingface_id = "gpt2", local_path = None, use_local_weights=True):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Loading model on {self.device}")

        self.huggingface_id = huggingface_id
        self.local_path = local_path
        self.use_local_weights = use_local_weights

        if self.use_local_weights and self.local_path:
            self.model = AutoModelForCausalLM.from_pretrained(self.local_path).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_path, padding_side="left")
        else: 
            self.model = AutoModelForCausalLM.from_pretrained(huggingface_id).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(huggingface_id, padding_side="left")
   
    def generate(self, prompt = None, max_length = 50):
        # If a prompt is provided, generate text conditioned on the prompt
        if prompt:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            outputs = self.model.generate(input_ids, max_length=max_length, pad_token_id=self.tokenizer.eos_token_id)
        else:
            outputs = self.model.generate(max_length=max_length, pad_token_id=self.tokenizer.eos_token_id)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def to_tokens_and_logprobs(self, texts = ["One plus one is two"]):
        """ 
        Purpose: Given a set of text strings, this function calculates and returns the log probabilites of each sequence.
        # Adapted from HF https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/17?u=meganrichards3
        
        Texts: a n-sized list of strings e.g ["One plus one is two", "I went to the store"] 
        Returns: a n-sized list of floats, with return[i] representing the sum of the log probability of sequence i in the provided 'outputs'. 

        """
        # Get the model EOS token 
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id

        # Tokenize the text and pass to the model
        input_ids = self.tokenizer(texts, padding=True, return_tensors="pt").input_ids
        outputs = self.model(input_ids)

        probs = torch.log_softmax(outputs.logits, dim=-1).detach()

        # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
        probs = probs[:, :-1, :]
        input_ids = input_ids[:, 1:]
        gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

        token_log_probs = []
        sequence_log_probs = []
        for input_sentence, input_probs in zip(input_ids, gen_probs):
            text_sequence = []
            prob_sequence = []
            for token, p in zip(input_sentence, input_probs):
                if token not in self.tokenizer.all_special_ids:
                    text_sequence.append((self.tokenizer.decode(token), p.item()))
                    prob_sequence.append(p.item())
            
            token_log_probs.append(text_sequence) # [(token, log_prob), ...]], e.g. [("One", e-24), ("plus", e-25), ...]
            sequence_log_probs.append(sum(prob_sequence)) # [log_prob, ...], e.g. [e-24 + e-25 + ...]

        return sequence_log_probs

### Llama Models 
class Llama32_1B(Model):
    def __init__(self):
        super().__init__(huggingface_id="meta-llama/Llama-3.2-1B", local_path = "/scratch/mr7401/projects/meta_comp/model_weights/meta-llama/Llama-3.2-1B")

class Llama31_8B(Model):
    def __init__(self):
        super().__init__(huggingface_id="meta-llama/Llama-3.1-8B", local_path = "/scratch/mr7401/projects/meta_comp/model_weights/meta-llama/Llama-3.1-8B")

### GPT Models
class GPT2(Model):
    def __init__(self):
        super().__init__(huggingface_id="gpt2", local_path = "/scratch/mr7401/projects/meta_comp/model_weights/gpt2")

class GPT2Medium(Model):
    def __init__(self):
        super().__init__(huggingface_id="gpt2-medium", local_path = "/scratch/mr7401/projects/meta_comp/model_weights/gpt-medium")

class GPT2Large(Model):
    def __init__(self):
        super().__init__(huggingface_id="gpt2-large", local_path = "/scratch/mr7401/projects/meta_comp/model_weights/gpt2-large")

class GPT2XLarge(Model):
    def __init__(self):
        super().__init__(huggingface_id="gpt2-xl", local_path = "/scratch/mr7401/projects/meta_comp/model_weights/gpt2-xl")

### OPT Models
class OPT_125M(Model):
    def __init__(self):
        super().__init__(huggingface_id="model_weights/facebook/opt-125m", local_path = "/scratch/mr7401/projects/meta_comp/model_weights/facebook/opt-125m")

class OPT_350M(Model):
    def __init__(self):
        super().__init__(huggingface_id="facebook/opt-350m", local_path = "/scratch/mr7401/projects/meta_comp/model_weights/facebook/opt-350m")

class OPT_1_3B(Model):
    def __init__(self):
        super().__init__(huggingface_id="facebook/opt-1.3b", local_path = "/scratch/mr7401/projects/meta_comp/model_weights/facebook/opt-1.3B")

class OPT_2_7B(Model):
    def __init__(self):
        super().__init__(huggingface_id="facebook/opt-2.7b", local_path = "/scratch/mr7401/projects/meta_comp/model_weights/facebook/opt-2.7B")

class OPT_6_7B(Model):
    def __init__(self):
        super().__init__(huggingface_id="facebook/opt-6.7b", local_path = "/scratch/mr7401/projects/meta_comp/model_weights/facebook/opt-6.7B")

class OPT_13B(Model):
    def __init__(self):
        super().__init__(huggingface_id="facebook/opt-13b", local_path = "/scratch/mr7401/projects/meta_comp/model_weights/facebook/opt-13B")

def get_model(model_name, use_local_weights = False):
    if model_name in AVAILABLE_MODELS:
        return AVAILABLE_MODELS[model_name](use_local_weights = use_local_weights)
    else:
        raise ValueError(f"Model {model_name} is not available. Currently available models are: {AVAILABLE_MODELS.keys()}")


AVAILABLE_MODELS = {
    "Llama32_1B": Llama32_1B,
    "Llama31_8B": Llama31_8B,
    "GPT2": GPT2,
    "GPT2Medium": GPT2Medium,
    "GPT2Large": GPT2Large,
    "GPT2XLarge": GPT2XLarge,
    "OPT_125M": OPT_125M,
    "OPT_350M": OPT_350M,
    "OPT_1_3B": OPT_1_3B,
    "OPT_2_7B": OPT_2_7B,
    "OPT_6_7B": OPT_6_7B,
    "OPT_13B": OPT_13B
}

def get_available_llms():
    return AVAILABLE_MODELS
