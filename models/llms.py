
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import os 
import torch

class Model:
    
    def __init__(self, huggingface_id = "gpt2", local_path = None):
        self.huggingface_id = huggingface_id
        self.local_path = local_path
        self.model = AutoModelForCausalLM.from_pretrained(huggingface_id)
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
        super().__init__(huggingface_id="meta-llama/Llama-3.2-1B")

class Llama32_3B(Model):
    def __init__(self):
        super().__init__(huggingface_id="meta-llama/Llama-3.2-3B")

class Llama31_8B(Model):
    def __init__(self):
        super().__init__(huggingface_id="meta-llama/Llama-3.1-8B")

class Llama31_70B(Model):
    def __init__(self):
        super().__init__(huggingface_id="meta-llama/Llama-3.1-70B")

class Llama31_405B(Model):
    def __init__(self):
        super().__init__(huggingface_id="meta-llama/Llama-3.1-405B")

### GPT Models
class GPT2(Model):
    def __init__(self):
        super().__init__(huggingface_id="gpt2")

class GPT2Medium(Model):
    def __init__(self):
        super().__init__(huggingface_id="gpt2-medium")

class GPT2Large(Model):
    def __init__(self):
        super().__init__(huggingface_id="gpt2-large")

class GPT2XLarge(Model):
    def __init__(self):
        super().__init__(huggingface_id="gpt2-xl")

### OPT Models
class OPT_125M(Model):
    def __init__(self):
        super().__init__(huggingface_id="model_weights/facebook/opt-125m")

class OPT_350M(Model):
    def __init__(self):
        super().__init__(huggingface_id="facebook/opt-350m")

class OPT_1_3B(Model):
    def __init__(self):
        super().__init__(huggingface_id="facebook/opt-1.3b")

class OPT_2_7B(Model):
    def __init__(self):
        super().__init__(huggingface_id="facebook/opt-2.7b")

class OPT_6_7B(Model):
    def __init__(self):
        super().__init__(huggingface_id="facebook/opt-6.7b")

class OPT_13B(Model):
    def __init__(self):
        super().__init__(huggingface_id="facebook/opt-13b")

