
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import torch

class Model:
    def __init__(self, huggingface_id = "gpt2", local_path = None, use_local_weights=False):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"    Model: Initializing Model, loading on {self.device}")

        self.huggingface_id = huggingface_id
        self.local_path = local_path
        self.use_local_weights = use_local_weights

        if self.use_local_weights and self.local_path:
            print(f"    Model: Loading model from local path {self.local_path}")
            self.model = AutoModelForCausalLM.from_pretrained(self.local_path).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_path, padding_side="left")
        else: 
            if self.use_local_weights:
                warnings.warn("\n\n\n\n\n ******** WARNING: Parameter 'use_local_weights' is set to True but no local path was provided. Loading model from huggingface instead. *****")
            print(f"    Model: Loading model from huggingface with huggingface id {self.huggingface_id}")
            self.model = AutoModelForCausalLM.from_pretrained(huggingface_id).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(huggingface_id, padding_side="left")
   
    def generate(self, prompt = None, max_length = 50):
        # If a prompt is provided, generate text conditioned on the prompt
        if prompt:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(input_ids, max_length=max_length, pad_token_id=self.tokenizer.eos_token_id)
        else:
            outputs = self.model.generate(max_length=max_length, pad_token_id=self.tokenizer.eos_token_id)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def to_tokens_and_logprobs(self, texts = ["One plus one is two"]):
        """ 
        Purpose: Given a set of text strings, this function calculates and returns the log probabilites of each sequence.
        # Adapted from HF https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/17?u=meganrichards3
        
        Texts: a n-sized list of strings e.g ["One plus one is two", "I went to the store"] 
        Returns: a n-sized list of floats, with return[i] representing the sum of the log probability of sequence i. 

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
    def __init__(self, local_path=None, use_local_weights=False):
        super().__init__(huggingface_id="meta-llama/Llama-3.2-1B", local_path=local_path, use_local_weights=use_local_weights)

class Llama32_3B(Model):
    def __init__(self, local_path=None, use_local_weights=False):
        super().__init__(huggingface_id="meta-llama/Llama-3.2-3B", local_path=local_path, use_local_weights=use_local_weights)

class Llama31_8B(Model):
    def __init__(self, local_path=None, use_local_weights=False):
        super().__init__(huggingface_id="meta-llama/Llama-3.1-8B", local_path=local_path, use_local_weights=use_local_weights)

class Llama31_70B(Model):
    def __init__(self, local_path=None, use_local_weights=False):
        super().__init__(huggingface_id="meta-llama/Llama-3.1-70B", local_path=local_path, use_local_weights=use_local_weights)

class Llama31_405B(Model):
    def __init__(self, local_path=None, use_local_weights=False):
        super().__init__(huggingface_id="meta-llama/Llama-3.1-405B", local_path=local_path, use_local_weights=use_local_weights)

class Llama3_8B(Model):
    def __init__(self, local_path=None, use_local_weights=False):
        super().__init__(huggingface_id="meta-llama/Llama-3-8B", local_path=local_path, use_local_weights=use_local_weights)

class Llama3_70B(Model):
    def __init__(self, local_path=None, use_local_weights=False):
        super().__init__(huggingface_id="meta-llama/Llama-3-70B", local_path=local_path, use_local_weights=use_local_weights)

class Llama2_7B(Model):
    def __init__(self, local_path=None, use_local_weights=False):
        super().__init__(huggingface_id="meta-llama/Llama-2-7B", local_path=local_path, use_local_weights=use_local_weights)

class Llama2_13B(Model):
    def __init__(self, local_path=None, use_local_weights=False):
        super().__init__(huggingface_id="meta-llama/Llama-2-13B", local_path=local_path, use_local_weights=use_local_weights)

### OPT Models 
class OPT125M(Model):
    def __init__(self, local_path=None, use_local_weights=False):
        super().__init__(huggingface_id="facebook/opt-125m", local_path=local_path, use_local_weights=use_local_weights)

class OPT350M(Model):
    def __init__(self, local_path=None, use_local_weights=False):
        super().__init__(huggingface_id="facebook/opt-350m", local_path=local_path, use_local_weights=use_local_weights)

class OPT1_3B(Model):
    def __init__(self, local_path=None, use_local_weights=False):
        super().__init__(huggingface_id="facebook/opt-1.3b", local_path=local_path, use_local_weights=use_local_weights)

class OPT2_7B(Model):
    def __init__(self, local_path=None, use_local_weights=False):
        super().__init__(huggingface_id="facebook/opt-2.7b", local_path=local_path, use_local_weights=use_local_weights)

class OPT6_7B(Model):
    def __init__(self, local_path=None, use_local_weights=False):
        super().__init__(huggingface_id="facebook/opt-6.7b", local_path=local_path, use_local_weights=use_local_weights)

class OPT13B(Model):
    def __init__(self, local_path=None, use_local_weights=False):
        super().__init__(huggingface_id="facebook/opt-13b", local_path=local_path, use_local_weights=use_local_weights)

class OPT30B(Model):
    def __init__(self, local_path=None, use_local_weights=False):
        super().__init__(huggingface_id="facebook/opt-30b", local_path=local_path, use_local_weights=use_local_weights)

class OPT66B(Model):
    def __init__(self, local_path=None, use_local_weights=False):
        super().__init__(huggingface_id="facebook/opt-66b", local_path=local_path, use_local_weights=use_local_weights)

### Gemma Models
class Gemma2B(Model):
    def __init__(self, local_path=None, use_local_weights=False):
        super().__init__(huggingface_id="google/gemma-2b", local_path=local_path, use_local_weights=use_local_weights)

class Gemma7B(Model):
    def __init__(self, local_path=None, use_local_weights=False):
        super().__init__(huggingface_id="google/gemma-7b", local_path=local_path, use_local_weights=use_local_weights)

class CodeGemma2B(Model):
    def __init__(self, local_path=None, use_local_weights=False):
        super().__init__(huggingface_id="google/codegemma-2b", local_path=local_path, use_local_weights=use_local_weights)

class CodeGemma7B(Model):
    def __init__(self, local_path=None, use_local_weights=False):
        super().__init__(huggingface_id="google/codegemma-7b", local_path=local_path, use_local_weights=use_local_weights)

### GPT Models
class GPT2(Model):
    def __init__(self, local_path=None, use_local_weights=False):
        super().__init__(huggingface_id="gpt2", local_path=local_path, use_local_weights=use_local_weights)

class GPT2Medium(Model):
    def __init__(self, local_path=None, use_local_weights=False):
        super().__init__(huggingface_id="gpt2-medium", local_path=local_path, use_local_weights=use_local_weights)

class GPT2Large(Model):
    def __init__(self, local_path=None, use_local_weights=False):
        super().__init__(huggingface_id="gpt2-large", local_path=local_path, use_local_weights=use_local_weights)

class GPT2XLarge(Model):
    def __init__(self, local_path=None, use_local_weights=False):
        super().__init__(huggingface_id="gpt2-xl", local_path=local_path, use_local_weights=use_local_weights)

### Bloom Models
class Bloom(Model):
    def __init__(self, local_path=None, use_local_weights=False):
        super().__init__(huggingface_id="bigscience/bloom", local_path=local_path, use_local_weights=use_local_weights)

class Bloom560M(Model):
    def __init__(self, local_path=None, use_local_weights=False):
        super().__init__(huggingface_id="bigscience/bloom-560m", local_path=local_path, use_local_weights=use_local_weights)

class Bloom1B7(Model):
    def __init__(self, local_path=None, use_local_weights=False):
        super().__init__(huggingface_id="bigscience/bloom-1b7", local_path=local_path, use_local_weights=use_local_weights)

class Bloom7B1(Model):
    def __init__(self, local_path=None, use_local_weights=False):
        super().__init__(huggingface_id="bigscience/bloom-7b1", local_path=local_path, use_local_weights=use_local_weights)


def get_model(model_name, local_path= None, use_local_weights=False):
    """
    Purpose: Allow other scripts to instatiate a model class based on it's string and any desired model parameters. 
    """
    if model_name in AVAILABLE_MODELS:
        return AVAILABLE_MODELS[model_name](local_path =local_path, use_local_weights=use_local_weights)
    else:
        raise ValueError(f"Model {model_name} is not available. Currently available models are: {AVAILABLE_MODELS.keys()}")


AVAILABLE_MODELS = {
    "Llama32_1B": Llama32_1B,
    "Llama32_3B": Llama32_3B,
    "Llama31_8B": Llama31_8B,
    "Llama31_70B": Llama31_70B,
    "Llama31_405B": Llama31_405B,
    "Llama3_8B": Llama3_8B,
    "Llama3_70B": Llama3_70B,
    "Llama2_7B": Llama2_7B,
    "Llama2_13B": Llama2_13B,
    "OPT125M": OPT125M,
    "OPT350M": OPT350M,
    "OPT1_3B": OPT1_3B,
    "OPT2_7B": OPT2_7B,
    "OPT6_7B": OPT6_7B,
    "OPT13B": OPT13B,
    "OPT30B": OPT30B,
    "OPT66B": OPT66B,
    "GPT2": GPT2,
    "GPT2Medium": GPT2Medium,
    "GPT2Large": GPT2Large,
    "GPT2XLarge": GPT2XLarge,
    "Gemma2B": Gemma2B,
    "Gemma7B": Gemma7B,
    "CodeGemma2B": CodeGemma2B,
    "CodeGemma7B": CodeGemma7B,
    "Bloom": Bloom,
    "Bloom560M": Bloom560M,
    "Bloom1B7": Bloom1B7,
    "Bloom7B1": Bloom7B1
}

def get_available_llms():
    return AVAILABLE_MODELS
