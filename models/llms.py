from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import torch
import sys 
import os
sys.path.append("/scratch/mr7401/projects/meta_comp")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

class Model:
    def __init__(self, huggingface_id = "gpt2", local_path = None, use_local_weights=False, name = "", load_model = True):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
        self.name = name
        self.huggingface_id = huggingface_id    
        self.local_path = local_path
        self.use_local_weights = use_local_weights
        self.model = None
        self.tokenizer = None

        if load_model: 
            print(f"Model: Initializing Model, device is set to: {self.device}", flush = True)
            self.load_model_and_tokenizer()
        else: 
            print(f"Only loading model class, without weights loaded. ")
    
    def load_model_and_tokenizer(self):
        if self.use_local_weights and self.local_path:
                print(f"Model: Loading model from local path {self.local_path}", flush = True)
                self.model = AutoModelForCausalLM.from_pretrained(self.local_path).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(self.local_path, padding_side="left")
        else: 
            if self.use_local_weights:
                warnings.warn("\n\n\n\n\n ******** WARNING: Parameter 'use_local_weights' is set to True but no local path was provided. Loading model from huggingface instead. *****")
            print(f"Model: Loading model from huggingface with huggingface id {self.huggingface_id}", flush = True)
            self.model = AutoModelForCausalLM.from_pretrained(self.huggingface_id).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.huggingface_id, padding_side="left")
        
        # Set up special tokens and get the max token length
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id

        possible_max_token_lengths = [self.tokenizer.model_max_length]
        if hasattr(self.model, "config"):
            possible_max_token_lengths.append(self.model.config.max_position_embeddings)
        
        self.max_token_length = min(possible_max_token_lengths)
        print(f"Model: Setting the Max Token Length to {self.max_token_length}", flush = True)
        return 
    
    
    def generate(self, prompts = None, max_length = 50, num_return_sequences = 1):
        
        # If any prompts are provided, generate text conditioned on the prompt
        if prompts:
            if isinstance(prompts, str):
                prompts = [prompts]
            
            inputs = self.tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True, truncation=True,  max_length= self.max_token_length)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            outputs = self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=None, max_length=max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id, num_return_sequences=num_return_sequences)
        else:
            outputs = self.model.generate(max_length=max_length, max_new_tokens=None, do_sample=True, pad_token_id=self.tokenizer.eos_token_id, num_return_sequences=num_return_sequences)

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True, padding=True)
    
    def calculate_perplexity(self, text, device="cpu", verbose=False, context_window_limit=None, stride_denominator=2):
        """
        Purpose: Calculate the perplexity of a given text using the model.
        ## Source: adapted from https://huggingface.co/docs/transformers/perplexity 
        ## Relevant: https://thegradient.pub/understanding-evaluation-metrics-for-language-models/"""
        encodings = self.tokenizer([text], return_tensors="pt").to(self.device)
        if context_window_limit is not None:
            max_length = min(self.max_token_length, context_window_limit)
        else:
            max_length = self.max_token_length

        stride = max_length // stride_denominator

        seq_len = encodings.input_ids.size(1)
        if verbose:
            print(f"sequence length: {seq_len}, stride = {stride}, max_length = {max_length}")

        nll_sum = 0.0
        n_tokens = 0
        prev_end_loc = 0

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone().to(self.device)
            target_ids[:, :-trg_len] = -100

            if verbose:
                print(f"------\nProcessing tokens from {begin_loc} to {end_loc}")
                print(f"Target length (trg_len): {trg_len}")
                # Decode and print each token in the current input_ids
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
                for idx, token in enumerate(tokens):
                    print(f"    Token {idx + begin_loc}: {token}")

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
            batch_size = target_ids.size(0)
            num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
            nll_sum += neg_log_likelihood * num_loss_tokens
            n_tokens += num_loss_tokens

            if verbose:
                print(f"\nString Subset Statistics (Token Range: {begin_loc}-{end_loc}):")
                print(f"    Negative Log-Likelihood: {neg_log_likelihood}")
                print(f"    Number of valid tokens: {num_valid_tokens}")
                print(f"    Number of loss tokens: {num_loss_tokens}")
                print(f"    Accumulated NLL Sum: {nll_sum}")
                print(f"    Total Tokens Processed: {n_tokens}")

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
        ppl = torch.exp(avg_nll)
        if verbose:
            print(f"\nCombined Statistics (over {n_tokens} Tokens):")
            print(f"    Total NLL Sum: {nll_sum}")
            print(f"    Average NLL: {avg_nll}")
            print(f"    Perplexity: {ppl}")
        return ppl, avg_nll

    
    def to_tokens_and_logprobs(self, texts = ["One plus one is two"], verbose = False):
        """ 
        Purpose: Given a set of text strings, this function calculates and returns the log probabilites of each sequence.
        # Adapted from HF https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/17?u=meganrichards3
        
        Texts: a n-sized list of strings e.g ["One plus one is two", "I went to the store"] 
        Returns: a n-sized list of floats, with return[i] representing the sum of the log probability of sequence i. 

        """

        # Tokenize the text and pass to the model
        if verbose:
            print(f"Model: given texts = {texts}", flush=True)

        input_ids = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_token_length, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model(input_ids)
        if verbose:
            print(f"Model: Output size = {outputs.logits.size()}", flush=True)
       
        probs = torch.log_softmax(outputs.logits, dim=-1).detach()
        #print(f"Model: probs = {probs}")

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

        # convert to floats
        log_probs_as_floats = []
        for log_prob in sequence_log_probs:
            if isinstance(log_prob, torch.Tensor):
                log_probs_as_floats.append(float(log_prob.cpu().item()))
            else: 
                log_probs_as_floats.append(float(log_prob))

        assert len(log_probs_as_floats) == len(texts), f"Length of sequence log probs {len(log_probs_as_floats)} does not match length of texts {len(texts)}"
        return log_probs_as_floats

### Llama Models 
class Llama32_1B(Model):
    def __init__(self, local_path="/scratch/mr7401/projects/meta_comp/model_weights/meta-llama/Llama-3.2-1B", use_local_weights=False, load_model=True):
        super().__init__(huggingface_id="meta-llama/Llama-3.2-1B", name="Llama32_1B", local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)

class Llama32_3B(Model):
    def __init__(self, local_path="/scratch/mr7401/projects/meta_comp/model_weights/meta-llama/Llama-3.2-3B", use_local_weights=False, load_model=True):
        super().__init__(huggingface_id="meta-llama/Llama-3.2-3B", name="Llama32_3B", local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)

class Llama31_8B(Model):
    def __init__(self, local_path="/scratch/mr7401/projects/meta_comp/model_weights/meta-llama/Llama-3.1-8B", use_local_weights=False, load_model=True):
        super().__init__(huggingface_id="meta-llama/Llama-3.1-8B", name="Llama31_8B", local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)

class Llama31_70B(Model):
    def __init__(self, local_path="/scratch/mr7401/projects/meta_comp/model_weights/meta-llama/Llama-3.1-70B", use_local_weights=False, load_model=True):
        super().__init__(huggingface_id="meta-llama/Llama-3.1-70B", name="Llama31_70B", local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)

class Llama31_405B(Model):
    def __init__(self, local_path="/scratch/mr7401/projects/meta_comp/model_weights/meta-llama/Llama-3.1-405B", use_local_weights=False, load_model=True):
        super().__init__(huggingface_id="meta-llama/Llama-3.1-405B", name="Llama31_405B", local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)

class Llama3_8B(Model):
    def __init__(self, local_path="/scratch/mr7401/projects/meta_comp/model_weights/meta-llama/Meta-Llama-3-8B", use_local_weights=False, load_model=True):
        super().__init__(huggingface_id="meta-llama/Meta-Llama-3-8B", name="Llama3_8B", local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)

class Llama3_70B(Model):
    def __init__(self, local_path="/scratch/mr7401/projects/meta_comp/model_weights/meta-llama/Llama-3-70B", use_local_weights=False, load_model=True):
        super().__init__(huggingface_id="meta-llama/Llama-3-70B", name="Llama3_70B", local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)

class Llama2_7B(Model):
    def __init__(self, local_path="/scratch/mr7401/projects/meta_comp/model_weights/meta-llama/Llama-2-7B-hf", use_local_weights=False, load_model=True):
        super().__init__(huggingface_id="meta-llama/Llama-2-7B-hf", name="Llama2_7B", local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)

class Llama2_13B(Model):
    def __init__(self, local_path="/scratch/mr7401/projects/meta_comp/model_weights/meta-llama/Llama-2-13B-hf", use_local_weights=False, load_model=True):
        super().__init__(huggingface_id="meta-llama/Llama-2-13B-hf", name="Llama2_13B", local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)

### OPT Models 
class OPT125M(Model):
    def __init__(self, local_path="/scratch/mr7401/projects/meta_comp/model_weights/facebook/opt-125m", use_local_weights=False, load_model=True):
        super().__init__(huggingface_id="facebook/opt-125m", name="OPT125M", local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)

class OPT350M(Model):
    def __init__(self, local_path="/scratch/mr7401/projects/meta_comp/model_weights/facebook/opt-350m", use_local_weights=False, load_model=True):
        super().__init__(huggingface_id="facebook/opt-350m", name="OPT350M", local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)

class OPT1_3B(Model):
    def __init__(self, local_path="/scratch/mr7401/projects/meta_comp/model_weights/facebook/opt-1.3b", use_local_weights=False, load_model=True):
        super().__init__(huggingface_id="facebook/opt-1.3b", name="OPT1_3B", local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)

class OPT2_7B(Model):
    def __init__(self, local_path="/scratch/mr7401/projects/meta_comp/model_weights/facebook/opt-2.7b", use_local_weights=False, load_model=True):
        super().__init__(huggingface_id="facebook/opt-2.7b", name="OPT2_7B", local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)

class OPT6_7B(Model):
    def __init__(self, local_path="/scratch/mr7401/projects/meta_comp/model_weights/facebook/opt-6.7b", use_local_weights=False, load_model=True):
        super().__init__(huggingface_id="facebook/opt-6.7b", name="OPT6_7B", local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)

class OPT13B(Model):
    def __init__(self, local_path="/scratch/mr7401/projects/meta_comp/model_weights/facebook/opt-13b", use_local_weights=False, load_model=True):
        super().__init__(huggingface_id="facebook/opt-13b", name="OPT13B", local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)

class OPT30B(Model):
    def __init__(self, local_path="/scratch/mr7401/projects/meta_comp/model_weights/facebook/opt-30b", use_local_weights=False, load_model=True):
        super().__init__(huggingface_id="facebook/opt-30b", name="OPT30B", local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)

class OPT66B(Model):
    def __init__(self, local_path="/scratch/mr7401/projects/meta_comp/model_weights/facebook/opt-66b", use_local_weights=False, load_model=True):
        super().__init__(huggingface_id="facebook/opt-66b", name="OPT66B", local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)

### Gemma Models
class Gemma_2B(Model):
    def __init__(self, local_path="/scratch/mr7401/projects/meta_comp/model_weights/google/gemma-2b", use_local_weights=False, load_model=True):
        super().__init__(huggingface_id="google/gemma-2b", name="Gemma_2B", local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)

class Gemma_7B(Model):
    def __init__(self, local_path="/scratch/mr7401/projects/meta_comp/model_weights/google/gemma-7b", use_local_weights=False, load_model=True):
        super().__init__(huggingface_id="google/gemma-7b", name="Gemma_7B", local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)

class Gemma2_2B(Model):
    def __init__(self, local_path="/scratch/mr7401/projects/meta_comp/model_weights/google/gemma-2-2b", use_local_weights=False, load_model=True):
        super().__init__(huggingface_id="google/gemma-2-2b", name="Gemma2_2B", local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)

class Gemma2_9B(Model):
    def __init__(self, local_path="/scratch/mr7401/projects/meta_comp/model_weights/google/gemma-2-9b", use_local_weights=False, load_model=True):
        super().__init__(huggingface_id="google/gemma-2-9b", name="Gemma2_9B", local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)

class CodeGemma2B(Model):
    def __init__(self, local_path="/scratch/mr7401/projects/meta_comp/model_weights/google/codegemma-2b", use_local_weights=False, load_model=True):
        super().__init__(huggingface_id="google/codegemma-2b", name="CodeGemma2B", local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)

class CodeGemma7B(Model):
    def __init__(self, local_path="/scratch/mr7401/projects/meta_comp/model_weights/google/codegemma-7b", use_local_weights=False, load_model=True):
        super().__init__(huggingface_id="google/codegemma-7b", name="CodeGemma7B", local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)

### GPT Models
class GPT2(Model):
    def __init__(self, local_path="/scratch/mr7401/projects/meta_comp/model_weights/gpt2", use_local_weights=False, load_model=True):
        super().__init__(huggingface_id="gpt2", name="GPT2", local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)

class GPT2Medium(Model):
    def __init__(self, local_path="/scratch/mr7401/projects/meta_comp/model_weights/gpt2-medium", use_local_weights=False, load_model=True):
        super().__init__(huggingface_id="gpt2-medium", name="GPT2Medium", local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)

class GPT2Large(Model):
    def __init__(self, local_path="/scratch/mr7401/projects/meta_comp/model_weights/gpt2-large", use_local_weights=False, load_model=True):
        super().__init__(huggingface_id="gpt2-large", name="GPT2Large", local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)

class GPT2XLarge(Model):
    def __init__(self, local_path="/scratch/mr7401/projects/meta_comp/model_weights/gpt2-xl", use_local_weights=False, load_model=True):
        super().__init__(huggingface_id="gpt2-xl", name="GPT2XLarge", local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)

### Bloom Models
class Bloom(Model):
    def __init__(self, local_path="/scratch/mr7401/projects/meta_comp/model_weights/bigscience/bloom", use_local_weights=False, load_model=True):
        super().__init__(huggingface_id="bigscience/bloom", name="Bloom", local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)

class Bloom560M(Model):
    def __init__(self, local_path="/scratch/mr7401/projects/meta_comp/model_weights/bigscience/bloom-560m", use_local_weights=False, load_model=True):
        super().__init__(huggingface_id="bigscience/bloom-560m", name="Bloom560M", local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)

class Bloom1B7(Model):
    def __init__(self, local_path="/scratch/mr7401/projects/meta_comp/model_weights/bigscience/bloom-1b7", use_local_weights=False, load_model=True):
        super().__init__(huggingface_id="bigscience/bloom-1b7", name="Bloom1B7", local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)

class Bloom7B1(Model):
    def __init__(self, local_path="/scratch/mr7401/projects/meta_comp/model_weights/bigscience/bloom-7b1", use_local_weights=False, load_model=True):
        super().__init__(huggingface_id="bigscience/bloom-7b1", name="Bloom7B1", local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)

class Qwen2_5_0_5B(Model):
    def __init__(self, local_path="/scratch/mr7401/projects/meta_comp/model_weights/qwen/Qwen2.5-0.5B", use_local_weights=False, load_model=True):
        super().__init__(huggingface_id="qwen/Qwen2.5-0.5B", name="Qwen2_5_0_5B", local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)

class Qwen2_5_3B(Model):
    def __init__(self, local_path="/scratch/mr7401/projects/meta_comp/model_weights/qwen/Qwen2.5-3B", use_local_weights=False, load_model=True):
        super().__init__(huggingface_id="qwen/Qwen2.5-3B", name="Qwen2_5_3B", local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)

def get_model(model_name, local_path=None, use_local_weights=False, load_model=True):
    """
    Purpose: Allow other scripts to instantiate a model class based on its string and any desired model parameters. 
    """
    if model_name in AVAILABLE_MODELS:
        if local_path is not None:
            return AVAILABLE_MODELS[model_name](local_path=local_path, use_local_weights=use_local_weights, load_model=load_model)
        else:
            return AVAILABLE_MODELS[model_name](use_local_weights=use_local_weights, load_model=load_model)
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
    #"OPT13B": OPT13B,
    #"OPT30B": OPT30B,
    #"OPT66B": OPT66B, # so big...
    "GPT2": GPT2,
    "GPT2Medium": GPT2Medium,
    "GPT2Large": GPT2Large,
    "GPT2XLarge": GPT2XLarge,
    "Gemma_2B": Gemma_2B,
    "Gemma_7B": Gemma_7B,
    "Gemma2_2B": Gemma2_2B, 
    "Gemma2_9B": Gemma2_9B,
    "CodeGemma2B": CodeGemma2B,
    "CodeGemma7B": CodeGemma7B,
    #"Bloom": Bloom,
    #"Bloom560M": Bloom560M,
    #"Bloom1B7": Bloom1B7,
    #"Bloom7B1": Bloom7B1,
    "Qwen2_5_0_5B": Qwen2_5_0_5B, 
    "Qwen2_5_3B": Qwen2_5_3B 
}

def get_available_llms():
    return AVAILABLE_MODELS
