#!/bin/bash

# Arg Passing Test
#python3 run_bash.py --multirun command="python3 /scratch/mr7401/projects/meta_comp/arg_test.py" model_name=a

# Simple Generation Run
#python3 run_bash.py --multirun command="python3 /scratch/mr7401/projects/meta_comp/generation/gen_model_samples.py" args.model_name=GPT2

# All Model Names
#Llama32_1B,Llama32_3B,Llama31_8B,Llama31_70B,Llama31_405B,OPT125M,OPT350M,OPT1_3B,OPT2_7B,OPT6_7B,OPT13B,OPT30B,OPT66B,GPT2,GPT2Medium,GPT2Large,GPT2XLarge,Gemma2B,Gemma7B,CodeGemma2B,CodeGemma7B,Bloom,Bloom560M,Bloom1B7,Bloom7B1

# Playgroun
python3 run_bash.py --multirun command="python3 /scratch/mr7401/projects/meta_comp/generation/gen_model_samples.py" args.model_name=Llama3_8B,Llama2_7B,Gemma_2B,Gemma_7B,Gemma2_2B,Gemma2_9B

