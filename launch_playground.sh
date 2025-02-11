#!/bin/bash

# Arg Passing Test
#python3 run_bash.py --multirun command="python3 /scratch/mr7401/projects/meta_comp/arg_test.py" model_name=a

# Making Generation Datasets 
python3 run_bash.py --multirun command="python3 /scratch/mr7401/projects/meta_comp/generation/gen_model_samples.py" args.model_name=GPT2,GPT2XLarge,OPT_125M,OPT_350M,OPT_13B,Llama32_1B,Llama32_3B

# Test Run
#python3 run_bash.py --multirun command="python3 /scratch/mr7401/projects/meta_comp/generation/gen_model_samples.py" args.model_name=GPT2