#!/bin/bash

# Tests
#python3 run_bash.py --multirun command="python3 /scratch/mr7401/projects/meta_comp/arg_test.py" model_name=a
#python3 run_bash.py --multirun command="python3 /scratch/mr7401/projects/meta_comp/tests/test_generation_strategies.py" hydra.launcher.mem=5G hydra.launcher.gres=gpu:1 hydra.launcher.time=10 logging.experiment=test

# Download Model Weights 
#python3 run_bash.py --multirun command="python3 /scratch/mr7401/projects/meta_comp/download_models.py" hydra.launcher.mem=100G hydra.launcher.gres=0 hydra.launcher.time=60 logging.experiment=model_download

# Simple Generation Run
#python3 run_bash.py --multirun command="python3 /scratch/mr7401/projects/meta_comp/generation/gen_model_samples.py" args.model_name=Qwen2_5_3B args.use_local_weights=True hydra.launcher.mem=1G hydra.launcher.gres=0 hydra.launcher.time=10 logging.experiment=test args.test=True

# Generate Samples
#python3 run_bash.py --multirun command="python3 /scratch/mr7401/projects/meta_comp/generation/gen_model_samples.py" args.num_sequences=2000 args.max_length=512 args.use_local_weights=True hydra.launcher.mem=40G hydra.launcher.gres=gpu:1 hydra.launcher.time=1440 logging.experiment=generate_samples args.model_name=Qwen2_5_0_5B,Qwen2_5_3B 

# Make Metadataset (original inefficient way)
#python3 run_bash.py --multirun command="python3 /scratch/mr7401/projects/meta_comp/generation/make_meta_dataset.py" args.model_name=OPT2_7B args.model_name2=OPT6_7B,OPT125M,Llama31_8B,OPT350M,GPT2

# Plotting
#python3 run_bash.py --multirun command="python3 /scratch/mr7401/projects/meta_comp/plotting/new_plots.py" logging.experiment=plots hydra.launcher.mem=10G hydra.launcher.time=30 hydra.launcher.gres=0

# Calculate log probs 
#python3 run_bash.py --multirun command="python3 /scratch/mr7401/projects/meta_comp/generation/calculate_log_likelihood.py" logging.experiment=loglikelihood args.n_subset=2000 args.data_dir=/scratch/mr7401/log_likelihoods_Truncation_Fixed_BS1 hydra.launcher.mem=20G hydra.launcher.gres=gpu:1 hydra.launcher.time=300 args.use_local_weights=True args.model_name=OPT125M,OPT350M  #OPT2_7B,OPT6_7B,Qwen2_5_0_5B,Qwen2_5_3B,Gemma2_2B,GPT2,GPT2Large,Llama31_8B,Llama32_3B

# Calculate perplexity
#python3 run_bash.py --multirun command="python3 /scratch/mr7401/projects/meta_comp/generation/calculate_perplexity.py" logging.experiment=perplexity args.data_dir=/scratch/mr7401/perplexity_test hydra.launcher.mem=20G hydra.launcher.gres=gpu:1 hydra.launcher.time=300 args.use_local_weights=True args.model_name=OPT125M,OPT350M,OPT2_7B,OPT6_7B,Qwen2_5_0_5B,Qwen2_5_3B,Gemma2_2B,GPT2,GPT2Large,Llama31_8B,Llama32_3B

# Generate samples from trained VAE models 
python3 run_bash.py --multirun command="python3 /scratch/mr7401/projects/meta_comp/generation/gen_vae_samples.py" args.model_name=vae_ldim_2 hydra.launcher.mem=1G hydra.launcher.gres=gpu:0 hydra.launcher.time=10 logging.experiment=generate_vae_samples