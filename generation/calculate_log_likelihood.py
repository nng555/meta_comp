import argparse
import uuid
import jsonlines
import os
os.sys.path.append('/scratch/mr7401/projects/meta_comp/')
from logger import Logger, add_log_args
from models.llms import get_model
from datasets import load_dataset
from torch.utils.data import DataLoader
from models.llms import Model
import warnings

"""
Purpose: The goal of this script is to improve efficiency by calculating the log likelihood of all generations under 1 model at a time. 

Usage:
    Run this script from the command line with the required arguments:
    --model_name: Name of the model to use for generation (check models/llms.py file)
    --data_dir: Path to directory to save the results to (NOT the file, just the path to the general results folder.)
    --use_local_weights: If True, uses local version of stored weights rather than the HF API.
Example:
    python calculate_log_likelihood.py --model_name GPT2 --data_dir /path/to/data --use_local_weights False --test False
"""

def calculate_log_likelihood(model_name, output_file, use_local_weights, logger, test = False, n_subset = None, verbose = False):
    
    # Load Model 
    model1 = get_model(model_name=model_name, use_local_weights=use_local_weights)
    
    print(f"Calculate_Log_Likelihood: Opening output file and starting to load generation datasets...", flush = True)
    with jsonlines.open(output_file, mode='w', flush = True) as writer:
        
        # For all model generations, including model 1, iterate the generations dataset and calculate the log likelihood 
        # of the samples under model 1. Write all into to a file with the ID. 

        total_complete = 0
        
        for model_name2 in ["OPT125M", "OPT350M", "OPT2_7B", "OPT6_7B", "GPT2", "GPT2Large", "Gemma2_2B"]:
            print(f"Calculate_Log_Likelihood: Starting to calculate LL for {model_name2}'s generations...", flush = True)
            per_model_complete = 0
            
            # Load generation dataset
            d2 = load_dataset("json", data_files=f"/scratch/mr7401/generations_no_prompts/{model_name2}/10000_512_generations.jsonl", split="train", streaming=False)
            
            # Subset if requested 
            if n_subset is not None:
                d2 = d2.select(range(n_subset))
            
            # Remove extraneous metadata columns
            cols_to_remove = [x for x in d2.column_names if x not in ["id", "sequence"]]
            if len(cols_to_remove) > 0:
                d2 = d2.remove_columns(cols_to_remove)

            # Make dataloader
            batch_size = 4
            dl2 = DataLoader(d2, batch_size=batch_size, shuffle=False)

            # Iterate dataloader, calculating log prob under model 1

            for batch in dl2:
                if verbose: 
                    print(f"    Batch size = {len(batch['sequence'])}, \n Batch = {batch}", flush=True)
                    print(f"    Example batch sequence: {batch["sequence"][0]}", flush = True)
                
                gen_log_likelihood = model1.to_tokens_and_logprobs(batch["sequence"], verbose = False) # This will produce a list with shape [batch_size]

                if verbose: 
                    print(f"    Gen Log Likelihood: {gen_log_likelihood}", flush=True)
                
                # Make dataset items (N = batch_size) and write them to a file
        
                for i in range(batch_size):
                    total_complete = total_complete + 1
                    per_model_complete = per_model_complete + 1
                    data_entry = {
                        "gen_source_model": model_name2,
                        "generation": batch["sequence"][i], 
                        "generation_id": batch["id"][i],
                        f"{model1.name}_ll": gen_log_likelihood[i]                
                    }
                    writer.write(data_entry)
                
                logger.log({f"{model_name2}_Complete": per_model_complete})
                logger.log({"Total Complete": total_complete})
            
            print(f"Completed batches")
                                  
    return  

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected in the form of: True/False, T/F, Yes/No, Y/N, or 1/0')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Make MetaDataset Between 2 Models")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use for generation')
    parser.add_argument('--data_dir', type=str, required=False, default="/scratch/mr7401/log_likelihoods", help='Path to directory to save data in')
    parser.add_argument('--use_local_weights', type=str2bool, required=False, default=False, help='If True, uses local version of stored weights rather than the HF API')
    parser.add_argument('--n_subset', type=int, required=False, default=None, help='If provided, only uses the first n_subset samples from the generation dataset')
    parser.add_argument('--test', type=str2bool, required=False, default=False, help='If True, run this script in testing mode (manual override of variables)')
    
    parser=add_log_args(parser)

    args, unknown_args = parser.parse_known_args()
    print("Args given to calculate_log_likelihood.py:", flush = True)
    print(args, flush = True)

    # Make logger 
    logging_name = f"{args.model_name}"
  
    # Make logger and saving folders
    logger = Logger(group = "LogLikelihood", logging_name=logging_name, **vars(args))
    
    os.makedirs(f"{args.data_dir}", exist_ok=True)
    os.makedirs(f"{args.data_dir}/{args.model_name}", exist_ok = True)
    if args.n_subset is not None:
        output_file = f"{args.data_dir}/{args.model_name}/log_likelihood_{args.n_subset}.jsonl"
    else: 
        output_file = f"{args.data_dir}/{args.model_name}/log_likelihood.jsonl"
    
    if os.path.exists(output_file): 
        new_output_file = f"{args.data_dir}/{args.model_name}/log_likelihood_NEW.jsonl"
        warnings.warn(f"\n\n\n\nWARNING: Output File {output_file} already exists. Saving to {new_output_file} instead.")
        output_file = new_output_file
        
    calculate_log_likelihood(model_name=args.model_name, output_file= output_file, use_local_weights=args.use_local_weights, logger=logger, n_subset=args.n_subset, verbose = False)

    print(f"Calculated Log Likelihood for All Datasets Under {args.model_name} and saved to {output_file}", flush = True)
    logger.finish()
    
