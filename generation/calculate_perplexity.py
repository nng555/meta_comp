import argparse
import uuid
import jsonlines
import os
# os.sys.path.append('/scratch/mr7401/projects/meta_comp/')
os.sys.path.append('/Users/mr7401/Projects/meta_comp/')
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
from logger import Logger, add_log_args
from models.llms import get_model
from datasets import load_dataset
from torch.utils.data import DataLoader
from models.llms import Model
import warnings
from torch.utils.data import Dataset
import pandas as pd

# Define str2bool function
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
def calculate_perplexity(model_name, output_file, use_local_weights, logger, test=False, n_subset=None, verbose=False, context_window_limit=None, stride_denominator=2):
    if verbose or test:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
    
    # Load Model
    model = get_model(model_name=model_name, use_local_weights=use_local_weights)
    
    print(f"Calculate_Perplexity: Opening output file and starting to load toy dataset...", flush=True)
    with jsonlines.open(output_file, mode='a', flush=True) as writer:
        
        # Create a dataset from nyt_articles_with_text.csv

        class NYTArticlesDataset(Dataset):
            def __init__(self, csv_file, n_subset=None):
                df = pd.read_csv(csv_file)
                # Filter out rows where 'text' column has empty strings
                df = df[df['text'].str.strip() != ""]
                # Filter out rows where 'web_url' column has empty strings
                df = df[df['web_url'].str.strip() != ""]
                # Filter out na in web_url or text 
                df = df.dropna(subset=['web_url', 'text'])
                
                self.data = df['text'].tolist()
                self.ids = df['web_url'].tolist()
                
                if n_subset is not None:
                    self.data = self.data[:n_subset]
                    self.ids = self.ids[:n_subset]
                
            def __len__(self):
                return len(self.data)
                
            def __getitem__(self, idx):
                return {"id": self.ids[idx], "sequence": self.data[idx]}
            
        dataset = NYTArticlesDataset(csv_file="nyt_articles_with_text.csv", n_subset=n_subset)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        total_complete = 0
        
        print(f"Calculate_Perplexity: Starting to calculate perplexity for the toy dataset...", flush=True)
        for batch in dataloader:
  
            if len(batch["sequence"]) > 0:
                if verbose:
                    print(f"    Batch size = {len(batch['sequence'])}, \n Batch = {batch}", flush=True)
                try:
                    perplexities = model.calculate_perplexity(batch["sequence"][0], verbose=verbose, context_window_limit = context_window_limit, stride_denominator=stride_denominator) 
                    
                    if verbose:
                        print(f"    Perplexities: {perplexities}", flush=True)
                    
                    # Make dataset items (N = batch_size) and write them to a file
                    for i in range(len(batch["sequence"])):
                        if i < len(batch["id"]):
                            total_complete += 1
                            data_entry = {
                                "generation_id": batch["id"][i],
                                # "generation": batch["sequence"][i],
                                f"{model.name}_perplexity": perplexities[i].item()
                            }
                            writer.write(data_entry)
                        else:
                            warnings.warn(f"Warning: A generation after {total_complete} calculations did not have a batch id, a batch sequence, and a perplexity.")
                    
                    logger.log({"Total Complete": total_complete})
                except Exception as e:
                    print(f"    Calculating the perplexity of the following batch failed: \n Batch: {batch['sequence']}\nSpecific Error: {e} \nSkipping.", flush=True)
        
        print(f"    Completed all batches")
    
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate Perplexity for a Model")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use for perplexity calculation')
    parser.add_argument('--data_dir', type=str, required=False, default="/scratch/mr7401/perplexity_results", help='Path to directory to save data in')
    parser.add_argument('--use_local_weights', type=str2bool, required=False, default=False, help='If True, uses local version of stored weights rather than the HF API')
    parser.add_argument('--n_subset', type=int, required=False, default=None, help='If provided, only uses the first n_subset samples from the dataset')
    parser.add_argument('--test', type=str2bool, required=False, default=False, help='If True, run this script in testing mode (manual override of variables)')
    parser.add_argument('--context_window_limit', type=int, required=False, default=None, help='Maximum context window size for the model')
    parser.add_argument('--stride_denominator', type=int, required=False, default=2, help='Denominator to calculate stride size (stride = context_window_limit / stride_denominator)')
    parser.add_argument('--verbose', action='store_true', help='If set, enables verbose output')
    parser = add_log_args(parser)

    args, unknown_args = parser.parse_known_args()
    print("Args given to calculate_perplexity.py:", flush=True)
    print(args, flush=True)

    # Make logger 
    logging_name = f"{args.model_name}"
  
    # Make logger and saving folders
    logger = Logger(group="PerplexityCalculation", logging_name=logging_name, **vars(args))
    
    os.makedirs(f"{args.data_dir}", exist_ok=True)
    os.makedirs(f"{args.data_dir}/{args.model_name}", exist_ok=True)
    if args.n_subset is not None:
        output_file = f"{args.data_dir}/{args.model_name}/perplexity_{args.n_subset}.jsonl"
    else: 
        output_file = f"{args.data_dir}/{args.model_name}/perplexity.jsonl"
    
    if os.path.exists(output_file): 
        new_output_file = f"{args.data_dir}/{args.model_name}/perplexity_NEW.jsonl"
        warnings.warn(f"\n\n\n\nWARNING: Output File {output_file} already exists. Saving to {new_output_file} instead.")
        output_file = new_output_file
    
    calculate_perplexity(
        model_name=args.model_name, 
        output_file=output_file, 
        use_local_weights=args.use_local_weights, 
        logger=logger, 
        n_subset=args.n_subset, 
        test=args.test, 
        context_window_limit=args.context_window_limit,
        stride_denominator=args.stride_denominator,
        verbose=args.verbose
    )

    print(f"Calculated Perplexity for All Datasets Under {args.model_name} and saved to {output_file}", flush=True)
    logger.finish()
    
