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

def mean(lst): 
    return sum(lst) / len(lst)

def compute_kl_difference(batch1:list[str], batch2:list[str], model1:Model, model2:Model):
    m1_m1_ll = model1.to_tokens_and_logprobs(batch1)
    #print(f"M1M1_LL = {m1_m1_ll}")
    m1_m2_ll = model1.to_tokens_and_logprobs(batch2)
    m2_m1_ll = model2.to_tokens_and_logprobs(batch1)
    m2_m2_ll = model2.to_tokens_and_logprobs(batch2)
    kl_diff = -(mean(m1_m1_ll) + mean(m1_m2_ll) + mean(m2_m1_ll) + mean(m2_m2_ll))
    return kl_diff


def generate_meta_dataset(model_name, model_name2, dataset1, dataset2, output_file, use_local_weights, logger, test = False):
    # Load both models  
    print("Generate_MetaDataset: Loading Models", flush = True)
    model1 = get_model(model_name=model_name, use_local_weights=use_local_weights)
    model2 = get_model(model_name=model_name2, use_local_weights=use_local_weights)
    
    # Load both generation datasets 
    print("Generate_MetaDataset: Loading Generation Datasets", flush = True)
    d1 = load_dataset("json", data_files=dataset1, split="train", streaming=False)
    d2 = load_dataset("json", data_files=dataset2, split="train", streaming=False)
    
    # Remove any metadata columns for efficiency
    print("Generate_MetaDataset: Selecting Columns to Use", flush = True)
    cols_to_remove = [x for x in d1.column_names if x not in ["id", "sequence"]]
    if len(cols_to_remove) > 0:
        d1 = d1.remove_columns(cols_to_remove)
        d2 = d2.remove_columns(cols_to_remove)

    # Make dataloaders
    print("Generate_MetaDataset: Making DataLoaders", flush = True)
    dl1 = iter(DataLoader(d1, batch_size=4, shuffle=True))
    dl2 = iter(DataLoader(d2, batch_size=4, shuffle=True))

    print("Generate_MetaDataset: Starting Generations", flush = True)
    with jsonlines.open(output_file, mode='w', flush = True) as writer:
        # Load all pairs of generations from models, and compute KL difference 
   
        for i in range(2000):
            if i % 10 == 0 and i != 0:
                    logger.log({"Progress": i})
            
            batch1 = next(dl1)
            batch2 = next(dl2)
               
            #print(f"Batch 1 = {batch1}")
            #print(f"Batch 2 = {batch2}")
            metric = compute_kl_difference(batch1["sequence"], batch2["sequence"], model1, model2)
            #print(f"KL Difference {metric}", flush = True)
            # Make a dataset item and add to list
            data_entry = {
                "id": str(uuid.uuid4()),
                "model1": model1.huggingface_id,
                "model2": model2.huggingface_id,
                "gen1": batch1, 
                "gen2": batch2,
                "metric": metric
                
            }
            writer.write(data_entry)
            
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
    parser.add_argument('--model_name2', type=str, required=True, help='Second Model used to compare')
    parser.add_argument('--data_dir', type=str, required=False, default="/scratch/mr7401/meta_datasets", help='Path to directory to save metadata in')
    parser.add_argument('--use_local_weights', type=str2bool, required=False, default=False, help='If True, uses local version of stored weights rather than the HF API')
    parser.add_argument('--test', type=str2bool, required=False, default=False, help='If True, run this script in testing mode (manual override of variables)')
    parser=add_log_args(parser)

    args, unknown_args = parser.parse_known_args()
    print("Args given to make_meta_dataset.py:", flush = True)
    print(args, flush = True)

    # Make logger 
    logging_name = f"{args.model_name}_{args.model_name2}"
    gen_dataset1 = f"/scratch/mr7401/generations_no_prompts/{args.model_name}/10000_512_generations.jsonl"
    gen_dataset2 = f"/scratch/mr7401/generations_no_prompts/{args.model_name2}/10000_512_generations.jsonl"
  
    # Make logger and saving folders
    logger = Logger(group = "MetaDataset", logging_name=logging_name, **vars(args))
    
    os.makedirs(f"{args.data_dir}", exist_ok=True)
    os.makedirs(f"{args.data_dir}/{args.model_name}_vs_{args.model_name2}", exist_ok = True)
    output_file = f"{args.data_dir}/{args.model_name}_vs_{args.model_name2}/meta_dataset.jsonl"
    
    if os.path.exists(output_file): 
        new_output_file = f"{args.data_dir}/{args.model_name}_vs_{args.model_name2}/meta_dataset_NEW.jsonl"
        warnings.warn(f"\n\n\n\nWARNING: Output File {output_file} already exists. Saving to {new_output_file} instead.")
        output_file = new_output_file
        

    generate_meta_dataset(args.model_name, args.model_name2, gen_dataset1, gen_dataset2, output_file, args.use_local_weights, logger)

    print(f"Generated metadataset between {args.model_name} and {args.model_name2} and saved to {output_file}", flush = True)
    logger.finish()
    