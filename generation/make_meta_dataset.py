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


def mean(lst): 
    return sum(lst) / len(lst)

def compute_kl_difference(batch1:list[str], batch2:list[str], model1:Model, model2:Model):
    m1_m1_ll = model1.to_tokens_and_logprobs(batch1)
    m1_m2_ll = model1.to_tokens_and_logprobs(batch2)
    m2_m1_ll = model2.to_tokens_and_logprobs(batch1)
    m2_m2_ll = model2.to_tokens_and_logprobs(batch2)
    kl_diff = -(mean(m1_m1_ll) + mean(m1_m2_ll) + mean(m2_m1_ll) + mean(m2_m2_ll))
    return kl_diff


def generate_meta_dataset(model_name, model_name2, dataset1, dataset2, output_file, use_local_weights, logger, test = False):
    # Load both models  
    model1 = get_model(model_name=model_name, use_local_weights=use_local_weights)
    model2 = get_model(model_name=model_name2, use_local_weights=use_local_weights)
    
    # Load both generation datasets 
    d1 = load_dataset("json", data_files=dataset1, split="train", streaming=False)
    d2 = load_dataset("json", data_files=dataset2, split="train", streaming=False)
    
    # Remove any metadata columns for efficiency
    cols_to_remove = [x for x in d1.column_names if x not in ["id", "sequence"]]
    if len(cols_to_remove) > 0:
        d1 = d1.remove_columns(cols_to_remove)
        d2 = d2.remove_columns(cols_to_remove)

    # Make datalaoders
    dl1 = DataLoader(d1, batch_size=4, shuffle=False)
    dl2 = DataLoader(d2, batch_size=4, shuffle=False)

    with jsonlines.open(output_file, mode='w', flush = True) as writer:
        # Load all pairs of generations from models, and compute KL difference 
        i = 0
        for batch1 in dl1:
            data_entries = []
            for batch2 in dl2:
                i=i+1
                if i % 10 == 0 and i != 0:
                    logger.log({"Progress": i})
        
                metric = compute_kl_difference(batch1, batch2, model1, model2)
                # Make a dataset item and add to list
                data_entry = {
                    "id": str(uuid.uuid4()),
                    "model1": model1.huggingface_id,
                    "model2": model2.huggingface_id,
                    "gen1": batch1, 
                    "gen2": batch2,
                    "metric": metric
                    
                }
                data_entries.append(data_entry)
            
            # After each iteration of dataset 1, write to file and reset iterator 
            writer.write_all(data_entries)
            dl2.dataset.__iter__()  
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
    parser.add_argument('--gen_dataset1', type=str, required=True, help='Path to generation file from model 1')
    parser.add_argument('--gen_dataset2', type=str, required=True, help='Path to generation file from model 2')
    parser.add_argument('--use_local_weights', type=str2bool, required=False, default=False, help='If True, uses local version of stored weights rather than the HF API')
    parser.add_argument('--test', type=str2bool, required=False, default=False, help='If True, run this script in testing mode (manual override of variables)')
    parser=add_log_args(parser)

    args, unknown_args = parser.parse_known_args()
    print("Args given to make_meta_dataset.py:")
    print(args)

    # Make logger 
    
    logging_name = f"{args.model_name}_{args.model_name2}"
  
    # Make logger and saving folders
    logger = Logger(group = "MetaDataset", logging_name=logging_name, **vars(args))

    os.makedirs(f"{args.data_dir}/{args.model_name}", exist_ok = True)
    output_file = f"{args.data_dir}/{args.model_name}_{args.model_name2}/meta_dataset.jsonl"

    generate_meta_dataset(args.model_name, args.model_name2, args.dataset1, args.dataset2, output_file, args.use_local_weights, logger)

    print(f"Generated metadataset between {args.model_name} and {args.model_name2} and saved to {output_file}")
    logger.finish()
    