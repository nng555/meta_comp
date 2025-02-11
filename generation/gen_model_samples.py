from transformers import pipeline, set_seed
import argparse
import uuid
from datetime import datetime
import json
import jsonlines
import os 
from logger import Logger, add_log_args
from models.llms import get_model

def generate_sequences(model_name, num_sequences, max_length, output_file, use_local_weights, logger):
    print(f"Writing to {output_file}")

    # ## Load a model and set the seed 
    model = get_model(model_name, use_local_weights=use_local_weights)
    date = datetime.now().date().isoformat()

    print(f"Running model on {model.device}")
 
    # Make a dataset file to dump to 
    with jsonlines.open(output_file, mode='w') as writer:

        # Generate sequences and write iteratively to file 
        for i in range(num_sequences):
            
            if i % 10 == 0 and i != 0:
                logger.log({"Progress": i})
        
            # generate
            sequence = model.generate(prompt="", max_length=max_length)
            data_entry = {
                "id": str(uuid.uuid4()),
                "sequence": sequence,
                "model": model_name,
                "date_generated": date
            }
            # save to file
            writer.write(data_entry)
    
    print("Completed Generations")
    return 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate text sequences")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use for generation')
    parser.add_argument('--num_sequences', type=int, required=True, help='Number of sequences to generate')
    parser.add_argument('--max_length', type=int, required=False, default = 512, help='Maximum length of each generated sequence')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory to make the dataset folder in')
    parser.add_argument('--use_local_weights', type=bool, required=False, default= True, help='If True, uses local version of stored weights rather than the HF API')
    parser=add_log_args(parser)

    args, unknown_args = parser.parse_known_args()
    
    # Make logger 
    logger = Logger(**vars(args))

    os.makedirs(f"{args.data_dir}/{args.model_name}", exist_ok = True)

    output_file = f"{args.data_dir}/{args.model_name}/{args.num_sequences}_{args.max_length}_generations.jsonl"
    
    generate_sequences(args.model_name, args.num_sequences, args.max_length, output_file, args.use_local_weights, logger)

    print(f"Generated {args.num_sequences} sequences using model {args.model_name} and saved to {output_file}")
    logger.finish()