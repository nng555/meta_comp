import argparse
import uuid
from datetime import datetime
import jsonlines
import os 
from logger import Logger, add_log_args
from models.llms import get_model
import warnings
from datasets import load_dataset

def generate_sequences(model_name, num_sequences, max_length, output_file, use_local_weights, logger, num_return_sequences_per_generation=1, test=False, prompting_data=""):
    if test: 
        warnings.warn("\n\n\n\n******** WARNING: gen_model_samples.py is running in testing mode! This means the model is loaded, but not used for the actual generations. ********\n\n\n\n ")

    # Load a model 
    model = get_model(model_name=model_name, use_local_weights=use_local_weights)
    date = datetime.now().date().isoformat()

    if prompting_data:
        ds = load_dataset("cerebras/SlimPajama-627B", split = "train")
        ds = ds.select(range(2000))

    if num_return_sequences_per_generation > 1: 
        num_sequences = num_sequences // num_return_sequences_per_generation
    
    print(f"Gen_Model_Samples: Generating {num_sequences} generations, with {num_return_sequences_per_generation} samples per generation.", flush=True)
    # Make a dataset file to dump 
    with jsonlines.open(output_file, mode='w', flush = True) as writer:

        first_five_generations = []
        try: 
            # Generate sequences and write iteratively to file 
            for i in range(num_sequences):

                if prompting_data:
                    # get the first 5 words of each prompt
                    batch_size = 5
                    n_words = 5
                    batch_full_prompts = ds.select(range(i*batch_size,i*batch_size+batch_size))['text'] 
                    prompts =[" ".join(s.split()[:n_words]) for s in batch_full_prompts] # list of strings of size batch_size
                else:
                    prompts = None

                if i % 10 == 0:
                    logger.log({"Progress": i})
                  
                # generate
                if test:
                    sequences = ["testing sequence"]
                else: 
                    sequences = model.generate(prompts=prompts, max_length=max_length, num_return_sequences=num_return_sequences_per_generation)
                
                for sequence in sequences:
                    data_entry = {
                        "id": str(uuid.uuid4()),
                        "sequence": sequence,
                        "model": model_name,
                        "date_generated": date
                    } 
                    # save to file
                    writer.write(data_entry)
                    if i in [0,1,2,3,4]: 
                        first_five_generations.append(sequence)
                    if i ==4: 
                        # Make sure we are not generating the same example over and over :)
                        num_unique_gen = len(list(set(first_five_generations)))
                        if num_unique_gen < 5 and not test:
                            print(f"First 5 Generations:\n\n{first_five_generations}")
                            raise ValueError("Identical Generations Detected in the First 5 Generations! Cancelling Job.")
                
        except Exception as e:
            print(f"Gen_Model_Samples: Error: {e}", flush = True) 
        
    print("Completed Saving Generations")
    return 

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected in the form of: True/False, T/F, Yes/No, Y/N, or 1/0')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate text sequences")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use for generation')
    parser.add_argument('--num_sequences', type=int, required=False, default = 10, help='Number of sequences to generate')
    parser.add_argument('--max_length', type=int, required=False, default = 512, help='Maximum token length of each generated sequence')
    parser.add_argument('--data_dir', type=str, required=False, default = "/scratch/mr7401/generations_no_prompts", help='Directory to make the dataset folder in. Result will be data_dir/{model_name}/{actual_data_file.jsonl}')
    parser.add_argument('--use_local_weights', type=str2bool, required=False, default=False, help='If True and local_path is passed (or default exists), we use the local version of stored weights rather than the HF API')
    parser.add_argument('--prompting_data', type=str, required=False, default=None, help='Name of a dataset to use')
    parser.add_argument('--local_path', type=str, required=False, help='If use_local_weights is set to True, the model uses the checkpoint at this path rather than the HF API')
    parser.add_argument('--test', type=str2bool, required=False, default=False, help='If True, run this script in testing mode (testing loading the models and saving, without generation)')
    parser=add_log_args(parser)

    args, unknown_args = parser.parse_known_args()
    #TODO for some reason the date for the run_name is not created correctly here. Fix! 
   
    logging_name = f"{args.model_name}_num{args.num_sequences}_length{args.max_length}"
  
    # Make logger and saving folders
    logger = Logger(group = "Generate_Samples_GPU", logging_name=logging_name, **vars(args))
    os.makedirs(f"{args.data_dir}/{args.model_name}", exist_ok = True)
   
    if args.test:
        # Generates 10 sequences with max size 50 to test file
        output_file=f"{args.data_dir}/{args.model_name}/{args.num_sequences}_{args.max_length}_generations_test.jsonl"
        generate_sequences(model_name=args.model_name, num_sequences=10, max_length=50, output_file=output_file, use_local_weights=args.use_local_weights, logger=logger, num_return_sequences_per_generation=1, test=True, prompting_data=args.prompting_data)
        print(f"Gen_Model_Samples: Generated {args.num_sequences} sequences using model {args.model_name} and saved to {output_file}", flush = True)
       
    else: 
        output_file = f"{args.data_dir}/{args.model_name}/{args.num_sequences}_{args.max_length}_generations.jsonl"
        generate_sequences(model_name=args.model_name, num_sequences=args.num_sequences, max_length=args.max_length, output_file=output_file, use_local_weights=args.use_local_weights, num_return_sequences_per_generation = 4, logger = logger, prompting_data=args.prompting_data)
        print(f"Gen_Model_Samples: Generated {args.num_sequences} sequences using model {args.model_name} and saved to {output_file}", flush = True)
    
    logger.finish()
    