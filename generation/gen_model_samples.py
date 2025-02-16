import argparse
import uuid
from datetime import datetime
import jsonlines
import os 
from logger import Logger, add_log_args
from models.llms import get_model
import warnings

def generate_sequences(model_name, num_sequences, max_length, output_file, use_local_weights, logger, test=False):
    if test: 
        warnings.warn("\n\n\n\n******** WARNING: gen_model_samples.py is running in testing mode! This means the model is loaded, but not used for the actual generations. ********\n\n\n\n ")

    # Load a model 
    model = get_model(model_name=model_name, use_local_weights=use_local_weights)
    date = datetime.now().date().isoformat()

    print(f"Gen_Model_Samples: Starting to Generate Samples", flush=True)
    # Make a dataset file to dump 
    with jsonlines.open(output_file, mode='w', flush = True) as writer:
        try: 
            # Generate sequences and write iteratively to file 
            for i in range(num_sequences):
                
                if i % 10 == 0:
                    logger.log({"Progress": i})
                  
                # generate
                if test:
                    sequence = "testing sequence"
                else: 
                    sequence = model.generate(prompt="", max_length=max_length)
                
                data_entry = {
                    "id": str(uuid.uuid4()),
                    "sequence": sequence,
                    "model": model_name,
                    "date_generated": date
                }
                # save to file
                writer.write(data_entry)
        except Exception as e:
            print(f"Gen_Model_Samples: Error: {e}", flush = True) 
        finally: 
            writer.flush()
    
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
    parser.add_argument('--num_sequences', type=int, required=True, help='Number of sequences to generate')
    parser.add_argument('--max_length', type=int, required=False, default = 512, help='Maximum token length of each generated sequence')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory to make the dataset folder in. Result will be data_dir/{new_folder_with_meta_info}/{actual_data_file.jsonl}')
    parser.add_argument('--use_local_weights', type=str2bool, required=False, default=False, help='If True and local_path is passed (or default exists), we use the local version of stored weights rather than the HF API')
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
        generate_sequences(args.model_name, 10, 50, output_file, args.use_local_weights, logger, test=True)
        print(f"Gen_Model_Samples: Generated {args.num_sequences} sequences using model {args.model_name} and saved to {output_file}", flush = True)
       
    else: 
        output_file = f"{args.data_dir}/{args.model_name}/{args.num_sequences}_{args.max_length}_generations.jsonl"
        generate_sequences(args.model_name, args.num_sequences, args.max_length, output_file, args.use_local_weights, logger)
        print(f"Gen_Model_Samples: Generated {args.num_sequences} sequences using model {args.model_name} and saved to {output_file}", flush = True)
    
    logger.finish()
    