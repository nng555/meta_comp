import argparse
import uuid
from datetime import datetime
import jsonlines
import os 
from logger import Logger, add_log_args # LOGGING can delete if you don't want to use the logger class. 
import warnings
from datasets import load_dataset
import torch 
from torchvision.transforms.functional import to_pil_image

def generate_images(model_name: str = "ldim_2", MN: str = "1", num_generations: int = 10, epoch_to_use: int = 29, checkpoint_dir: str = "/meta_comp/checkpoints_loss_checks", save_dir:str = "/scratch/mr7401/meta_comp_data/generations", logger: Logger = None, test=False):
    if test: 
        warnings.warn("\n\n\n\n******** WARNING: gen_vae_samples.py is running in testing mode! This means the model is loaded, but not used for the actual generations. ********\n\n\n\n ")

    # Load a model 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Gen_VAE_Samples: Loading model {model_name} from {checkpoint_dir}, using device {device}", flush=True)
    checkpoint_path = f"{checkpoint_dir}/MN_{MN}/{model_name}_e{epoch_to_use}.pt"
    m1 = torch.load(checkpoint_path, map_location=device)
    m1.device = device # this changes the device in the model class
    m1.to(device).eval() # this moves weights 
    date = datetime.now().date().isoformat()
    
    if test:
        num_generations = 5
        batch_size = 1
        output_file = f"{save_dir}/MN_{MN}/{model_name}/generations_test.jsonl"
        save_dir = f"{save_dir}/MN_{MN}/{model_name}/samples_test/"
        os.makedirs(save_dir, exist_ok=True)
    else: 
        batch_size = 1
        output_file = f"{save_dir}/MN_{MN}/{model_name}/generations.jsonl"
        save_dir = f"{save_dir}/MN_{MN}/{model_name}/samples/"
        os.makedirs(save_dir, exist_ok=True)
    
    print(f"Gen_VAE_Samples: Generating {num_generations} generations.", flush=True)
    # Make a dataset file to dump 
    with jsonlines.open(output_file, mode='w', flush = True) as writer:

        try: 
            # Generate images and write to files
            for i in range(num_generations):
                if logger is not None:
                    if i % 10 == 0:
                        logger.log({"Progress": i})
                m1_samples, m1_m1_ll = m1.sample(batch_size, device) # images, log_likelihood
                if device.type == 'cuda':
                    m1_samples = m1_samples.detach().cpu()
                    m1_m1_ll = m1_m1_ll.detach().cpu()

                for j, (sample, _ ) in enumerate(zip(m1_samples, m1_m1_ll)):
                    # Save image to file 
                    id = str(uuid.uuid4())
                    location = os.path.join(save_dir, f"{id}.png")
                    img = to_pil_image(sample)
                    img.save(location)
                    # Add a metadata entry to the jsonl file 
                    data_entry = {
                        "id": id,
                        "location": location,
                        "model": model_name,
                        "date_generated": date, 
                        "model_checkpoint": checkpoint_path,
                    }
                    # save to file
                    writer.write(data_entry)
                
        except Exception as e:
            print(f"Gen_Model_Samples: Generation Error: {e}", flush = True) 
        
    print("Completed Saving Generations")
    return 

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected in the form of: True/False, T/F, Yes/No, Y/N, or 1/0')

############## USING THIS SCRIPT ##################
# This script relies on 
# 1) the VAE class in models/vae.py. 
# 2) the logger class in logger.py. You can delete the logger reliance using the LOGGING comments below and the import.  

# Example use from command line:
# python3 gen_vae_samples.py --model_name "ldim_2" --num_generations 100

# Example use from slurm launcher: 
# run_bash.py is a script that launches the slurm job using the config.yaml file and any overrides you specify. 
# launch_playground.py has example calls to run_bash.py script. Roughly, it looks something like:
#       'python run_bash.py --multirun command="python3 /scratch/mr7401/projects/meta_comp/generation/gen_vae_samples.py" args.model_name=vae_ldim_2 hydra.launcher.mem=1G hydra.launcher.gres=gpu:0 hydra.launcher.time=10 logging.experiment=generate_vae_samples'

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate images using a VAE model")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use for generation')
    parser.add_argument('--MN', type=str, required=True, help='MN used for the model')
    parser.add_argument('--epoch_to_use', type=int, required=True, help='Epoch to use for the model')
    parser.add_argument('--num_generations', type=int, required=True, help='Number of images to generate')
    parser.add_argument('--checkpoint_dir', type=str, required=False, default = "/scratch/mr7401/projects/meta_comp/checkpoints_loss_checks", help='Directory containing model checkpoints')
    parser.add_argument('--save_dir', type=str, required=False, default = "/scratch/mr7401/meta_comp_data/vaes/generations", help='Directory to save the generated images')
    parser.add_argument('--test', type=str2bool, required=False, default=False, help='If True, run this script in testing mode (testing loading the models and saving, without generation)')
    parser = add_log_args(parser)

    args, unknown_args = parser.parse_known_args()
  
    # LOGGING Set up logging. Can delete this and pass logger = None if you don't want to use the logger class.
    logging_name = f"{args.model_name}_MN_{args.MN}_num{args.num_generations}_epoch_{args.epoch_to_use}"
    logger = Logger(group = "Generate_VAE_Samples", logging_name=logging_name, **vars(args))
    # LOGGING Uncomment this line to disable progress logging in wandb. Slurm will still capture the output and any errors / print outs. 
    # logger = None 

    # Call function 
    generate_images(model_name = args.model_name, MN = args.MN, epoch_to_use=args.epoch_to_use, num_generations = args.num_generations, checkpoint_dir = args.checkpoint_dir, save_dir= args.save_dir, logger = logger,test=args.test)
    print(f"Gen_VAE_Samples: Generated {args.num_generations} sequences using model {args.model_name} and saved in {args.save_dir}", flush = True)
    
    logger.finish()
    