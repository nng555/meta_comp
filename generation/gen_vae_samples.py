import argparse
import uuid
from datetime import datetime
import jsonlines
import os 
from logger import Logger, add_log_args
from models.llms import get_model
import warnings
from datasets import load_dataset
import torch 
from torchvision.transforms.functional import to_pil_image

def generate_images(model_name, num_generations, checkpoint_dir, save_dir, logger, test=False):
    if test: 
        warnings.warn("\n\n\n\n******** WARNING: gen_model_samples.py is running in testing mode! This means the model is loaded, but not used for the actual generations. ********\n\n\n\n ")

    # Load a model 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m1 = torch.load(f"{checkpoint_dir}/{model_name}_e19.pt", weights_only=False)
    m1.to(device).eval()
    date = datetime.now().date().isoformat()

    output_file = f"{save_dir}/{model_name}/generations.jsonl"
    save_dir = f"{save_dir}/{model_name}/samples/"
    os.makedirs(save_dir, exist_ok=True)
    batch_size = 1
    print(f"Gen_VAE_Samples: Generating {num_generations} generations.", flush=True)
    # Make a dataset file to dump 
    with jsonlines.open(output_file, mode='w', flush = True) as writer:

        try: 
            # Generate images and write to files
            for i in range(num_generations):
                if i % 10 == 0:
                    logger.log({"Progress": i})
                  
                m1_samples, m1_m1_ll = m1.sample(batch_size, device) # images, log_likelihood
                m1_samples = m1_samples.detach().cpu()
                m1_samples = m1_m1_ll.detach().cpu()

                for j, (sample, log_likelihood) in enumerate(zip(m1_samples, m1_m1_ll)):
                    # Save image to file 
                    id - str(uuid.uuid4())
                    location = os.path.join(save_dir, f"{id}.png")
                    img = to_pil_image(sample)
                    img.save(location)
            
                    # Add a metadata entry to the jsonl file 
                    data_entry = {
                        "id": id,
                        "location": location,
                        "model": model_name,
                        "date_generated": date, 
                        f"{model_name}_ll": log_likelihood.item()
                    }
                    # save to file
                    writer.write(data_entry)
                
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

    parser = argparse.ArgumentParser(description="Generate images using a VAE model")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use for generation')
    parser.add_argument('--num_generations', type=int, required=False, default = 2000, help='Number of images to generate')
    parser.add_argument('--checkpoint_dir', type=str, required=False, default = "/scratch/mr7401/projects/meta_comp/checkpoints/", help='Directory containing model checkpoints')
    parser.add_argument('--save_dir', type=str, required=False, default = "/scratch/mr7401/vae_generations/", help='Directory to save the generated images')
    parser.add_argument('--test', type=str2bool, required=False, default=False, help='If True, run this script in testing mode (testing loading the models and saving, without generation)')
    parser = add_log_args(parser)

    args, unknown_args = parser.parse_known_args()
  
    # Set up logging
    logging_name = f"{args.model_name}_num{args.num_generations}"
    logger = Logger(group = "Generate_VAE_Samples_GPU", logging_name=logging_name, **vars(args))
    os.makedirs(f"{args.data_dir}/{args.model_name}", exist_ok = True)
   
    # Set up the output saving
    generate_images(model_name = args.model_name, num_generations = args.num_generations, checkpoint_dir = args.checkpoint_dir, save_dir= args.save_dir, logger = logger, test=args.test)
    print(f"Gen_VAE_Samples: Generated {args.num_generations} sequences using model {args.model_name} and saved in {args.save_dir}", flush = True)
    
    logger.finish()
    