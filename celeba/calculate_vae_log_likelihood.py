import argparse
import uuid
import jsonlines
import os
os.sys.path.append('/scratch/mr7401/projects/meta_comp/')
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
from logger import Logger, add_log_args
from models.llms import get_model
from datasets import load_dataset
from torch.utils.data import DataLoader
from models.llms import Model
import warnings
from PIL import Image
import torch
from torchvision import transforms

def calculate_log_likelihood(model_name, output_file, logger, test = False, verbose = False):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if verbose or test:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
    
    # Load Model 
    model1 = torch.load(f"/scratch/mr7401/projects/meta_comp/checkpoints/{model_name}_e19.pt", map_location=device)
    model1.device = device # this changes the device in the model class
    model1.to(device).eval() # this moves weights 
    print(f"Calculate_Log_Likelihood: Opening output file and starting to load generation datasets...", flush = True)
    with jsonlines.open(output_file, mode='a', flush = True) as writer:
        
        # For all model generations, including model 1, iterate the generations dataset and calculate the log likelihood 
        # of the samples under model 1. Write all into to a file with the ID. 

        total_complete = 0
        
        all_dims = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 1384] 
        all_model_names = [f"vae_ldim_{dim}" for dim in all_dims]

        for model_name2 in all_model_names: 
            print(f"Calculate_Log_Likelihood: Starting to calculate LL for {model_name2}'s generations...", flush = True)
            per_model_complete = 0
            
            try: 
                # Load samples dataset
                samples_path = f"/scratch/mr7401/vae_generations/{model_name2}/samples/"
                if not os.path.exists(samples_path):
                    raise FileNotFoundError(f"Calculate Log Likelihood: File not found: {samples_path}. Please check the path and try again.")
                
                # Get all .png files in the directory
                image_files = [os.path.join(samples_path, f) for f in os.listdir(samples_path) if f.endswith('.png')]
                print(f"Found {len(image_files)}")

                # Define a simple transformation for the images
                transform = transforms.Compose([
                    transforms.ToTensor(),
                ])

                # Create a simple dataset from the image files
                class ImageDataset:
                    def __init__(self, image_files, transform=None):
                        self.image_files = image_files
                        self.transform = transform

                    def __len__(self):
                        return len(self.image_files)

                    def __getitem__(self, idx):
                        image_path = self.image_files[idx]
                        image = Image.open(image_path).convert("RGB")
                        if self.transform:
                            image = self.transform(image)
                        return {"image": image, "id": os.path.basename(image_path)}

                # Instantiate the dataset and dataloader
                batch_size = 1
                image_dataset = ImageDataset(image_files, transform=transform)
                dataloader = DataLoader(image_dataset, batch_size=1, shuffle=False)
                if logger is not None:
                    logger.log({"Using Batch Size": batch_size})

    
                # Iterate dataloader, calculating log prob under model 1
                for batch in dataloader:
                    batch_ll = model1.log_likelihood(batch["image"], n_samples=len(batch["image"]))
                    
                    if verbose: 
                        print(f"    Gen Log Likelihood: {batch_ll}", flush=True)
                    
                    # Make dataset items (N = batch_size) and write them to a file
                    for i in range(len(batch["id"])):
                        if i < len(batch["id"]):
                            total_complete = total_complete + 1
                            per_model_complete = per_model_complete + 1
                            data_entry = {
                                "gen_source_model": model_name2,
                                "generation_id": batch["id"][i],
                                f"{model_name}_ll": batch_ll[i].item()                
                            }
                            writer.write(data_entry)
                        else: 
                            warnings.warn(f"Warning: A generation after {per_model_complete} calculations did not have a batch id, a batch sequence, and a log likelihood. " 
                                            f"    - Information: \nLen(Batch[)={len(batch["id"])}, Len(Gen_Log_Likelihood) = {len(batch_ll)}\n")
                        
                    if logger is not None:
                        logger.log({f"{model_name2}_Complete": per_model_complete})
                        logger.log({"Total Complete": total_complete})
            except Exception as e:
                    print(f"    Calculating the log likelihood of the following batch failed: \n Batch: {batch["id"]}\nSpecific Error: {e} \nSkipping.", flush = True)
            
    print(f"    *********** Completed batches! ************** ", flush = True)
                                  
    return  

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected in the form of: True/False, T/F, Yes/No, Y/N, or 1/0')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate LLs using a VAE model")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use for generation')
    parser.add_argument('--checkpoint_dir', type=str, required=False, default = "/scratch/mr7401/projects/meta_comp/checkpoints/", help='Directory containing model checkpoints')
    parser.add_argument('--save_dir', type=str, required=False, default = "/scratch/mr7401/vae_generations/", help='Directory to save the generated images')
    parser.add_argument('--test', type=str2bool, required=False, default=False, help='If True, run this script in testing mode (testing loading the models and saving, without generation)')
    parser = add_log_args(parser)

    args, unknown_args = parser.parse_known_args()
  
    # Set up logging
    logging_name = f"{args.model_name}"
    logger = Logger(group = "Generate_VAE_LogLikelihoods", logging_name=logging_name, **vars(args))

    os.makedirs(f"{args.save_dir}", exist_ok=True)
    os.makedirs(f"{args.save_dir}/{args.model_name}", exist_ok = True)
    
    output_file = f"{args.save_dir}/{args.model_name}/log_likelihood.jsonl"

    # Call function 
    calculate_log_likelihood(model_name=args.model_name, output_file= output_file, logger=logger, verbose = False)

    print(f"Calculated Log Likelihood for All Datasets Under {args.model_name} and saved to {output_file}", flush = True)
    logger.finish()