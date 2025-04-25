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
from transformers import AutoFeatureExtractor, AutoModel, ViTImageProcessor
from tqdm import tqdm

def get_embeddings(model, extractor, imgs, device):
    inputs = extractor(
        images=[x.detach().cpu().numpy() for x in imgs], return_tensors="pt"
    ).to(device)
    if device !="cpu":
        embeds = model(**inputs).last_hidden_state[:, 0].cpu()
    else: 
        embeds = model(**inputs).last_hidden_state[:, 0]
    return embeds

def generate_encodings(logger, test = False, verbose = False):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    if verbose or test:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
    
    # Load Encoding Model
    embed_model_name="google/vit-base-patch16-224"

    models_with_samples = os.listdir("/scratch/mr7401/vae_generations/")
    models_with_samples = [model for model in models_with_samples if os.path.exists(f"/scratch/mr7401/vae_generations/{model}/samples/")]
    if test: 
        models_with_samples = models_with_samples[:1]

    with torch.no_grad():
        
        extractor = ViTImageProcessor.from_pretrained(embed_model_name, do_rescale=False)
        embed_model = AutoModel.from_pretrained(embed_model_name, output_hidden_states=True)
        embed_model.to(device)
        
        total_complete = 0
        for model_name in models_with_samples:
            sample_paths = [os.path.join(f"/scratch/mr7401/vae_generations/{model_name}/samples/", sample) for sample in os.listdir(f"/scratch/mr7401/vae_generations/{model_name}/samples/") if sample.endswith(".png")]
            print(f"Processing {model_name}'s samples, found {len(sample_paths)} samples in folder /scratch/mr7401/vae_generations/{model_name}/samples/", flush=True)
       
            per_model_complete = 0
            
            try: 
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
                image_dataset = ImageDataset(sample_paths, transform=transform)
                dataloader = DataLoader(image_dataset, batch_size=1, shuffle=False)
                if logger is not None:
                    logger.log({"Using Batch Size": batch_size})

                # Create the output directory if it doesn't exist
                output_dir = f"/scratch/mr7401/vae_generations/{model_name}/encodings/google-vit-base-patch16-224/"
                os.makedirs(output_dir, exist_ok=True)
                
                # Iterate dataloader, computing the encodings of each image 
                for batch in tqdm(dataloader, desc=f"Processing {model_name}"):
                    # Move the image to the device
                    images = batch["image"].to(device)

                    # Get embeddings for the batch
                    embeddings = get_embeddings(embed_model, extractor, images, device)

                    # Log the embeddings if verbose
                    if verbose:
                        print(f"    Embeddings Shape: {embeddings.shape}", flush=True)
                    
                    # Save each embedding individually with its corresponding ID
                    for i, embedding in enumerate(embeddings):
                        embedding_id = batch["id"][i]
                        embedding_path = os.path.join(output_dir, f"{embedding_id}.pt")
                        if device != "cpu":
                            embedding = embedding.cpu()
                        torch.save(embedding.cpu(), embedding_path)
    
                if logger is not None:
                    logger.log({f"{model_name}_Complete": per_model_complete})
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

    parser = argparse.ArgumentParser(description="Generate Embeddings for Generated VAE Samples using visual encoder") 
    parser.add_argument('--verbose', type=str2bool, default=False, help='Whether to print verbose output')
    parser.add_argument('--test', type=str2bool, default=False, help='Whether to run in test mode')
    parser = add_log_args(parser)

    args, unknown_args = parser.parse_known_args()
  
    # Set up logging
    logging_name = f"{args.model_name}"
    logger = Logger(group = "Generate_VAE_Embeddings", logging_name=logging_name, **vars(args))

    # Call function 
    generate_encodings(logger, test = args.test, verbose = args.verbose)

    print(f"Calculated Embeddings for All Images", flush = True)
    logger.finish()