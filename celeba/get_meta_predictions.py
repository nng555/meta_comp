from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm
import torch
import numpy as np
import os
from torch.optim import AdamW
import wandb
import sys 
sys.path.append("/scratch/mr7401/projects/meta_comp/")
sys.path.append("../")
import sys 
sys.path.append("../")
sys.path.append("../gmm")
from gmm.set_transformer2 import *
from logger import Logger 
np.set_printoptions(suppress=True)
import torch 
import pickle 
import pandas as pd
import argparse
import io 

import pandas as pd 
import wandb
from datetime import datetime
api = wandb.Api()
def get_wandb_runs_for_project(project_name = "vae_meta_learning_mse_dir_rerun"): 
     # Project is specified by <entity/project-name>
    runs = api.runs(project_name)
    
    summary_list, config_list, name_list, id_list, created_at_list = [], [], [], [], []
    for run in runs: 
        if run.state != "finished":
            continue
    
        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
              if not k.startswith('_')})
    
        # .name is the human-readable name of the run.
        name_list.append(run.name)
    
        # .id is the unique identifier of the run.
        id_list.append(run.id)
    
        # .created_at is the creation date of the run.
        created_at_list.append(run.created_at)
    
    # Unpack the config so that each key becomes a column
    config_df = pd.DataFrame(config_list)
    
    # Combine the unpacked config with the other data
    runs_df = pd.concat([
        pd.DataFrame({
            "name": name_list,
            "id": id_list,
             "created_at": [datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d %H:%M:%S") for date in created_at_list]
        }),
        config_df
    ], axis=1)
    return runs_df 

def get_wandb_run_id(project_name = "vae_meta_learning_mse_dir_rerun", lr = 0.01, seed = 0, n_per_set = 5): 
    runs_df = get_wandb_runs_for_project(project_name) # cols = name, id, created_at, <all keys in config>
    print(f"- Found {len(runs_df)} completed runs in project {project_name}", flush = True)

    if len(runs_df) > 0: 
        runs_with_lr = runs_df[runs_df["lr"] == lr]
        print(f"- Found {len(runs_with_lr)} run(s) with learning rate {lr}", flush = True)
        runs_with_lr_and_seed = runs_with_lr[runs_with_lr["seed"] == seed]
        print(f"- Found {len(runs_with_lr_and_seed)} run(s) with learning rate {lr} and seed {seed}", flush = True)
        runs_with_lr_seed_and_n_per_set = runs_with_lr_and_seed[runs_with_lr_and_seed["n_per_set"] == n_per_set]
        print(f"- Found {len(runs_with_lr_seed_and_n_per_set)} run(s) with learning rate {lr}, seed {seed}, and n_per_set {n_per_set}", flush = True)
        if len(runs_with_lr_seed_and_n_per_set) == 1: 
            return runs_with_lr_seed_and_n_per_set["id"].iloc[0]
        elif len(runs_with_lr_seed_and_n_per_set) == 0: 
            print(f"- Available n_per_set values for this run appear to be: {runs_with_lr_and_seed['n_per_set'].to_list()}")
            return ""
        else: 
            return runs_with_lr_seed_and_n_per_set["id"].to_list()
    else:
        return ""


class MetaDataset(Dataset):

    def __init__(self, meta_x1, meta_x2, meta_y, metadata = None):
        self.x1 = meta_x1 # N x M x D
        self.x2 = meta_x2 # N x M x D
        self.y = meta_y # N x M x M
        print(f"Example y samples: {self.y[0:10]}", flush=True)

        if metadata is not None: 
        
            self.m1_ids = metadata["m1_ids"] # N x M
            self.m2_ids = metadata["m2_ids"] # N x M 
            self.m1s = metadata["m1s"] # N x M
            self.m2s = metadata["m2s"] # N x M

    def __len__(self): 
        return len(self.y)
        
    def get_metadata(self, idx): 
        return self.m1_ids[idx], self.m2_ids[idx], self.m1s[idx], self.m2s[idx] 
    
    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.y[idx], idx # returns the explicit data needed for prediction along with the index

def diagnose_file_missing(file_path):
    print(f"Diagnosing missing file: {file_path}", flush =True)
    current_path = file_path

    for level in range(3):
        current_path = os.path.dirname(current_path)
        if os.path.exists(current_path):
            num_files = len(os.listdir(current_path))
            print(f"Level {level + 1}: {current_path} contains {num_files} files.")
        else:
            print(f"Level {level + 1}: {current_path} does not exist.")

def get_model_checkpoint_path(MN, n_per_set, lr, seed, epoch):
    project_name = "vae_meta_learning_mse_dir_rerun"
    wandb_run_id = get_wandb_run_id(project_name = project_name, lr = lr, seed = seed)
    if wandb_run_id == "":
        raise ValueError(f"No wandb run found for project {project_name} with parameters: lr={lr}, seed={seed}.")
    elif isinstance(wandb_run_id, list):
        raise ValueError(f"Multiple wandb runs found for project {project_name} with parameters: lr={lr},=seed {seed}.\n The runs found were IDs: {wandb_run_id}")
    
    if "mse_dir" in project_name: 
        wandb_run_id = f"{wandb_run_id}-mse_dir"
    
    checkpoint_path = f"/scratch/mr7401/meta_comp_data/vaes/checkpoints/MN_{MN}/{n_per_set}_10000/{wandb_run_id}/epoch_checkpoints/epoch_{epoch}.pt"
    if not os.path.exists(checkpoint_path):
        diagnose_file_missing(checkpoint_path)
        
        raise ValueError(f"Checkpoint path {checkpoint_path} does not exist. Please check the parameters.")
    return checkpoint_path



def get_test_datasets (MN, n_per_set, seed, device = "cpu"):
    test_names = [s for s in os.listdir(f"/scratch/mr7401/meta_comp_data/vaes/metadatasets/MN_{MN}/{n_per_set}_2000/") if "test" in s and f"seed_{seed}" in s and ".pt" in s]
    print(f"Found {len(test_names)} test datasets: {test_names}", flush=True)
    test_names = [s.split("/")[-1].split(".")[0] for s in test_names]
    test_datasets = {}

    for test_name in tqdm(test_names, desc="Loading test datasets"):
        #print("Loading test dataset:", test_name, flush=True)
        
        dataset_path = f"/scratch/mr7401/meta_comp_data/vaes/metadatasets/MN_{MN}/{n_per_set}_2000/{test_name}.pt" 
        #print("Loading test dataset from:", dataset_path, flush=True)
        test_data = torch.load(dataset_path)

        metadata_path = dataset_path.replace("_seed", "_metadata_seed")
        metadata_path = metadata_path.replace(".pt", ".pkl")
        #print("Loading test metadata from:", metadata_path, flush=True)
        
        with open(metadata_path, "rb") as f:
            test_metadata = pickle.load(f)
        
        # Move model and data to GPU if available
        if device != "cpu":
            for key, value in test_data.items():
                if isinstance(value, torch.Tensor):
                    test_data[key] = value.to(device)
                else:
                    test_data[key] = value # list
        
        test_datasets[test_name] = MetaDataset(
            meta_x1=test_data["x1"], 
            meta_x2=test_data["x2"], 
            meta_y=test_data["y"],
            metadata = test_metadata
        )
        test_metadata[test_name] = test_metadata

    return test_datasets


def get_meta_predictions(nepochs, batch_size=32, MN=2, training_seed = 0, n_per_set = 5, lr=1e-3, loss_fn='mse', n_to_sample = 0, test = False, verbose = False):
    # Set seed 
    torch.manual_seed(training_seed)
    np.random.seed(training_seed)

    if torch.cuda.is_available():
        print("Using CUDA", flush=True)
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')
    print("Using device:", device, flush=True)
    print("Setting up logger...", flush=True)
    os.makedirs("/scratch/mr7401/logs/test/wandb/", exist_ok=True)

    logger = Logger(logger = "wandb", log_dir = "/scratch/mr7401/logs/meta_learning", project = "vae_meta_learning_rerun_predictions", logging_name = f"LR_{lr}_seed_{training_seed}_{device}", group = f"LR_{lr}", seed= training_seed, lr= lr, n_per_set = n_per_set, MN = MN, batch_size = batch_size, nepochs = nepochs, loss_fn = loss_fn)

    dataset_path = f"/scratch/mr7401/meta_comp_data/vaes/metadatasets/MN_{MN}/{n_per_set}_10000/train.pt" 
    metadata_path = f"/scratch/mr7401/meta_comp_data/vaes/metadatasets/MN_{MN}/{n_per_set}_10000/train_metadata.pkl"
    checkpoint_path = get_model_checkpoint_path(MN, n_per_set, lr, training_seed, nepochs)
   
    print("Loading dataset from:", dataset_path, flush=True)
    print("Loading metadata from:", metadata_path, flush=True)
    print("Saving checkpoints to:", checkpoint_path, flush=True)

    train_data = torch.load(dataset_path)
    print(f"Found {len(train_data['x1'])} training samples", flush=True)
    import pickle
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    # test_datasets= get_test_datasets(MN, n_per_set, seed = 2, device = device)

    if n_to_sample > 0:
        print(f"WARNING: Sampling {n_to_sample} samples from train and test data", flush=True)
        train_data = {
            "x1": train_data["x1"][:n_to_sample],
            "x2": train_data["x2"][:n_to_sample],
            "y": train_data["y"][:n_to_sample]
        }
        # for test_name, test_dataset in test_datasets.items():
        #     test_datasets[test_name] = MetaDataset(
        #         meta_x1=test_dataset.x1[:n_to_sample], 
        #         meta_x2=test_dataset.x2[:n_to_sample], 
        #         meta_y=test_dataset.y[:n_to_sample]
        #     )
    

    # Build Model 
    n_features = 768
    model = SetTransformer5(
        n_inputs=n_features,
        n_outputs=1,
        n_enc_layers=3,
        dim_hidden=128,
        norm='set_norml',
        sample_size=batch_size,
    )

    # Load the model checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading model checkpoint from {checkpoint_path}", flush=True)
        with open(checkpoint_path, 'rb') as f:
            checkpoint = torch.load(io.BytesIO(f.read()))
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model checkpoint loaded successfully.", flush=True)
    else:
        raise ValueError(f"Checkpoint path {checkpoint_path} does not exist.")

    # Move model and data to GPU if available
    if device != "cpu":
        model = model.to(device)
        for key, value in train_data.items():
            if isinstance(value, torch.Tensor):
                train_data[key] = value.to(device)
            else:
                train_data[key] = value # list
      
    # Make dataloaders 
    train_dataset = MetaDataset(
        meta_x1=train_data["x1"], 
        meta_x2=train_data["x2"], 
        meta_y=train_data["y"]
    )
   
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=None)

    # test_dataloaders = {}
    # for test_name, test_dataset in test_datasets.items():
    #     test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=None)
    #     test_dataloaders[test_name] = test_dataloader
    
    # test_dataloaders["train"] = train_dataloader

    test_dataloaders = {}
    test_dataloaders["train"] = train_dataloader
    print(f"Starting to generate predictions!", flush=True)

    # Set model to eval mode 
    with torch.no_grad():
        model.eval()
        for set_name, dataloader in test_dataloaders.items():
      
            indices = []  # Initialize as a Python list
            mag_predictions = [] 
            sign_predictions = [] 
            targets = []
            set_names = []

            # Getting set predictions 
            for i, (x1_batch, x2_batch, y, inds) in enumerate(dataloader):
                sample_indices = inds # tensor[batch_size]
            
                # Get model output
                batch_x = [nset[0] for nset in x1_batch] # B x N x M x D
                x_lengths = torch.tensor([len(x) for x in batch_x])
                if verbose:
                    print("Batch x1 shape:", x1_batch.shape, flush=True)
                    print("Batch x2 shape:", x2_batch.shape, flush=True)
                    print("Batch y shape:", y.shape, flush=True)
                    print("X lengths:", x_lengths, flush=True)
                    print("Sample indices:", sample_indices, flush=True)
                    print("Y values: ", y, flush = True)
            
                out = model(x1_batch, x2_batch, x_lengths)
                if verbose: 
                    print("Model 0 output:", out[0], flush=True)
                    print("Model 1 output:", out[1], flush=True)
                    print(f"Mag Predictions before: {mag_predictions}")
                
                # Add to saving lists 
                indices.append(sample_indices)
                mag_predictions.append(out[0].squeeze())
                sign_predictions.append(out[1])
                targets.append(y)
                set_names.extend([set_name] * len(y))
                if verbose: 
                    print("Mag Predictions after:", mag_predictions, flush=True)
                    print("Sign Predictions after:", sign_predictions, flush=True)
    
               
            # Save the predictions to a big CSV file :) 
            e = os.path.join(os.path.dirname(checkpoint_path), "predictions", f"epoch_{nepochs}") 
            os.makedirs(e, exist_ok=True)
            predictions_path = os.path.join(e, f"{set_name}.csv")

            # Save the last epoch predictions
            indices = torch.cat(indices)
            mag_predictions = torch.cat(mag_predictions)
            sign_predictions = torch.cat(sign_predictions)
            targets = torch.cat(targets)
    
            # Print shapes for debugging
            print("Indices shape:", indices.shape, flush=True)
            print("Sign predictions shape:", sign_predictions.shape, flush=True)
            print("Magnitude predictions shape:", mag_predictions.shape, flush=True)
            print("Targets shape:", targets.shape, flush=True)
    
            if device != "cpu":
                indices = indices.cpu()
                mag_predictions = mag_predictions.cpu()
                targets = targets.cpu()
                sign_predictions = sign_predictions.cpu()

            predictions_df = pd.DataFrame({
                "sample_index": indices.detach().numpy().flatten(),
                "prediction": mag_predictions.detach().numpy().flatten(),
                "target": targets.detach().numpy().flatten(), 
                "sign_prediction":sign_predictions.detach().numpy().flatten()
            })
            #predictions_df.to_csv(predictions_path, index=False)
            print(f"Last epoch predictions for {set_name} set saved to {predictions_path}", flush=True)

            def check_sign(x):
                    if x < 0: 
                        return -1 
                    else: 
                        return 1 
            # Combine with metadata 
            def make_into_df(data, metadata): 
                r = pd.DataFrame({"sample_index": metadata["sample_index"], "m1_ids": metadata["m1_ids"], "m2_ids": metadata["m2_ids"], "m1s": metadata["m1s"], "m2s": metadata["m2s"]})
                r["m1s"] = r["m1s"].apply(lambda x: x[0])
                r["m2s"] = r["m2s"].apply(lambda x: x[0])
                
                data_df = data
                # Merge on sample_index
                merged_df = pd.merge(r, data_df, on="sample_index")
                merged_df["sign_label"] = (merged_df["target"] > 0).astype("int")
                merged_df["predict_sign_multiplier"] = merged_df["sign_prediction"].apply(check_sign)
                merged_df["prediction_with_sign"] = merged_df["prediction"] * merged_df["predict_sign_multiplier"]
                return merged_df
    
            df = make_into_df(predictions_df, metadata) 
            df.to_csv(predictions_path, index=False)


    logger.finish()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the MetaDataset model.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--training_seed", type=int, default=0, help="Random seed for training.")
    parser.add_argument("--loss_fn", type=str, default="mse_dir", choices=["mse", "mse_dir", "bce"], help="Loss function to use.")
    parser.add_argument("--n_to_sample", type=int, default=0, help="Number of samples to use from the dataset.")
    parser.add_argument("--verbose", type = bool, default = False, help="Enable verbose logging.")

    args = parser.parse_args()

    get_meta_predictions(
        nepochs = 29,
        lr=args.lr,
        training_seed=args.training_seed,
        loss_fn=args.loss_fn,
        n_to_sample=args.n_to_sample,
        verbose=args.verbose,
        test = True
    )
