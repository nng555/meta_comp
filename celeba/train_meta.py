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


class MetaDataset(Dataset):

    def __init__(self, meta_x1, meta_x2, meta_y, metadata):
        self.x1 = meta_x1 # N x M x D
        self.x2 = meta_x2 # N x M x D
        self.y = meta_y # N x M x M
        
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

def get_test_datasets (MN, n_per_set, seed = 0, device = "cpu"):
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

def train(nepochs, batch_size=32, MN=2, training_seed = 0, n_per_set = 5, lr=1e-3, loss_fn='mse', temperature=1.0, n_to_sample = 0, verbose = False):
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
    logger = Logger(logger = "wandb", log_dir = "/scratch/mr7401/logs/meta_learning", project = "vae_meta_learning_mse_dir", logging_name = f"LR_{lr}_seed_{training_seed}", group = f"LR_{lr}", seed= training_seed, lr= lr, n_per_set = n_per_set, MN = MN, batch_size = batch_size, nepochs = nepochs, loss_fn = loss_fn)

    dataset_path = f"/scratch/mr7401/meta_comp_data/vaes/metadatasets/MN_{MN}/{n_per_set}_10000/train_seed_2.pt" 
    metadata_path = f"/scratch/mr7401/meta_comp_data/vaes/metadatasets/MN_{MN}/{n_per_set}_10000/train_metadata_seed_2.pkl"
    checkpoint_dir = f"/scratch/mr7401/meta_comp_data/vaes/checkpoints/MN_{MN}/{n_per_set}_10000/{wandb.run.id}-mse_dir"
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("Loading dataset from:", dataset_path, flush=True)
    print("Loading metadata from:", metadata_path, flush=True)
    print("Saving checkpoints to:", checkpoint_dir, flush=True)

    train_data = torch.load(dataset_path)
    with open(metadata_path, "rb") as f:
        train_metadata = pickle.load(f)
    
    if n_to_sample > 0:
        print(f"WARNING: Sampling {n_to_sample} samples from train and test data", flush=True)
        train_data = {
            "x1": train_data["x1"][:n_to_sample],
            "x2": train_data["x2"][:n_to_sample],
            "y": train_data["y"][:n_to_sample]
        }
        train_metadata = {
            "m1_ids": train_metadata["m1_ids"][:n_to_sample],
            "m2_ids": train_metadata["m2_ids"][:n_to_sample],
            "m1s": train_metadata["m1s"][:n_to_sample],
            "m2s": train_metadata["m2s"][:n_to_sample]
        }
    
    print(f"Found {len(train_data['x1'])} training samples", flush=True)
    test_datasets= get_test_datasets(MN, n_per_set, seed = 2, device = device)

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
        meta_y=train_data["y"],
        metadata = train_metadata
    )
   
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=None)

    test_dataloaders = {}
    for test_name, test_dataset in test_datasets.items():
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=None)
        test_dataloaders[test_name] = test_dataloader

    # Set up optimizer
    optim = AdamW(model.parameters(), lr=lr)

    print(f"Starting training for {nepochs} epochs with batch size {batch_size} and learning rate {lr}, using loss function {loss_fn}", flush=True)
    epoch_train_losses = []
    epoch_test_losses = {}
    for test_name in test_dataloaders.keys():
        epoch_test_losses[test_name] = []

    # For the last epoch, save train set predictions
    last_epoch_train_indices = []
    last_epoch_train_predictions = [] 
    last_epoch_train_targets = []
    
    for epoch in tqdm(range(nepochs)):
        losses = []

        model.train()
        for i, (x1_batch, x2_batch, y, indices) in enumerate(train_dataloader):
            sample_indices = indices # tensor[batch_size]
        
            # Set optimizer and get model output
            optim.zero_grad()
            # Get model output
            batch_x = [nset[0] for nset in x1_batch] # B x N x M x D
            x_lengths = torch.tensor([len(x) for x in batch_x])
            if verbose:
                print("Batch x1 shape:", x1_batch.shape, flush=True)
                print("Batch x2 shape:", x2_batch.shape, flush=True)
                print("Batch y shape:", y.shape, flush=True)
                print("X lengths:", x_lengths, flush=True)
                print("Sample indices:", sample_indices, flush=True)
        
            out = model(x1_batch, x2_batch, x_lengths)

            # If last epoch, save predictions
            if epoch == nepochs - 1:
                last_epoch_train_indices.append(sample_indices)
                last_epoch_train_predictions.append(out[0])
                last_epoch_train_targets.append(y)
        
            if verbose:
                print("Model output shapes: ", out[0].shape, out[1].shape, flush=True)

            # Compute losses
            if loss_fn == 'mse':
                out = out[0]
                out = out.squeeze()
                loss = torch.mean((F.relu(out) - torch.abs(y))**2)
                if verbose:
                    print("Loss shape:", loss.shape, flush=True)
                    print("Loss:", loss, flush=True)
            elif loss_fn == 'mse_dir':
                mag_loss = torch.mean((out[0].squeeze() - torch.abs(y))**2)
                dir_loss = F.binary_cross_entropy_with_logits(out[1].squeeze(), (y > 0).float())
                loss = mag_loss + dir_loss
            elif loss_fn == "bce":
                out = out.squeeze()
                out /= temperature
                y /= temperature
                py = F.sigmoid(y)
                loss = F.binary_cross_entropy_with_logits(out, py)
            
            # Add to logging 
            losses.append(loss.item())
            if loss_fn == 'mse_dir':
                wandb.log({'mse_loss': mag_loss.item(),
                           'bce_loss': dir_loss.item(),
                           'bce_acc': ((out[1].squeeze() > 0) == (y > 0)).float().mean(),
                           'mse_dir_loss': loss.item(),
                })
                out = out[0].squeeze() * ((out[1].squeeze() > 0).float() * 2 - 1)
            else:
                wandb.log({'train_' + loss_fn + "_loss": loss.item()})
            loss.backward()
            optim.step()
    
        # Report average training loss 
        avg_train_loss = np.mean(losses)
        epoch_train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1}/{nepochs}, Train Loss: {avg_train_loss:.4f}", flush=True)
        wandb.log({'train_avg_' + loss_fn + "_loss": avg_train_loss})
        
    
        # If testing, evaluate on any test sets 
        for test_dl_name, test_dataloader in test_dataloaders.items():
            # Evaluate on test set
            model.eval()
            test_losses = []
            with torch.no_grad():
                for i, (x1_batch, x2_batch, y, indices) in enumerate(test_dataloader):
                    batch_x = [nset[0] for nset in x1_batch]
                    x_lengths = torch.tensor([len(x) for x in batch_x])
                    
                    if device != "cpu":
                        x1_batch = x1_batch.to(device)
                        x2_batch = x2_batch.to(device)
                        y = y.to(device)
                        x_lengths = x_lengths.to(device)
                    
                    out = model(x1_batch, x2_batch, x_lengths)
                    
                    if loss_fn == 'mse':
                        out = out[0].squeeze()
                        test_loss = torch.mean((F.relu(out) - torch.abs(y))**2)
                    elif loss_fn == 'mse_dir':
                        mag_loss = torch.mean((out[0].squeeze() - torch.abs(y))**2)
                        dir_loss = F.binary_cross_entropy_with_logits(out[1].squeeze(), (y > 0).float())
                        test_loss = mag_loss + dir_loss
                    elif loss_fn == "bce":
                        out = out.squeeze()
                        out /= temperature
                        y /= temperature
                        py = F.sigmoid(y)
                        test_loss = F.binary_cross_entropy_with_logits(out, py)
                    
                    test_losses.append(test_loss.item())
            
            avg_test_loss = np.mean(test_losses)
            epoch_test_losses[test_dl_name].append(avg_test_loss)
            print(f"Epoch {epoch + 1}/{nepochs}, {test_dl_name} Test Loss: {avg_test_loss:.4f}", flush=True)
            wandb.log({test_dl_name + loss_fn + "_loss": avg_test_loss})

        # At the end of an epoch, save the model checkpoint
        e = os.path.join(checkpoint_dir, "epoch_checkpoints")
        os.makedirs(e, exist_ok=True)
        checkpoint_path = os.path.join(e, f"epoch_{epoch + 1}.pt")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': avg_train_loss,
        }, checkpoint_path)

        print(f"Checkpoint saved at {checkpoint_path}", flush=True)

    print(f"Training complete. Saving final model and predictions...", flush=True)
    
    # Save the last epoch predictions
    last_epoch_train_indices = torch.cat(last_epoch_train_indices)
    last_epoch_train_predictions = torch.cat(last_epoch_train_predictions)
    last_epoch_train_targets = torch.cat(last_epoch_train_targets)
    
    # Print shapes for debugging
    print("Last epoch indices shape:", last_epoch_train_indices.shape, flush=True)
    print("Last epoch predictions shape:", last_epoch_train_predictions.shape, flush=True)
    print("Last epoch targets shape:", last_epoch_train_targets.shape, flush=True)
    
    if device != "cpu":
        last_epoch_train_indices = last_epoch_train_indices.cpu()
        last_epoch_train_predictions = last_epoch_train_predictions.cpu()
        last_epoch_train_targets = last_epoch_train_targets.cpu()

    # Save the last epoch predictions, targets, and indices to a CSV file
    train_predictions_csv_path = os.path.join(checkpoint_dir, "last_epoch_train_predictions.csv")
    train_predictions_df = pd.DataFrame({
        "sample_index": last_epoch_train_indices.detach().numpy().flatten(),
        "prediction": last_epoch_train_predictions.detach().numpy().flatten(),
        "target": last_epoch_train_targets.detach().numpy().flatten()
    })
    train_predictions_df.to_csv(train_predictions_csv_path, index=False)
    print(f"Last epoch trainng set predictions saved to {train_predictions_csv_path}", flush=True)

    # Save all epoch losses to CSV using pandas
    csv_path = os.path.join(checkpoint_dir, "epoch_losses.csv")
    loss_dict = {"epoch": range(1, len(epoch_train_losses) + 1), "train_loss": epoch_train_losses}
    for test_name, test_losses in epoch_test_losses.items():
        loss_dict[test_name] = test_losses
    df = pd.DataFrame(loss_dict)
    df.to_csv(csv_path, index=False)
    print(f"All epoch train and test losses saved to {csv_path}", flush=True)
 
    logger.finish()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the MetaDataset model.")
    parser.add_argument("--nepochs", type=int, default=30, help="Number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--training_seed", type=int, default=0, help="Random seed for training.")
    parser.add_argument("--loss_fn", type=str, default="mse_dir", choices=["mse", "mse_dir", "bce"], help="Loss function to use.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for loss scaling.")
    parser.add_argument("--n_to_sample", type=int, default=0, help="Number of samples to use from the dataset.")
    parser.add_argument("--verbose", type = bool, default = False, help="Enable verbose logging.")

    args = parser.parse_args()

    train(
        nepochs=args.nepochs,
        batch_size=args.batch_size,
        lr=args.lr,
        training_seed=args.training_seed,
        loss_fn=args.loss_fn,
        temperature=args.temperature,
        n_to_sample=args.n_to_sample,
        verbose=args.verbose,
    )
