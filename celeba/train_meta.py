from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import os
from torch.optim import AdamW
import wandb
import sys 
sys.path.append("/scratch/mr7401/projects/meta_comp/")
# sys.path.append("../")
# import sys 
# sys.path.append("../")
# sys.path.append("../gmm")
from models.new_models_to_use import SetTransformer2New
from logger import Logger 
np.set_printoptions(suppress=True)
import torch 
import pickle 
import pandas as pd
import argparse
import numpy as np


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
            
    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.y[idx], idx # returns the explicit data needed for prediction along with the index

def get_test_datasets (MN, label_type, n_per_set, seed = 0, device = "cpu", ood_only = True):
    if label_type == "kldiff":    
        test_dir = f"/scratch/mr7401/meta_comp_data/vaes/metadatasets/MN_{MN}/{n_per_set}_2000/"
    elif label_type == "binary_diff": 
        test_dir = f"/scratch/mr7401/meta_comp_data/vaes/binary_metadatasets/MN_{MN}/{n_per_set}_2000/"
    
    if not os.path.exists(test_dir):
        print(f"\n\nERROR Test directory {test_dir} does not exist. Please check the path and try again.", flush=True)
        return {}
    
    test_names = [s for s in os.listdir(test_dir) if "test" in s and ".pt" in s]
    if ood_only:
        test_names = [s for s in test_names if "both" in s]
    print(f"Found {len(test_names)} test datasets: {test_names}", flush=True)
    test_names = [s.split("/")[-1].split(".")[0] for s in test_names]
    test_datasets = {}

    for test_name in tqdm(test_names, desc="Loading test datasets"):
        #print("Loading test dataset:", test_name, flush=True)
        if label_type == "kldiff":
            dataset_path = f"/scratch/mr7401/meta_comp_data/vaes/metadatasets/MN_{MN}/{n_per_set}_2000/{test_name}.pt" 
        elif label_type == "binary_diff":
            dataset_path = f"/scratch/mr7401/meta_comp_data/vaes/binary_metadatasets/MN_{MN}/{n_per_set}_2000/{test_name}.pt" 
        print("Loading test dataset from:", dataset_path, flush=True)
        test_data = torch.load(dataset_path)

        print(f"Found {len(test_data['x1'])} samples in {test_name}", flush=True)
        if label_type == "kldiff":
            print(f"KLDiff Label Checks: \n- Number of Unique Label Values: {len(list(set([x.item() for x in test_data['y']])))} \n- {100*(test_data['y'] > 0).sum() / len(test_data['y']):.3f}% Greater than 0")
        
        elif label_type == "binary_diff":
            print(f"Binary Label Checks: \n- Label Values: {list(set([x.item() for x in test_data['y']]))} \n- {100*(test_data['y'] > 0).sum() / len(test_data['y']):.3f}% Greater than 0")

        
        metadata_path = dataset_path.replace(".pt", "_metadata.pkl")
        print("Loading test metadata from:", metadata_path, flush=True)
        
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

    new_test_datasets = {}
    for original_name, dataset in test_datasets.items():
        tokens = original_name.split('_')
        new_name = f"{tokens[-2]} vs {tokens[-1]}"
        new_test_datasets[new_name] = dataset
    test_datasets = new_test_datasets

    print("Using Test Sets:", ", ".join(test_datasets.keys()), flush=True)
    return test_datasets

def check_label_loss_compatability(label_type, loss_fn):
    if label_type == "kldiff":
        if loss_fn == "bce":
            print("WARNING: Using kldiff labels with bce loss function. This will mean the magnitude prediction head is unused.", flush=True)
    elif label_type == "binary_diff":
        if loss_fn == "mse" or loss_fn == "mse_dir":
            raise ValueError("Using binary_diff labels with mse or mse_dir loss function is not supported. Please use bce loss function, or change the label_type to 'kldiff'")
    return 

def train(nepochs, label_type = "kldiff", batch_size=32, MN=2, training_seed = 0, n_per_set = 5, lr=1e-3, loss_fn='mse', temperature=1.0, n_to_sample = 0, verbose = False, test= False):
    # Set training seed, devices, and logging 
    torch.manual_seed(training_seed)
    np.random.seed(training_seed)
    torch.autograd.set_detect_anomaly(True)

    if torch.cuda.is_available():
        print("Using CUDA", flush=True)
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')
    print("Using device:", device, flush=True)
    print("Setting up logger...", flush=True)
    os.makedirs("/scratch/mr7401/logs/test/wandb/", exist_ok=True)
    check_label_loss_compatability(label_type = label_type, loss_fn = loss_fn)

    if test: 
        project_name = "test"
    else: 
        project_name = "vae_meta_learning_binary_testing"
    logger = Logger(logger = "wandb", log_dir = "/scratch/mr7401/logs/meta_learning_binary_mag", 
                    project =  project_name, 
                    logging_name = f"LR_{lr}_seed_{training_seed}_{device}", 
                    group = f"{label_type}", 
                    seed= training_seed, 
                    lr= lr, 
                    n_per_set = n_per_set, 
                    MN = MN, 
                    batch_size = batch_size, 
                    nepochs = nepochs, 
                    loss_fn = loss_fn, 
                    label_type = label_type)


    # Define data reading paths and where to save checkpoints 
    if label_type == "kldiff": 
        dataset_path = f"/scratch/mr7401/meta_comp_data/vaes/metadatasets/MN_{MN}/{n_per_set}_10000/train.pt" 
        metadata_path = f"/scratch/mr7401/meta_comp_data/vaes/metadatasets/MN_{MN}/{n_per_set}_10000/train_metadata.pkl"
        checkpoint_dir = f"/scratch/mr7401/meta_comp_data/vaes/checkpoints/MN_{MN}/{n_per_set}_10000/{wandb.run.id}-mse_dir"
    elif label_type == "binary_diff": 
        dataset_path = f"/scratch/mr7401/meta_comp_data/vaes/binary_metadatasets/MN_{MN}/{n_per_set}_10000/train.pt" 
        metadata_path = f"/scratch/mr7401/meta_comp_data/vaes/binary_metadatasets/MN_{MN}/{n_per_set}_10000/train_metadata.pkl"
        checkpoint_dir = f"/scratch/mr7401/meta_comp_data/vaes/binary_checkpoints/MN_{MN}/{n_per_set}_10000/{wandb.run.id}"
    
    if checkpoint_dir: 
        os.makedirs(checkpoint_dir, exist_ok=True)

    print("Loading dataset from:", dataset_path, flush=True)
    print("Loading metadata from:", metadata_path, flush=True)
    print("Saving checkpoints to:", checkpoint_dir, flush=True)

    train_data = torch.load(dataset_path)
    with open(metadata_path, "rb") as f:
        train_metadata = pickle.load(f)
    
    if n_to_sample > 0:
        print(f"WARNING: Sampling {n_to_sample} samples from train data", flush=True)
        indices = np.random.permutation(len(train_data["x1"]))[:n_to_sample]
        train_data = {
            "x1": train_data["x1"][indices],
            "x2": train_data["x2"][indices],
            "y": train_data["y"][indices]
        }
        train_metadata = {
            "m1_ids": [train_metadata["m1_ids"][i] for i in indices],
            "m2_ids": [train_metadata["m2_ids"][i] for i in indices],
            "m1s": [train_metadata["m1s"][i] for i in indices],
            "m2s": [train_metadata["m2s"][i] for i in indices]
        }
    
    print(f"Found {len(train_data['x1'])} training samples", flush=True)
    if label_type == "kldiff":
        print(f"KLDiff Label Checks: \n- Number of Unique Label Values: {len(list(set([x.item() for x in train_data['y']])))} \n- {100*(train_data['y'] > 0).sum() / len(train_data['y']):.3f}% Greater than 0")
        print(f"First two y values: {train_data['y'][:2]}")
    
    elif label_type == "binary_diff":
        print(f"Binary Label Checks: \n- Label Values: {list(set([x.item() for x in train_data['y']]))} \n- {100*(train_data['y'] > 0).sum() / len(train_data['y']):.3f}% Greater than 0")
        print(f"First two y values: {train_data['y'][:2]}")
    if not test: 
        test_datasets= get_test_datasets(MN, label_type, n_per_set, seed = 2, device = device)
    else: 
        test_datasets = {}


    # Build Model 
    n_features = 768
    model = SetTransformer2New(
        n_inputs=n_features,
        n_outputs=1,
        n_enc_layers=3,
        dim_hidden=128,
        norm='set_norml',
        sample_size=batch_size
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
    if not test:
        for test_name, test_dataset in test_datasets.items():
            
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=None)
            test_dataloaders[test_name] = test_dataloader

    # Set up optimizer
    optim = AdamW(model.parameters(), lr=lr)

    print(f"Starting training for {nepochs} epochs with batch size {batch_size} and learning rate {lr}, using loss function {loss_fn}", flush=True)
    epoch_train_losses = []
    epoch_test_losses = {}

    epoch_train_accs = []
    epoch_test_accs = {}

    for test_name in test_dataloaders.keys():
        epoch_test_losses[test_name] = []
        epoch_test_accs[test_name] = []

    for epoch in tqdm(range(nepochs)):
        losses = []
        accs = []

        model.train()
        for i, (x1_batch, x2_batch, y, inds) in enumerate(train_dataloader):
            if i % 10 == 0:
                print(f"Batch {i}/{len(train_dataloader)}", flush=True)
            sample_indices = inds # tensor[batch_size]
        
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

            if verbose:
                print("Model output shapes: ", out[0].shape, out[1].shape, flush=True)
                #print("Model output, index 0:", out[0], flush=True)
                print("Model output (all): ", out, flush=True)

            # Compute losses
            if loss_fn == 'mse':
                out_ = out[0]
                out_ = out_.squeeze()
                loss = torch.mean((F.relu(out_) - torch.abs(y))**2)
                if verbose:
                    print("Loss shape:", loss.shape, flush=True)
                    print("Loss:", loss, flush=True)
            elif loss_fn == 'mse_dir':
                mag_loss = torch.mean((out[0].squeeze() - torch.abs(y))**2)
                dir_loss = F.binary_cross_entropy_with_logits(out[1].squeeze(), (y > 0).float())
                loss = mag_loss + dir_loss
            elif loss_fn == "bce":
                out_ = out.squeeze() # use the magnitude prediction head
                loss = F.binary_cross_entropy(out_, y.float())
            
            # Add to logging 
            losses.append(loss.item())
            
            if loss_fn == 'mse_dir':
                acc = ((out[1].squeeze() > 0) == (y > 0)).float().mean()
                accs.append(acc.item())
                wandb.log({'train/mse_loss': mag_loss.item(),
                           'train/bce_loss': dir_loss.item(),
                           'train/bce_acc': acc.item(),
                           'train/mse_dir_loss': loss.item(),
                           'epoch': epoch + 1,
                }) 
            elif loss_fn == 'bce':
                #acc = ((torch.sigmoid(out[0].squeeze()) > 0.5) == (y > 0)).float().mean()
                out_ = out.squeeze()
                acc = ((out_ > 0.5) == (y > 0)).float().mean()
                accs.append(acc.item())
                wandb.log({
                        f'train/bce_loss': loss.item(),
                        f'train/bce_acc': acc.item(),
                        'epoch': epoch + 1,
                })
            else:
               wandb.log({'train/' + loss_fn + "_loss": loss.item()})
               print("")

            loss.backward()
            optim.step()
    
        # Report average training loss 
        avg_train_loss = np.mean(losses)
        epoch_train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1}/{nepochs}, Train Loss: {avg_train_loss:.4f}", flush=True)
        wandb.log({'train/avg_' + loss_fn + "_loss": avg_train_loss})
        if len(accs) > 0:
            avg_train_acc = np.mean(accs)
            epoch_train_accs.append(avg_train_acc)
            print(f"Epoch {epoch + 1}/{nepochs}, Train Accuracy: {avg_train_acc:.4f}", flush=True)
            wandb.log({'train/avg_bce_acc': avg_train_acc})


        # If testing, evaluate on any test sets 
        for test_dl_name, test_dataloader in test_dataloaders.items():

            model.eval()
            test_losses = []
            test_acc = []
            with torch.no_grad():
                for i, (x1_batch, x2_batch, y, inds) in enumerate(test_dataloader):
                    sample_indices = inds
                    batch_x = [nset[0] for nset in x1_batch]
                    x_lengths = torch.tensor([len(x) for x in batch_x])
                    
                    if device != "cpu":
                        x1_batch = x1_batch.to(device)
                        x2_batch = x2_batch.to(device)
                        y = y.to(device)
                        x_lengths = x_lengths.to(device)
                    
                    out = model(x1_batch, x2_batch, x_lengths)
                    
                   # Compute losses
                    if loss_fn == 'mse':
                        out_ = out[0]
                        out_ = out_.squeeze()
                        loss = torch.mean((F.relu(out_) - torch.abs(y))**2)
                        if verbose:
                            print("Loss shape:", loss.shape, flush=True)
                            print("Loss:", loss, flush=True)
                        test_losses.append(loss.item())
                    
                    elif loss_fn == 'mse_dir':
                        mag_loss = torch.mean((out[0].squeeze() - torch.abs(y))**2)
                        dir_loss = F.binary_cross_entropy_with_logits(out[1].squeeze(), (y > 0).float())
                        loss = mag_loss + dir_loss
                        acc = ((out[1].squeeze() > 0) == (y > 0)).float().mean()
                        test_acc.append(acc.item())
                        test_losses.append(loss.item())
                    
                    elif loss_fn == "bce":
                        out_ = out.squeeze() 
                        loss = F.binary_cross_entropy(out_, y.float())
                        #acc = ((torch.sigmoid(out[0].squeeze()) > 0.5) == (y > 0)).float().mean()
                        acc = ((out_ > 0.5) == (y > 0)).float().mean()
                        if (epoch == 0 or epoch == nepochs - 1) and i < 3:
                            print(f"\nOUTPUT TEST, {test_dl_name}, Epoch {epoch + 1} Batch {i} \nModel output: {out}\nModel output squeezed: {out_}\nLabels:{y}\nAccuracy: {((out_ > 0.5) == (y > 0)).float()}\nAcc Mean: {acc}\nLoss: {loss}", flush=True)
                        if verbose:
                            print("Test Loss, Batch:", loss, flush=True)
                            print("Test Acc, Batch:", acc, flush=True)
                        test_acc.append(acc.item())
                        test_losses.append(loss.item())
               
            # Keep track of losses
            avg_test_loss = np.mean(test_losses)
            epoch_test_losses[test_dl_name].append(avg_test_loss)
            print(f"Epoch {epoch + 1}/{nepochs}, {test_dl_name} Test Loss: {avg_test_loss:.4f}", flush=True)
            wandb.log({"test_loss_" + loss_fn + "/" + test_dl_name: avg_test_loss})
            
            # Keep track of accuracies 
            if len(test_acc) > 0:
                avg_test_acc = np.mean(test_acc)
                print(f"Epoch {epoch + 1}/{nepochs}, {test_dl_name} Test Accuracy: {avg_test_acc:.4f}", flush=True)
                wandb.log({ "test_acc_avg" + "/" + test_dl_name: avg_test_acc})

        # At the end of an epoch, save the model checkpoint
        if checkpoint_dir:
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


    print(f"Training complete. Saving losses..", flush=True)

    # Save all epoch losses to CSV using pandas
    if checkpoint_dir:
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
    parser.add_argument("--label_type", type= str, default = "kldiff", choices=["kldiff", "binary_diff"], help = "Label type to use.")
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
        label_type = args.label_type, 
        batch_size=args.batch_size,
        lr=args.lr,
        training_seed=args.training_seed,
        loss_fn=args.loss_fn,
        temperature=args.temperature,
        n_to_sample=args.n_to_sample,
        verbose=args.verbose,
    )
