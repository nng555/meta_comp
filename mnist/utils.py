import sys
from einops import rearrange
import itertools
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler
import torch.nn.functional as F
import os
import torch
from torch.utils.data import Dataset
from pathlib import Path
import os
from typing import Dict, Any

def gen_collate_fn(max_samples):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def collate_fn(batch): # B x 2 x N

        # ragged tensors
        batch_x1 = [b['x1'] for b in batch]
        batch_x2 = [b['x2'] for b in batch]
        x_lengths = [len(x) for x in batch_x1]
        batch_y = torch.stack([b['y'].mean() for b in batch])

        # right pad to max_samples with zeros
        batch_x1 = [F.pad(bx, (0, 0, 0, max_samples - bl), 'constant', 0) for bx, bl in zip(batch_x1, x_lengths)]
        batch_x1 = torch.stack(batch_x1)
        batch_x2 = [F.pad(bx, (0, 0, 0, max_samples - bl), 'constant', 0) for bx, bl in zip(batch_x2, x_lengths)]
        batch_x2 = torch.stack(batch_x2)

        x_lengths = torch.Tensor(x_lengths).to(device)

        return batch_x1, batch_x2, x_lengths, batch_y

    return collate_fn

class MetaDataset(Dataset):
    """
    A PyTorch Dataset that processes data from a directory structure.

    For each subdirectory in the root directory, it loads 'samples.pt'
    into self.x and processes other '*_ll.pt' files relative to 'base_ll.pt'
    into self.y.

    Args:
        root_dir (str): The path to the root directory containing subdirectories
                        for each data sample.
    """
    def __init__(self, root_dir: str):
        super().__init__()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.root_path = Path(root_dir)
        if not self.root_path.is_dir():
            raise ValueError(f"Provided root_dir '{root_dir}' is not a valid directory.")

        self.x: Dict[str, torch.Tensor] = {}
        self.y: Dict[str, Dict[str, torch.Tensor]] = {}
        self.folder_names: list[str] = []

        for item in self.root_path.iterdir():
            if item.is_dir():
                folder_name = item.name
                self.folder_names.append(folder_name)
                self.y[folder_name] = {}

                samples_path = item / "samples.pt"
                base_ll_path = item / "base_ll.pt"

                if samples_path.exists():
                    self.x[folder_name] = torch.load(samples_path, map_location=device)

                base_ll_tensor = None
                if base_ll_path.exists():
                    base_ll_tensor = torch.load(base_ll_path, map_location=device)

                if base_ll_tensor is not None:
                    for file_path in item.glob('*_ll.pt'):
                        if file_path.name == "base_ll.pt":
                            continue

                        current_tensor = torch.load(file_path, map_location=device)
                        y_key = file_path.name.removesuffix('_ll.pt')
                        self.y[folder_name][y_key] = current_tensor - base_ll_tensor

        import ipdb; ipdb.set_trace()

        # Sort folder names to ensure consistent ordering if needed
        self.folder_names.sort()

    def __len__(self) -> int:
        """Returns the number of processed folders."""
        return len(next(iter(self.x.values())))

    def __getitem__(self, idx):

        # assume
        mname1 = self.folder_names[idx[0][0]]
        mname2 = self.folder_names[idx[0][1]]
        nidx = idx[1]
        x1_data = self.x.get(mname1)[nidx]
        x2_data = self.x.get(mname2)[nidx]

        y_data = self.y.get(mname1).get(mname2)[nidx] - self.y.get(mname2).get(mname1)[nidx]

        import ipdb; ipdb.set_trace()

        return {
            'x1': x1_data,
            'x2': x2_data,
            'y': y_data,
        }

class SetBatchSampler(BatchSampler):

    def __init__(self, batch_size, sampler, model_keys, min_samples=20, max_samples=30):
        super(SetBatchSampler, self).__init__(sampler, batch_size, drop_last=True)
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.model_keys = model_keys

    def build_batch(self, sampler_iter):
        nsamples = np.random.randint(self.min_samples, high=self.max_samples + 1, size=self.batch_size)
        batch = [
            [
                np.random.choice(len(self.model_keys), size=2, replace=False),
                [*itertools.islice(sampler_iter, n)],
            ] for n in nsamples
        ]
        return batch

    def __iter__(self):
        sampler_iter = iter(self.sampler)
        batch = self.build_batch(sampler_iter)
        while batch[-1][-1]:
            yield batch
            batch = self.build_batch(sampler_iter)

if __name__ == "__main__":
    data = MetaDataset('generations/')
    rs = RandomSampler(data)
    bs = SetBatchSampler(16, rs, data.folder_names, min_samples=5, max_samples=10)
    collate_fn = gen_collate_fn(10)
    loader = DataLoader(data, batch_sampler=bs, collate_fn=collate_fn)
    for b in loader:
        import ipdb; ipdb.set_trace()

def print_img(image_array, max_width=80):
    """
    Displays a black and white image from a NumPy array in the terminal.

    Args:
        image_array (numpy.ndarray): A 2D NumPy array representing the image.
            Should have values between 0 and 255 (for grayscale).
        max_width (int, optional): The maximum width of the displayed image in characters.
            Defaults to 80.
    """
    if image_array.ndim != 2:
        print("Error: Input image array must be 2D (grayscale).", file=sys.stderr)
        return

    height, width = image_array.shape
    if width > max_width:
        scale = max_width / width
        new_height = int(height * scale)
        # Use simple interpolation (fastest)
        img = image_array[::int(1/scale), ::int(1/scale)]
        height, width = img.shape
    else:
        img = image_array

    chars = " .:-=+*#%@"  # More visually appealing
    num_chars = len(chars)

    for row in img:
        for pixel in row:
            # Clamp pixel value to the valid range [0, 255]
            clamped_pixel = max(0, min(pixel, 255))
            char_index = int((clamped_pixel / 255.0) * (num_chars - 1))
            sys.stdout.write(chars[char_index])
        sys.stdout.write('\n')  # Newline at the end of each row
    sys.stdout.flush() # Ensure the output is printed immediately.
