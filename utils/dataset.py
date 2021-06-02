from PIL import Image
from model.config import *
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, dataloader, random_split
from torchvision import transforms

class SpeechDataset(Dataset):
    """TGS Salt Identification dataset."""
    
    def __init__(self, root_dir=DATA_PATH, transform=None):
        """
        Args:
            root_path (string): Directory with all the images.
            transformer (function): whether to apply the data augmentation scheme
                mentioned in the paper. Only applied on the train split.
        """

        #load noisy voice & clean voice spectrograms created by data_creation mode
        self.image      = np.load(root_dir +'noisy_voice_amp_db.npy')
        clean_voice     = np.load(root_dir +'voice_amp_db.npy')

        # mask is created from noisy voice substract to clean voice
        self.mask       = self.image-clean_voice
        
        # normalize to -1 and 1
        self.image      = scaled_in(self.image)
        self.mask       = scaled_ou(self.mask)
        
        if transform is None:
            self.transfrom = transforms.Compose([transforms.ToTensor(),])

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image = self.transfrom(self.image[index])
        mask  = self.transfrom(self.mask[index])

        return image, mask

def scaled_in(matrix_spec):
    "global scaling apply to noisy voice spectrograms (scale between -1 and 1)"
    matrix_spec = (matrix_spec + 46)/50
    return matrix_spec

def scaled_ou(matrix_spec):
    "global scaling apply to noise models spectrograms (scale between -1 and 1)"
    matrix_spec = (matrix_spec -6 )/82
    return matrix_spec

def inv_scaled_in(matrix_spec):
    "inverse global scaling apply to noisy voices spectrograms"
    matrix_spec = matrix_spec * 50 - 46
    return matrix_spec

def inv_scaled_ou(matrix_spec):
    "inverse global scaling apply to noise models spectrograms"
    matrix_spec = matrix_spec * 82 + 6
    return matrix_spec

def get_dataloader(dataset, 
                    batch_size=BATCH_SIZE, random_seed=RANDOM_SEED, 
                    valid_ratio=VALID_RATIO, shuffle=True, num_workers=NUM_WORKERS):
    """
    Params:
    -------
    - dataset: the dataset.
    - batch_size: how many samples per batch to load.
    - random_seed: fix seed for reproducibility.
    - valid_ratio: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    """

    error_msg = "[!] valid_ratio should be in the range [0, 1]."
    assert ((valid_ratio >= 0) and (valid_ratio <= 1)), error_msg

    # split the dataset
    n = len(dataset)
    n_valid = int(valid_ratio*n)
    n_train = n - n_valid

    # init random seed
    torch.manual_seed(random_seed)

    train_dataset, valid_dataset = random_split(dataset, (n_train, n_valid))

    train_loader = DataLoader(train_dataset, batch_size, shuffle=shuffle, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader
