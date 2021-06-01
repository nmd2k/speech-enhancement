from PIL import Image
from model.config import *
import torch
import os
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

        self.root_dir   = root_dir
        self.ids        = os.listdir(root_dir)
        
        if transform is None:
            self.transfrom = transforms.Compose([transforms.Resize((INPUT_SIZE, INPUT_SIZE)), 
                                                  transforms.Grayscale(), 
                                                  transforms.ToTensor(),])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id    = self.ids[index]

        # file should be unzipped
        image = Image.open(self.root_dir+IMAGE_PATH+id+'.png')
        mask  = Image.open(self.root_dir+MASK_PATH+id+'.png')
    
        image = self.transfrom(image)
        mask  = self.transfrom(mask)

        return image, mask

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
