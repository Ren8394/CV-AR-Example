from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
np.random.seed(2024)
import pandas as pd
import torch

CLASS_LIST = [
    "sitting", "using_laptop", "hugging", "sleeping", "drinking", 
    "clapping", "dancing", "cycling", "calling", "laughing",
    "eating", "fighting", "listening_to_music", "running", "texting"
]

class HAR(Dataset):
    """
    Human Activity Recognition Dataset
    Source:https://dphi-live.s3.eu-west-1.amazonaws.com/dataset/Human+Action+Recognition-20220526T101201Z-001.zip
    """
    def __init__(self, status="train", transform=None):

        # status: train, val, or test
        assert status in ["train", "val", "test"]
        self.status = status

        self.root = Path.cwd().joinpath("data", "Human Action Recognition")
        self.annot = pd.read_csv(
            self.root.joinpath(f"{self.status}.csv"),
            header=0, names=["filename", "label"]
        )

        # Transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self):
        return self.annot.shape[0]

    def __getitem__(self, idx):
        img_filename, label = self.annot.loc[idx, "filename"], self.annot.loc[idx, "label"]
        img = Image.open(str(self.root.joinpath("train", img_filename)))
        img = self.transform(img)
        y = self.one_hot_encoder(label)

        return img, y
    
    def one_hot_encoder(self, label):
        onehot = np.zeros(15)
        idx = CLASS_LIST.index(label)
        onehot[idx] = 1

        return onehot

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    dataloader = DataLoader(HAR(), batch_size=1, shuffle=False)
    for x, y in dataloader:
        print(x.shape, y)
        plt.imshow(x[0].permute(1, 2, 0))
        plt.show()
        break