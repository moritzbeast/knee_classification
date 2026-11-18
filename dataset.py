
from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
from torchvision import tv_tensors

def read_img(path_file):
    xray = cv2.imread(path_file, cv2.IMREAD_GRAYSCALE)

    return xray

def load_xray(path_file):
    # 1 x H x W, scaled to 0-1, marked as image so the v2 transforms know how to treat it
    xray = torch.from_numpy(read_img(path_file)).float().unsqueeze(0) / 255.0

    return tv_tensors.Image(xray)

def compute_mean_std(dataset):
    # mean and std over all pixels of the given images (scaled to 0-1)
    imgs = np.stack([read_img(path) for path in dataset["image_path"]]) / 255.0

    return imgs.mean(), imgs.std()

class knee_xray_dataset(Dataset):
    def __init__(self, dataset, mean, std, transform=None):
        self.dataset = dataset
        self.mean = mean
        self.std = std
        self.transform = transform

    def __len__(self):
        return (len(self.dataset))

    def __getitem__(self, index):
        img = load_xray(self.dataset["image_path"].iloc[index])

        # augmentation on the 0-1 image, before normalization
        if self.transform is not None:
            img = self.transform(img)

        img = (img - self.mean) / self.std
        img = img.repeat(3, 1, 1)
        name = (self.dataset["image_path"].iloc[index]).split("/")[-1]
        target = self.dataset["label"].iloc[index]

        res = {
            "xray": img,
            "name": name,
            "target": target
        }

        return res
        

