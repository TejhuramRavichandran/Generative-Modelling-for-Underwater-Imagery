import numpy as np
import config
import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision import transforms

IMG_SIZE=config.IMG_SIZE
BATCH_SIZE=config.BATCH_SIZE
def load_dataset(raw_folder, ref_folder):
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  
        transforms.Lambda(lambda t: (t * 2) - 1)  
    ]

    data_transform = transforms.Compose(data_transforms)

    class UnderwaterImageDataset(torch.utils.data.Dataset):
        def __init__(self, raw_folder, ref_folder, transform=None):
            self.raw_folder = raw_folder
            self.ref_folder = ref_folder
            self.transform = transform
            self.raw_images = sorted(os.listdir(raw_folder)) 
            self.ref_images = sorted(os.listdir(ref_folder)) 

        def __len__(self):
            return len(self.raw_images)

        def __getitem__(self, idx):
            raw_image_path = os.path.join(self.raw_folder, self.raw_images[idx])
            ref_image_path = os.path.join(self.ref_folder, self.ref_images[idx])

            raw_image = Image.open(raw_image_path).convert('RGB')
            ref_image = Image.open(ref_image_path).convert('RGB')

            if self.transform:
                raw_image = self.transform(raw_image)
                ref_image = self.transform(ref_image)

            return raw_image, ref_image

    train_dataset = UnderwaterImageDataset(raw_folder, ref_folder, transform=data_transform)
    return DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)




