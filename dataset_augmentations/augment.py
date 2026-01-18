# Case 1: without having to download the data everytime the pre-processing/augmenting changes
import pandas as pd
import numpy as np
import torch, torchvision
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
from torchvision.transforms import v2
import os 

transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    ]) 

# path_to_dataset = "C:\\Users\\acer\\Downloads\\dataset" --> used as dynamic parameter in the function below
transform_augment = v2.Compose([
    v2.RandomRotation(degrees=30, expand=True, fill=0),
    v2.Resize((224,224)),
    
    v2.ToImage(),                                 # convert PIL → tensor
    v2.ToDtype(torch.float32, scale=True),        # scale to [0,1]
    v2.GaussianNoise(mean=0.0, sigma=0.01, clip=True)])

def get_dataset_and_dataloader(path_to_dataset)->list[datasets.ImageFolder, DataLoader]:
    '''Returns dataset and dataloader for specified fundus dataset path'''
    dataset = datasets.ImageFolder(root = path_to_dataset, transform = transform)
    dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)
    return dataset, dataloader


def get_augmented_dataset_and_dataloader(path_to_dataset)->list[datasets.ImageFolder, DataLoader]: 
    '''Returns augmented tensor dataset and dataloader, by taking original dataset path as input''' 
    # original dataset
    dataset = datasets.ImageFolder(root = path_to_dataset, transform = transform)
    # augmented dataset
    augmented_dataset = datasets.ImageFolder(root = path_to_dataset,transform = transform_augment) 
    # concatenating two datasets
    dataset_full = torch.utils.data.ConcatDataset([dataset, augmented_dataset])
    dataloader_full = DataLoader(dataset = dataset_full,batch_size = 64,shuffle = True)
    return dataset_full,dataloader_full 

# Case 2: expects everyone to work on pre-processed version of data by first downloading it if they don't have it
def download_dataset(dataset, dataloader, path_absolute)->str: 
    '''Downloads dataset based on ImageLoader dataset'''
    path = path_absolute
    os.makedirs(path, exist_ok = True)
    classes = dataset.datasets[0].classes  # or [1], both are same
    for batch, (images, labels) in enumerate(dataloader):
        for i in range(images.shape[0]):
            image = images[i]
            label:str = classes[labels[i].item()]
            class_dir = os.path.join(path, label)
            os.makedirs(class_dir, exist_ok = True)
            file_path = os.path.join(class_dir, f"img_{batch}_{i}.png")
            torchvision.utils.save_image(image, file_path) # saves tensors to png (make sure its unnormalized)
    return f"Dataset stored successfully. Location: {path}"

if __name__ == "__main__":
    dataset_augmented, dataloader_augmented = get_augmented_dataset_and_dataloader("C:/Users/acer/Desktop/dataset")
    download_dataset(dataset_augmented, dataloader_augmented, "C:\\Users\\acer\\Desktop\\augmented_dataset")
