import os
import sys
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
# import torchvision.transforms as transforms
from torchvision import transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader

## import azure package
from azureml.core import Workspace, Dataset

## import Classes
from classes import CustomDataset, AlexNet_multi_input
from train_lib import generate_avg_soh_values, generate_filename_soh_pair, train

# dir(models)
from PIL import Image
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_percentage_error

from copy import deepcopy

## import variables from other files
import config, azureconfig

# from google.colab import drive

# torch.cuda.empty_cache()

class CustomDataset(Dataset):    
    def __init__(self, file_path, transform = None):
        
        self.transform = transform
        self.file_soh = pd.read_csv(file_path)
        
    def __len__(self):
        return len(self.file_soh)
    
    def __getitem__(self, idx):
        voltage_img_filename = self.file_soh.loc[idx, 'voltage_filenames']
        current_img_filename = self.file_soh.loc[idx, 'current_filenames']
        # temperature_img_filename = self.file_soh.loc[idx, 'temperature_filenames']
        soh_val = self.file_soh.loc[idx, 'soh_values']
        
        voltage_img = Image.open(voltage_img_filename).convert('RGB')
        current_img = Image.open(current_img_filename).convert('RGB')
        # temperature_img = Image.open(temperature_img_filename).convert('RGB')
        voltage_img_T = self.transform(voltage_img)
        current_img_T = self.transform(current_img)
        # temperature_img_T = self.transform(temperature_img)

        # print(current_img_T.shape)
        
        return (voltage_img_T, current_img_T), soh_val



def main():
    torch.manual_seed(0)

    # drive.mount('/content/drive')

    global device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print('Device -- ', device)

    # base_path = "/content/drive/MyDrive/From PC/Model/"
    # GLOBAL base_path = './'

    ## Downloading Dataset
    print("Downloading Data from AZ Storage")
    init()
    print("Download Complete")

    ## call train process
    train_process()


def init():
    workspace = Workspace(azureconfig.subscription_id, 
                            azureconfig.resource_group, 
                            azureconfig.workspace_name)

    dataset = Dataset.get_by_name(workspace,
                                    name = azureconfig.datasetname)

    dataset.download(target_path = '.', overwrite=True)


def train_process():

    generate_avg_soh_values(config.bat_names)
    generate_avg_soh_values(config.test_bat_names)

    generate_filename_soh_pair(config.bat_names,
                                config.base_path+'ToAzure/subset_image_files_oct12_20cycles/file_soh_multi_input.csv')
    generate_filename_soh_pair(config.test_bat_names,
                                config.base_path+'ToAzure/subset_image_files_oct12_20cycles/test_file_soh_multi_input.csv')

    train_data = CustomDataset(config.base_path+'ToAzure/subset_image_files_oct12_20cycles/file_soh_multi_input.csv', config.transform)
    test_data = CustomDataset(config.base_path+'ToAzure/subset_image_files_oct12_20cycles/test_file_soh_multi_input.csv', config.transform)

    dataloader = DataLoader(train_data, batch_size = config.train_batch_size, shuffle = False)

    model_multi_input = AlexNet_multi_input()
    model_multi_input = model_multi_input.float()

    train(model_multi_input, 2, 1e-4, device, dataloader)

    print('train process complete')
    print('and the device is -- ', device)

    print('Saving Model')
    file_path = config.base_path + '../outputs/azure_devops_test.pkl'
    torch.save(model_multi_input, file_path)
    print('Save Complete')



if __name__ == '__main__':
    main()