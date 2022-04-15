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
from azureml.core import Workspace, Dataset, Datastore

## import Classes
from classes import CustomDataset, AlexNet_multi_input
from train_lib import generate_avg_soh_values, generate_filename_soh_pair, train
from train_lib import session_training, session_generate_filename_soh_pair

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

class SessionDataset(Dataset):    
    def __init__(self, file_path, session ,transform = None):
        
        self.transform = transform
        self.session = session
        self.file_soh = pd.read_csv(file_path)
        self.length = len(self.file_soh)

        print('length of file soh -- ', self.length)
        self.partition = self.length // 10
        print('partition of file soh -- ', self.partition)

        if session == 9:
          self.file_soh = self.file_soh[self.session*self.length : ]
        else:
          self.file_soh = self.file_soh[self.session*self.partition : (self.session+1)*self.partition]

        self.file_soh = self.file_soh.reset_index()
        print(len(self.file_soh))
        
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
        

        ## change return statement based on images to pass as input
        # return (voltage_img_T, current_img_T, temperature_img_T), soh_val

        return (voltage_img_T, current_img_T), soh_val


def main():
    torch.manual_seed(0)

    global device

    device = torch.device('cpu')
    print('Device -- ', device)

    ## Downloading Dataset
    print("Downloading Data from AZ Storage")
    session_data_init()
    print('Data Download Complete')

    os.rename('%2FForSessionTraining', 'ForSessionTraining')    

    ## call session train process
    session_train_process()
    print('training complete')

    ## register model
    ## save model in training model directory
    

    ## delete all files from the container


def session_data_init():
    workspace = Workspace(azureconfig.subscription_id, 
                            azureconfig.resource_group, 
                            azureconfig.workspace_name)
    # print('i am here')
    web_path = 'https://sessiontrainingstorage.blob.core.windows.net/sessionbatterydata/'
    session_dataset = Dataset.File.from_files(path = web_path)
    # print('i am here')
    # dataset = Dataset.get_by_name(workspace,
                                    # name = azureconfig.session_datasetname)
    # print('i am here')
    session_dataset.download(target_path = './', overwrite=True)

def session_train_process():

    # generate_avg_soh_values(config.session_bat_names)

    # session_generate_filename_soh_pair(config.session_bat_names,
    #                             config.base_path + '%2FForSessionTraining/subset_image_files_oct12_20cycles/session_file_soh_multi_input.csv')

    # session_data = SessionDataset(config.base_path + '%2FForSessionTraining/subset_image_files_oct12_20cycles/session_file_soh_multi_input.csv',0,config.transform)

    session_data = CustomDataset(config.base_path + '%2FForSessionTraining/subset_image_files_oct12_20cycles/session_file_soh_multi_input.csv',config.transform)

    session_dataloader = DataLoader(session_data, batch_size=32, shuffle=False)

    model_multi_input_saved = AlexNet_multi_input()
    model_multi_input_saved = model_multi_input_saved.float()
    
    ## load model
    temp_file_path = "../download_models/FSLL_march2_pretrain_set_9_10_11_finetune_rw1.pkl"
    model_multi_input_saved = torch.load(temp_file_path, map_location = torch.device(device))

    session_training(model_multi_input_saved, 20, device, 
                    session_dataloader, learning_rate = 0.0001, hyperparameter = 20)

    file_path = "../trained_models/FSLL_march2_pretrain_set_9_10_11_finetune_rw1.pkl"
    torch.save(model_multi_input_saved, file_path)
    

if __name__ == '__main__':
    main()