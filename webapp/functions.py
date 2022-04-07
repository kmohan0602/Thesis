import os
import sys
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
from torchvision import models
from torch.utils.data import Dataset, DataLoader

from PIL import Image

from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_percentage_error
from copy import deepcopy


def predict_helper(model, testdataloader):
    model.eval()
    predictions = []
    actual = []
    device = torch.device('cpu')
    for i, (images, target_soh) in enumerate(testdataloader):
        
        # images = images.to(device).float()
        img1 = images[0].to(device).float()
        img2 = images[1].to(device).float()
        # img3 = images[2].to(device).float()
        target_soh = target_soh.to(device).float()

        # target_soh_flatten = target_soh.flatten

        # actual.append(target_soh.item())
        actual.extend(target_soh.cpu().data.numpy().flatten())


        pred_soh = model(img1, img2)
        
#         print(pred_soh.item())
        
        # predictions.append(pred_soh.item())
        predictions.extend(pred_soh.cpu().data.numpy().flatten())

        # plt.plot(predictions)
        # plt.plot(actual)
        
    mse = mean_squared_error(actual, predictions)
    mape = mean_absolute_percentage_error(actual, predictions)
        
    print('mean square error -- ',mse)
    print('mean absolute percentage error -- ',mape)