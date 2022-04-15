import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, make_response
# import pickle
import socket
import encodings
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import io
from datetime import datetime

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

sys.path.append('./../training')

# from training.train_lib import generate_filename_soh_pair
from train_lib import predict_generate_filename_soh_pair
from functions import predict_helper
# from training.config import transform
from config import transform

HOST = '127.0.0.1'
PORT = 65432

global soh_output

app = Flask(__name__, template_folder='templates')
# model = pickle.load(open('../outputs/azure_devops_test.pkl', 'rb'))

# global device

@app.route('/')
def home():
    # device = torch.device('cpu')
    template_data = {
        'soh' : 0,
        'soc' : 0,
        'time' : 0
    }
    return render_template('index.html', **template_data)

@app.route('/predict',methods = ['POST','GET'])
def predict():
    for x in request.form.values():
        print(x)
    print('kaise ho')
    print(request.values.get('battery'))
    bat_name = request.values.get('battery')

    print(type(bat_name))

    device = torch.device('cpu')

    # return render_template('home.html', prediction_text="AQI for Jaipur {}".format(prediction[0]))

    predict_generate_filename_soh_pair(bat_name,
                            '../training/ForWebApp/subset_image_files_oct12_20cycles/test_file_soh_multi_input.csv')
    test_data = CustomDataset('../training/ForWebApp/subset_image_files_oct12_20cycles/test_file_soh_multi_input.csv',
                                 transform)
    testdataloader = DataLoader(test_data, batch_size =1, shuffle=False)

    model_multi_input = AlexNet_multi_input().float()
    model_path = '../outputs/FSLL_march2_pretrain_set_9_10_11_finetune_rw1.pkl'
    model_multi_input = torch.load(model_path, map_location=torch.device(device))

    soh_output = predict_helper(model_multi_input, testdataloader)
    print(soh_output)
    plt.plot(soh_output)
    plt.savefig('./static/images/soh_plot.png')

    ## add code for graph


    ## get data from bms
    ## send soh value
    soh_value = 90
    soh_value = soh_output[-1]
    soc_value = 20
    soc_value  = run_websocket_server(soh_value)
    print('SOC value -- ', soc_value)


    now = datetime.now()
    time = now.strftime("%m/%d/%Y, %H:%M:%S")
    
    template_data = {
        'soh_value' : soh_value,
        'soc_value' : soc_value,
        'time' : time,
        'url' : './static/images/soh_plot.png'
    }


    return render_template('index.html', **template_data)


@app.route('/plot/SOH')
def plot_soh():
    # predictions = get_data()
    predictions = soh_output
    print('Predictions')
    print(predictions)
    ys = predictions
    fig = Figure()
    axis = fig.add_subplot(1,1,1)
    axis.set_title("Battery SOH in Ah")
    axis.set_xlabel('Cycles')
    axis.set_ylabel('Ah')
    axis.grid(True)
    xs = range(len(predictions))
    axis.plot(xs, ys)
    canvas = FigureCanvas(fig)
    output = io.BytesIO()
    canvas.print_png(output)
    response = make_response(output.getvalue())
    response.mimetype = 'image/png'
    return response


def run_websocket_server(soh_value):
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print("Server Started waiting for client to connect")
        s.bind((HOST, PORT))
        s.listen(5)
        conn, addr = s.accept()

        # print(conn)
        # print(addr)

        with conn:
            print('Connected by', addr)
            while True:
                print('Receiving SOC value')
                data = conn.recv(1024).decode('utf-8')

                print('Sending SOH value')
                if str(data)=="Data":
                    print("Ok Sending Data")

                    soh_value = str(soh_value)
                    soh_value_encoded = soh_value.encode('utf-8')

                    conn.sendall(soh_value_encoded)

                elif str(data) == "Quit":
                    print("Shutting Server")
                    break
                    
                if not data:
                    break
                else:
                    pass

    return data


## Dataset Class
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


## Model Class
class AlexNet_multi_input(nn.Module):
    def __init__(self):
        
        super().__init__()
        
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
#         self.conv2 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)

        # self.net1 = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
        #     nn.ReLU(),
        #     nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
        #     nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
        #     nn.Conv2d(96, 256, 5, padding=2),  # (b x 256 x 27 x 27)
        #     nn.ReLU(),
        #     nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
        #     nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
        #     nn.Conv2d(256, 384, 3, padding=1),  # (b x 384 x 13 x 13)
        #     nn.ReLU(),
        #     nn.Conv2d(384, 384, 3, padding=1),  # (b x 384 x 13 x 13)
        #     nn.ReLU(),
        #     nn.Conv2d(384, 256, 3, padding=1),  # (b x 256 x 13 x 13)
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
        # )
        
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
            # nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5, stride=2),  # (b x 96 x 55 x 55)
            # nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1),  # (b x 96 x 55 x 55)
            # nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.net2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
            # nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5, stride=2),  # (b x 96 x 55 x 55)
            # nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1),  # (b x 96 x 55 x 55)
            # nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # self.net3 = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
        #     # nn.BatchNorm2d(96),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5, stride=2),  # (b x 96 x 55 x 55)
        #     # nn.BatchNorm2d(96),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1),  # (b x 96 x 55 x 55)
        #     # nn.BatchNorm2d(96),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        # )



    
        # self.net2 = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
        #     nn.ReLU(),
        #     nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
        #     nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
        #     nn.Conv2d(96, 256, 5, padding=2),  # (b x 256 x 27 x 27)
        #     nn.ReLU(),
        #     nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
        #     nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
        #     nn.Conv2d(256, 384, 3, padding=1),  # (b x 384 x 13 x 13)
        #     nn.ReLU(),
        #     nn.Conv2d(384, 384, 3, padding=1),  # (b x 384 x 13 x 13)
        #     nn.ReLU(),
        #     nn.Conv2d(384, 256, 3, padding=1),  # (b x 256 x 13 x 13)
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
        # )
        
#         self.regressor = nn.Sequential(
# #             nn.Dropout(p=0.2,inplace=True),
#             nn.Linear(in_features=18432, out_features=8192),
#             nn.ReLU(), ## have doubt about the activation fn
#             nn.Dropout(p=0.2,inplace=False),
#             nn.Linear(in_features=8192, out_features=2048),
#             nn.ReLU(), ## have doubt about the activation fn
#             nn.Dropout(p=0.2,inplace=False),
#             nn.Linear(in_features=2048, out_features=1),
#         )

        ## simple model with 2 images
        self.regressor = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(), ## have doubt about the activation fn
            nn.Dropout(p=0.2,inplace=False),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(), ## have doubt about the activation fn
            nn.Dropout(p=0.2,inplace=False),
            nn.Linear(in_features=256, out_features=1),
        )

        ## simple model with 3 images
        # self.regressor = nn.Sequential(
        #     nn.Linear(in_features=1536, out_features=512),
        #     nn.ReLU(), ## have doubt about the activation fn
        #     nn.Dropout(p=0.2,inplace=False),
        #     nn.Linear(in_features=512, out_features=256),
        #     nn.ReLU(), ## have doubt about the activation fn
        #     nn.Dropout(p=0.2,inplace=False),
        #     nn.Linear(in_features=256, out_features=1),
        # )



        
    def forward(self, x, y):
        
        # print(x.size())
        # print(y.size())
        
        # sys.exit()

        x = self.net1(x)
        y = self.net2(y)
        # temperature = self.net3(temperature)
        
        N,_,_,_ = x.size()

        # print(x.size())
        # print(y.size())
        
        # sys.exit()
        
        x = x.view(N,-1)
        y = y.view(N,-1)
        # temperature = temperature.view(N, -1)
        
        z = torch.cat((x,y), 1)
        
        # print("z shape")
        # print(z.shape)
        # sys.exit()

        output = self.regressor(z)
        
        return output

if __name__ == '__main__':
    global device
    device = torch.device('cpu')
    app.run()