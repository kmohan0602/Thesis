import pandas as pd
import numpy as np
import config
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from copy import deepcopy

def generate_avg_soh_values(bat_names):
    for x in bat_names:
        averages = []
        # soh_file_path = base_path+'../Preprocessing/soh_values_oct12/soh_values_'+x+'.csv'
        # soh_file_path = config.base_path+'../data/soh_values_oct12/soh_values_'+x+'.csv'    
        soh_file_path = config.base_path+'ToAzure/soh_values_oct12/soh_values_'+x+'.csv'    

        soh_values = pd.read_csv(soh_file_path)
        print('len of soh values -- ', x,' ----', len(soh_values))

        first_col = True
        for i in range(0, len(soh_values), 20): ## this 20 is important incase if we have to change the gap
            l = i+1
            if(first_col):
                l = i
                first_col = False
            r = i+20+1
            temp_list = []
            for j in range(l, r):
                if(j >=len(soh_values)):
                    break
                temp_list.append(soh_values.loc[j,'0'])

            # print(len(temp_list))
            averages.append(sum(temp_list)/len(temp_list))

        print(len(averages), '---- ', x)
        averages_df = pd.DataFrame(averages)
        # averages_df.to_csv(base_path+'../Preprocessing/subset_image_files_oct12_20cycles/'+x+'/soh_values_avg_'+x+'.csv')
        # averages_df.to_csv(config.base_path+'../data/subset_image_files_oct12_20cycles/'+x+'/soh_values_avg_'+x+'.csv')
        averages_df.to_csv(config.base_path+'ToAzure/subset_image_files_oct12_20cycles/'+x+'/soh_values_avg_'+x+'.csv')


def generate_filename_soh_pair(bat_names, output_path):
    voltage_filenames = []
    current_filenames = []
    temperature_filenames = []
    soh_values_toexcel = []

    for x in bat_names:
        # soh_file_path = base_path+'../Preprocessing/soh_values_oct12/soh_values_'+x+'.csv'
        # soh_file_path = config.base_path+'../data/soh_values_oct12/soh_values_'+x+'.csv'
        soh_file_path = config.base_path+'ToAzure/soh_values_oct12/soh_values_'+x+'.csv'

        soh_values = pd.read_csv(soh_file_path)
        print('len of soh values -- ', x,' ----', len(soh_values))

        length = len(soh_values)

        test =[]
        voltage_temp_filenames = []
        current_temp_filenames = []
        temperature_temp_filenames = []
        temp_soh_values = []

        count = 0
        # soh_avg_file = config.base_path+'../data/subset_image_files_oct12_20cycles/'+x+'/soh_values_avg_'+x+'.csv'
        soh_avg_file = config.base_path+'ToAzure/subset_image_files_oct12_20cycles/'+x+'/soh_values_avg_'+x+'.csv'
        soh_avg = pd.read_csv(soh_avg_file)

        for i in range(0,length,20):
            l = i
            r=i+20
            # voltage_temp_filenames.append(config.base_path+'../data/subset_image_files_oct12_20cycles/'+x+'/wavelet_images/'+x+'_'+str(l)+'_'+str(r)+'_voltage.png')
            # current_temp_filenames.append(config.base_path+'../data/subset_image_files_oct12_20cycles/'+x+'/wavelet_images/'+x+'_'+str(l)+'_'+str(r)+'_current.png')
            # # temperature_temp_filenames.append(base_path+'../data/subset_image_files_oct12_20cycles/'+x+'/wavelet_images/'+x+'_'+str(l)+'_'+str(r)+'_temp.png')
            voltage_temp_filenames.append(config.base_path+'ToAzure/subset_image_files_oct12_20cycles/'+x+'/wavelet_images/'+x+'_'+str(l)+'_'+str(r)+'_voltage.png')
            current_temp_filenames.append(config.base_path+'ToAzure/subset_image_files_oct12_20cycles/'+x+'/wavelet_images/'+x+'_'+str(l)+'_'+str(r)+'_current.png')
            # temperature_temp_filenames.append(base_path+'../data/subset_image_files_oct12_20cycles/'+x+'/wavelet_images/'+x+'_'+str(l)+'_'+str(r)+'_temp.png')
            temp_soh_values.append(soh_avg.loc[count, '0'])
            count += 1
        print(x)    
        print(len(voltage_temp_filenames))
        print(len(current_temp_filenames))
        # print(len(temperature_temp_filenames))
        print(len(temp_soh_values))

        voltage_filenames.extend(voltage_temp_filenames[:-1])
        current_filenames.extend(current_temp_filenames[:-1])
        # temperature_filenames.extend(temperature_temp_filenames[:-1])
        soh_values_toexcel.extend(temp_soh_values[:-1])


    print(len(voltage_filenames))

    file_soh_dict = {'voltage_filenames':voltage_filenames, 'current_filenames':current_filenames, 'soh_values':soh_values_toexcel}
    file_soh_df = pd.DataFrame(file_soh_dict)
    file_soh_df.to_csv(output_path)


def session_generate_filename_soh_pair(bat_names, output_path):
    voltage_filenames = []
    current_filenames = []
    temperature_filenames = []
    soh_values_toexcel = []

    for x in bat_names:
        # soh_file_path = base_path+'../Preprocessing/soh_values_oct12/soh_values_'+x+'.csv'
        # soh_file_path = config.base_path+'../data/soh_values_oct12/soh_values_'+x+'.csv'
        soh_file_path = config.base_path+'%2FForSessionTraining/soh_values_oct12/soh_values_'+x+'.csv'

        soh_values = pd.read_csv(soh_file_path)
        print('len of soh values -- ', x,' ----', len(soh_values))

        length = len(soh_values)

        test =[]
        voltage_temp_filenames = []
        current_temp_filenames = []
        temperature_temp_filenames = []
        temp_soh_values = []

        count = 0
        # soh_avg_file = config.base_path+'../data/subset_image_files_oct12_20cycles/'+x+'/soh_values_avg_'+x+'.csv'
        soh_avg_file = config.base_path+'%2FForSessionTraining/subset_image_files_oct12_20cycles/'+x+'/soh_values_avg_'+x+'.csv'
        soh_avg = pd.read_csv(soh_avg_file)

        for i in range(0,length,20):
            l = i
            r=i+20
            # voltage_temp_filenames.append(config.base_path+'../data/subset_image_files_oct12_20cycles/'+x+'/wavelet_images/'+x+'_'+str(l)+'_'+str(r)+'_voltage.png')
            # current_temp_filenames.append(config.base_path+'../data/subset_image_files_oct12_20cycles/'+x+'/wavelet_images/'+x+'_'+str(l)+'_'+str(r)+'_current.png')
            # # temperature_temp_filenames.append(base_path+'../data/subset_image_files_oct12_20cycles/'+x+'/wavelet_images/'+x+'_'+str(l)+'_'+str(r)+'_temp.png')
            voltage_temp_filenames.append(config.base_path+'%2FForSessionTraining/subset_image_files_oct12_20cycles/'+x+'/wavelet_images/'+x+'_'+str(l)+'_'+str(r)+'_voltage.png')
            current_temp_filenames.append(config.base_path+'%2FForSessionTraining/subset_image_files_oct12_20cycles/'+x+'/wavelet_images/'+x+'_'+str(l)+'_'+str(r)+'_current.png')
            # temperature_temp_filenames.append(base_path+'../data/subset_image_files_oct12_20cycles/'+x+'/wavelet_images/'+x+'_'+str(l)+'_'+str(r)+'_temp.png')
            temp_soh_values.append(soh_avg.loc[count, '0'])
            count += 1
        print(x)    
        print(len(voltage_temp_filenames))
        print(len(current_temp_filenames))
        # print(len(temperature_temp_filenames))
        print(len(temp_soh_values))

        voltage_filenames.extend(voltage_temp_filenames[:-1])
        current_filenames.extend(current_temp_filenames[:-1])
        # temperature_filenames.extend(temperature_temp_filenames[:-1])
        soh_values_toexcel.extend(temp_soh_values[:-1])


    print(len(voltage_filenames))

    file_soh_dict = {'voltage_filenames':voltage_filenames, 'current_filenames':current_filenames, 'soh_values':soh_values_toexcel}
    file_soh_df = pd.DataFrame(file_soh_dict)
    file_soh_df.to_csv(output_path)


def predict_generate_filename_soh_pair(bat_names, output_path):
    voltage_filenames = []
    current_filenames = []
    temperature_filenames = []
    soh_values_toexcel = []

    # for x in bat_names:
    # soh_file_path = base_path+'../Preprocessing/soh_values_oct12/soh_values_'+x+'.csv'
    # soh_file_path = config.base_path+'../data/soh_values_oct12/soh_values_'+x+'.csv'
    soh_file_path = config.test_base_path+'ForSessionTraining/soh_values_oct12/soh_values_'+bat_names+'.csv'

    soh_values = pd.read_csv(soh_file_path)
    print('len of soh values -- ', bat_names,' ----', len(soh_values))

    length = len(soh_values)

    test =[]
    voltage_temp_filenames = []
    current_temp_filenames = []
    temperature_temp_filenames = []
    temp_soh_values = []

    count = 0
    # soh_avg_file = config.base_path+'../data/subset_image_files_oct12_20cycles/'+x+'/soh_values_avg_'+x+'.csv'
    soh_avg_file = config.test_base_path+'ForSessionTraining/subset_image_files_oct12_20cycles/'+bat_names+'/soh_values_avg_'+bat_names+'.csv'
    soh_avg = pd.read_csv(soh_avg_file)

    for i in range(0,length,20):
        l = i
        r=i+20
        # voltage_temp_filenames.append(config.base_path+'../data/subset_image_files_oct12_20cycles/'+x+'/wavelet_images/'+x+'_'+str(l)+'_'+str(r)+'_voltage.png')
        # current_temp_filenames.append(config.base_path+'../data/subset_image_files_oct12_20cycles/'+x+'/wavelet_images/'+x+'_'+str(l)+'_'+str(r)+'_current.png')
        # # temperature_temp_filenames.append(base_path+'../data/subset_image_files_oct12_20cycles/'+x+'/wavelet_images/'+x+'_'+str(l)+'_'+str(r)+'_temp.png')
        voltage_temp_filenames.append(config.test_base_path+'ForSessionTraining/subset_image_files_oct12_20cycles/'+bat_names+'/wavelet_images/'+bat_names+'_'+str(l)+'_'+str(r)+'_voltage.png')
        current_temp_filenames.append(config.test_base_path+'ForSessionTraining/subset_image_files_oct12_20cycles/'+bat_names+'/wavelet_images/'+bat_names+'_'+str(l)+'_'+str(r)+'_current.png')
        # temperature_temp_filenames.append(base_path+'../data/subset_image_files_oct12_20cycles/'+x+'/wavelet_images/'+x+'_'+str(l)+'_'+str(r)+'_temp.png')
        temp_soh_values.append(soh_avg.loc[count, '0'])
        count += 1
    print(bat_names)    
    print(len(voltage_temp_filenames))
    print(len(current_temp_filenames))
    # print(len(temperature_temp_filenames))
    print(len(temp_soh_values))

    voltage_filenames.extend(voltage_temp_filenames[:-1])
    current_filenames.extend(current_temp_filenames[:-1])
    # temperature_filenames.extend(temperature_temp_filenames[:-1])
    soh_values_toexcel.extend(temp_soh_values[:-1])


    print(len(voltage_filenames))

    file_soh_dict = {'voltage_filenames':voltage_filenames, 'current_filenames':current_filenames, 'soh_values':soh_values_toexcel}
    file_soh_df = pd.DataFrame(file_soh_dict)
    file_soh_df.to_csv(output_path)


def train(model, num_epochs, learning_rate, device, dataloader):
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    
    step_loss = []
    epoch_loss = []
    
    model = model.to(device)
    model.train()
    
    for epoch in range(num_epochs):

        for i, (images, soh) in enumerate(dataloader):

#             images = images.to(device).float()
            img1 = images[0].to(device).float()
            img2 = images[1].to(device).float()
            # img3 = images[2].to(device).float()

            soh = torch.reshape(soh, (soh.shape[0], 1))
            soh = soh.to(device).float()

            # print(soh.shape)
            # sys.exit()

            optimizer.zero_grad()

            pred = model(img1, img2)

            # print(pred.shape)
            # print(soh.shape)
              
            # sys.exit()
        
            loss = criterion(pred, soh)

            # print(loss)
            # sys.exit()

            loss.backward()
            
            optimizer.step()
            step_loss.append(loss.item())

            # if i%1 ==0:
            #   print(loss.item())

        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        epoch_loss.append(loss.item())
        
    plt.plot(epoch_loss)


def session_training(model, num_epochs, device, dataloader, learning_rate, hyperparameter):
  
    # weight_list = []
    weight_list = {}
    freeze_list = {}
    previous_named_parameters = {}
    threshold_values = {}

    for n,p in model.named_parameters():
        if 'weight' in n and 'regressor' not in n:
            x = p.data.flatten()
            temp_list = []

            for temp in x:
                # weight_list.append(temp.item()**2)
                temp_list.append(temp.item()**2)

            weight_list[n] = temp_list


    # print(len(weight_list))
    # threshold_limit = len(weight_list)//70
    # weight_list = sorted(weight_list, reverse = True)
    # threshold_value = weight_list[-threshold_limit]

    for n in weight_list.keys():
        # print(len(weight_list[n]))
        threshold_limit = len(weight_list[n])//8
        weight_list[n] = sorted(weight_list[n])
        threshold_values[n] = weight_list[n][threshold_limit]

    print('threshold value - ', threshold_values)

    for n,p in model.named_parameters():
        if 'weight' in n and 'regressor' not in n:
            weight_data = p.data.cpu().numpy()

            grad_tensor = p.data.cpu().numpy()

            # grad_tensor = np.where(weight_data >= threshold_value, 0, grad_tensor)
            grad_tensor = np.where(weight_data**2 >= threshold_values[n], 0, grad_tensor)

            grad_tensor = torch.from_numpy(grad_tensor).to(device)
            freeze_list[n] = grad_tensor

    print(freeze_list.keys())

    # print(freeze_list)
    # sys.exit()


    # previous_named_parameters = model.named_parameters()
    for n,p in model.named_parameters():
        previous_named_parameters[n] = p.data

    previous_state_dict = deepcopy(model.state_dict())

    criterion = nn.MSELoss()
    l1_loss = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

    step_loss = []
    epoch_loss = []

    model = model.to(device)
    model.train()

    for epoch in range(num_epochs):
        for i, (images, soh) in enumerate(dataloader):

            # images = images.to(device).float()
            img1 = images[0].to(device).float()
            img2 = images[1].to(device).float()
            # img3 = images[2].to(device).float()

            soh = torch.reshape(soh, (soh.shape[0], 1))
            soh = soh.to(device).float()

            # print(soh.shape)
            # sys.exit()

            optimizer.zero_grad()

            # print(images.shape)
            # pred = model(images)
            pred = model(img1, img2)

            # print(pred.shape)
            # print(soh.shape)

            loss = criterion(pred, soh)

            ## implement l1 loss
            wt_loss = 0
            bias_loss = 0
            # for (n1, p1),(n2,p2) in zip(model.named_parameters(), previous_named_parameters):
            #   if 'weight' in n1 and 'weight' in n2:
            #     wt_loss += l1_loss(p1.data, p2.data)

            # for n, p in model.named_parameters():
            #   wt_loss += l1_loss(p.data, previous_named_parameters[n])


            for temp in model.state_dict():
                if 'weight' in temp and 'fc' not in temp:
                    wt_loss += l1_loss(model.state_dict()[temp], previous_state_dict[temp])
                if 'bias' in temp and 'fc' not in temp:
                    bias_loss += l1_loss(model.state_dict()[temp], previous_state_dict[temp])

            print('loss -- ', loss)
            # sys.exit()
            
            print('weight_loss - ', wt_loss)
            print('bias_loss - ', bias_loss)
            
            loss += hyperparameter * wt_loss
            loss.backward()

            for n, p in model.named_parameters():
                if n in freeze_list:
                    freeze_tensor = freeze_list[n].data.cpu().numpy()
                    grad_tensor = p.grad.data.cpu().numpy()
                    grad_tensor = np.where(freeze_tensor == 0, 0, grad_tensor)

                    # p.grad.data = torch.where(freeze_list[n] == 0, 0, p.grad)
                    p.grad.data = torch.from_numpy(grad_tensor).to(device)

                if 'bias' in n:
                    p.requires_grad = False

                if 'regressor' in n:
                    p.requires_grad = False

            # previous_named_parameters = model.named_parameters()
            # for n,p in model.named_parameters():
            #   previous_named_parameters[n] = p.data

            previous_state_dict = deepcopy(model.state_dict())

            optimizer.step()      
            
            step_loss.append(loss.item())

            # if i%1 ==0:
            #   print(loss.item())

        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        epoch_loss.append(loss.item())

    plt.plot(epoch_loss)

