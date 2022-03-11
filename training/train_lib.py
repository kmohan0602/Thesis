import pandas as pd
import config
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

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
