import os
import uuid
import sys
import time
import pandas as pd
import shutil
# from azure.storage.blob import BlockBlobService, PublicAccess
from azure.storage.blob import BlobServiceClient, PublicAccess

def connector_fn():
    try:
        account_name = 'sessiontrainingstorage'
        account_url = 'https://'+account_name+'.blob.core.windows.net'
        account_key = 'HS1Z8fmCdNHt314wk+CjpC7DlIFQfRp5AxbqWO+ZCH6c0zSkVMJFPlri8YOIoFiaKRIPo+mQ2BvwjAac5TXOIw=='
        blob_service_client = BlobServiceClient(account_url, account_key)

        container_name = 'sessionbatterydata'
    
        ## create container name
        # blob_service_client.create_container(container_name)

        ## set the permission so the blobs are public
        # blob_service_client.set_container_acl(
        #                         container_name, public_access=PublicAccess.Container
        # )

    except Exception as e:
        print(e)

    return blob_service_client, container_name


def checkAndDeleteData(blob_service_client, container_name):
    container_client = blob_service_client.get_container_client(container_name)
    print('List of Blobs')
    generator = container_client.list_blobs()
    for blob in generator:
        # print("\t Blob name: " + blob.name)
        container_client.delete_blob(blob)
    
    print('Delete Complete')   


def Delay(t):
    print('in delay')
    time.sleep(t)
    print('delay complete')


def sendNextSessionData(blob_service_client, container_name):
    container_client = blob_service_client.get_container_client(container_name)
    path_remove = "D:/OneDrive - IIT Kanpur/Thesis/Thesis_Git/Thesis/training"
    local_path = 'D:/OneDrive - IIT Kanpur/Thesis/Thesis_Git/Thesis/training/ForSessionTraining'

    for r,d,f in os.walk(local_path):
        if f:
            for file in f:
                file_path_on_azure = os.path.join(r, file).replace(path_remove, "")
                file_path_on_azure = file_path_on_azure.replace('\\', "/")
                file_path_on_local = os.path.join(r, file)
                file_path_on_local = file_path_on_local.replace('\\', "/")

                # print(file_path_on_local)
                # print(file_path_on_azure)

                # blob_service_client.create_blob_from_path(container_name,
                #                                         file_path_on_azure,
                #                                         file_path_on_local
                #                                         )

                blob_client = container_client.get_blob_client(file_path_on_azure)

                with open(file_path_on_local, 'rb') as data:
                    blob_client.upload_blob(data)

    print('Upload Complete')

def gatherFilesToSend(sessionnum):
    ## remove folder
    # try:
    #     os.rmdir('./ForSessionTraining')
    # except OSError as error:
    #     print(error)
    shutil.rmtree('./ForSessionTraining')

    path = './ForSessionTraining'
    soh_values_dir = os.path.join(path, 'soh_values_oct12')
    images_dir = os.path.join(path, 'subset_image_files_oct12_20cycles')

    try: 
        os.mkdir(path) 
    except OSError as error: 
        print(error)
    
    try: 
        os.mkdir(soh_values_dir) 
    except OSError as error: 
        print(error)  

    try: 
        os.mkdir(images_dir) 
        bat_dir = os.path.join(images_dir,'RW7')
        os.mkdir(bat_dir)
        os.mkdir(os.path.join(bat_dir, 'wavelet_images'))
    except OSError as error: 
        print(error)  

    df = pd.read_csv('./Database/subset_image_files_oct12_20cycles/test_file_soh_multi_input.csv')
    file_len = len(df)
    partition = file_len // 10
    if sessionnum == 9:
        df = df[sessionnum*partition : ]
    else:
        df = df[sessionnum*partition : (sessionnum+1)*partition]

    # print(df)
    src_path = './Database/subset_image_files_oct12_20cycles/RW7/wavelet_images/'
    dest_path = './ForSessionTraining/subset_image_files_oct12_20cycles/RW7/wavelet_images/'
    for idx in range(len(df)):
        voltage_img_filename = df.loc[idx+partition*sessionnum, 'voltage_filenames']
        current_img_filename = df.loc[idx+partition*sessionnum, 'current_filenames']

        voltage_img_filename = voltage_img_filename.split('/')[-1]
        current_img_filename = current_img_filename.split('/')[-1]

        # print(voltage_img_filename)
        # print(current_img_filename)

        # src_path
        shutil.copyfile(src_path+voltage_img_filename, dest_path+voltage_img_filename)
        shutil.copyfile(src_path+current_img_filename, dest_path+current_img_filename)

    df = df.reset_index()
    print(df)
    df.to_csv('./ForSessionTraining/subset_image_files_oct12_20cycles/session_file_soh_multi_input.csv')

    

def send_data(blob_service_client, container_name):
    
    for i in range(10):
        print('Running Iteration num -- ', i)
        checkAndDeleteData(blob_service_client, container_name)
        Delay(30)
        gatherFilesToSend(i)
        sendNextSessionData(blob_service_client, container_name)

        Delay(60*15)
        print('\n')


def run_sample():
    try:
        account_name = 'sessiontrainingstorage'
        account_url = 'https://sessiontrainingstorage.blob.core.windows.net'
        account_key = 'HS1Z8fmCdNHt314wk+CjpC7DlIFQfRp5AxbqWO+ZCH6c0zSkVMJFPlri8YOIoFiaKRIPo+mQ2BvwjAac5TXOIw=='
        # blob_service_client = BlockBlobService(account_name, account_key)
        blob_service_client = BlobServiceClient(account_url, account_key)

        container_name = 'sessionbatterydata'
    
        ## create container name
        # blob_service_client.create_container(container_name)
        container_client = blob_service_client.get_container_client(container_name)

        ## set the permission so the blobs are public
        # blob_service_client.set_container_acl(
        #                         container_name, public_access=PublicAccess.Container
        # )

        path_remove = "D:/OneDrive - IIT Kanpur/Thesis/Thesis_Git/Thesis/training"
        local_path = 'D:/OneDrive - IIT Kanpur/Thesis/Thesis_Git/Thesis/training/ForSessionTraining'

        for r,d,f in os.walk(local_path):
            if f:
                for file in f:
                    file_path_on_azure = os.path.join(r, file).replace(path_remove, "")
                    file_path_on_azure = file_path_on_azure.replace('\\', "/")
                    file_path_on_local = os.path.join(r, file)
                    file_path_on_local = file_path_on_local.replace('\\', "/")

                    # print(file_path_on_local)
                    # print(file_path_on_azure)

                    # blob_service_client.create_blob_from_path(container_name,
                    #                                         file_path_on_azure,
                    #                                         file_path_on_local
                    #                                         )

                    blob_client = container_client.get_blob_client(file_path_on_azure)

                    with open(file_path_on_local, 'rb') as data:
                        blob_client.upload_blob(data)
        
        # print('List of Blobs')
        # generator = blob_service_client.list_blobs(container_name)
        # for blob in generator:
        #     print("\t Blob name: " + blob.name)


        # temp_file_local = 
        # temp_file_azure = 
        # blob_service_client.create_blob_from_path(container_name,
                                                # file_path_on_azure,
                                                # temp_file_local
                                                # )
        print('Upload compelete')

    except Exception as e:
        print(e)


def delete_blob():
    try:
        account_name = 'sessiontrainingstorage'
        account_url = 'https://sessiontrainingstorage.blob.core.windows.net'
        account_key = 'HS1Z8fmCdNHt314wk+CjpC7DlIFQfRp5AxbqWO+ZCH6c0zSkVMJFPlri8YOIoFiaKRIPo+mQ2BvwjAac5TXOIw=='
        # blob_service_client = BlockBlobService(account_name, account_key)
        blob_service_client = BlobServiceClient(account_url, account_key)

        container_name = 'sessionbatterydata'

        # container_client = blob_service_client.get_container_client(container_name)
        ## create container name
        # blob_service_client.create_container(container_name)

        ## set the permission so the blobs are public
        # blob_service_client.set_container_acl(
        #                         container_name, public_access=PublicAccess.Container
        # )
        
        container_client = blob_service_client.get_container_client(container_name)

        print('List of Blobs')
        generator = container_client.list_blobs()
        for blob in generator:
            print("\t Blob name: " + blob.name)
            # blob_service_client.delete_blob(container_name, blob)
            container_client.delete_blob(blob)

        # my_blobs = container_client.list_blobs(name_starts_with="my_blob")
        # container_client.delete_blobs(*generator)

        # blob_service_client.delete_container(container_name)

    except Exception as e:
        print(e)


# Main method
if __name__ == '__main__':
    # delete_blob()
    # run_sample()

    blob_service_client, container_name = connector_fn()

    send_data(blob_service_client, container_name)

    # gatherFilesToSend(0)


        