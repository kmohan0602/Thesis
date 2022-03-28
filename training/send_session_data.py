import os
import uuid
import sys
from azure.storage.blob import BlockBlobService, PublicAccess

def run_sample():
    try:
        account_name = 'sessiontrainingstorage'
        account_key = 'HS1Z8fmCdNHt314wk+CjpC7DlIFQfRp5AxbqWO+ZCH6c0zSkVMJFPlri8YOIoFiaKRIPo+mQ2BvwjAac5TXOIw=='
        blob_service_client = BlockBlobService(account_name, account_key)

        container_name = 'sessionbatterydata'
    
        ## create container name
        blob_service_client.create_container(container_name)

        ## set the permission so the blobs are public
        blob_service_client.set_container_acl(
                                container_name, public_access=PublicAccess.Container
        )

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

                    blob_service_client.create_blob_from_path(container_name,
                                                            file_path_on_azure,
                                                            file_path_on_local
                                                            )
        
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
        account_key = 'HS1Z8fmCdNHt314wk+CjpC7DlIFQfRp5AxbqWO+ZCH6c0zSkVMJFPlri8YOIoFiaKRIPo+mQ2BvwjAac5TXOIw=='
        blob_service_client = BlockBlobService(account_name, account_key)

        container_name = 'sessionbatterydata'

        # container_client = blob_service_client.get_container_client(container_name)
        ## create container name
        # blob_service_client.create_container(container_name)

        ## set the permission so the blobs are public
        # blob_service_client.set_container_acl(
        #                         container_name, public_access=PublicAccess.Container
        # )
        
        # print('List of Blobs')
        # generator = blob_service_client.list_blobs(container_name)
        # for blob in generator:
        #     print("\t Blob name: " + blob.name)
        #     blob_service_client.delete_blob(container_name, blob)

        # my_blobs = container_client.list_blobs(name_starts_with="my_blob")
        # container_client.delete_blobs(*generator)

        blob_service_client.delete_container(container_name)

    except Exception as e:
        print(e)


# Main method
if __name__ == '__main__':
    # delete_blob()
    run_sample()



        