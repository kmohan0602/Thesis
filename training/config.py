from torchvision import transforms

base_path = './'
train_batch_size = 64

bat_names = [
            'RW1',
            # 'RW2',
            # 'RW7',
            # 'RW8',
            # 'RW3',
            # 'RW4',
            # 'RW5',
            # 'RW6',
            # 'RW9',
            # 'RW10',
            # 'RW11',
            # 'RW12',
            ]

test_bat_names = [
                'RW1'
                # 'RW8',
                # 'RW6',
                # 'RW12',
                # 'RW11'
                # 'RW6',
                ]

session_bat_names = [
                'RW7'
                ]

transform = transforms.Compose([
    transforms.Resize((256,256)),
#     transforms.CenterCrop(224), ## not sure about this
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485,0.456,0.406],
        std = [0.229,0.224,0.225]
    )
])