import os

from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms



transform = transforms.Compose(
    transforms=[
        transforms.ToTensor(), transforms.CenterCrop(size=(480, 800))
    ]
)

# dataset = ImageFolder(root='/media/hasadi/myDrive/Datasets/visionDataset/VISION/iframe_720x1280', transform=transform)



def create_laoder(data_path, train_precent, batch_size, nw=1):
    dataset = ImageFolder(root=data_path, transform=transform)
    data_size = len(dataset)
    train_size = int(train_precent*data_size)
    test_size = data_size - train_size
    train_set, test_set = random_split(dataset=dataset, lengths=[train_size, test_size])

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=nw, pin_memory=True, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, num_workers=nw, pin_memory=True, shuffle=True)

    return train_loader, test_loader


if __name__ == "__main__":
    local_path = '/media/hasadi/myDrive/Datasets/visionDataset/VISION/iframe_720x1280'
    train_loader, test_loader = create_laoder(data_path=local_path, batch_size=128, train_precent=0.8)
    X, Y = next(iter(test_loader))
    print(X.shape, Y.shape)
    print(Y)