import os
import random
import numpy as np
import torch
from torch import optim
from torch import nn
from conf import Paths
import datasetup as dst
import model as m
import engine


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
paths = Paths()

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

if __name__ == "__main__":
    epochs = 1000000
    lr = 1e-3
    ks = 5

    model = m.ConstNet(ks=ks, inch=3, res_ch=10, num_cls=33, dev=dev)
    opt = optim.SGD(params=model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    data_loader, test_loader = dst.create_laoder(data_path=paths.server_data_path, train_precent=0.8, batch_size=128, nw=22)
    model.to(dev)
    model.train()
    for epoch in range(epochs):
        train_loss = engine.train_step(model=model, opt=opt, criterion=criterion, loader=data_loader, dev=dev)
        torch.save(model.state_dict(), os.path.join(paths.model, f'ckpoint_{epoch}.pt'))
        print(train_loss)
