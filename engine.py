import os
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader





def train_step(model:nn.Module, opt:Optimizer, criterion:nn.Module, loader:DataLoader, dev):
    epoch_loss = 0
    model.train()
    crt1 = nn.L1Loss()
    crt2 = nn.BCEWithLogitsLoss()

    for X, Y in loader:
        out, out00, out01, out02 = model(X.to(dev))
        y = torch.zeros_like(out00[0])
        loss1 = crt1(out00[1], y) + crt1(out01[1], y) + crt1(out02[1], y)
        loss2 = crt2(out00[0]/100000, y) + crt2(out01[0]/1000000, y) + crt2(out02[0]/1000000, y)
        loss3 = criterion(out, Y.to(dev))
        loss = loss1 + loss2 + loss3
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss += loss.item()

    return epoch_loss


def eval_step(model:nn.Module, loader:DataLoader, dev):
    epoch_loss = 0
    model.eval()
    constlayer = model.constlayer
    print(constlayer.weight)
    acc = 0
    with torch.no_grad():
        for X, y in loader:
            out, out00, out01, out02 = model(X.to(dev))
            prediction = torch.argmax(out, dim=1)
            cmp = prediction==y.to(dev)
            acc += torch.sum(cmp)/len(cmp)

        resuduals = constlayer(X[0:1].to(dev))
    
    return acc, X[0:1], resuduals
