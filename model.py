import os

import torch
from torch import nn

def create_dummy(ks:int, inch:int):
    if inch == 3:
        xnut = torch.zeros(size=(1, 1, ks, ks))

        x0 = torch.zeros(size=(1, 1, ks, ks))
        x0[0, 0, ks//2, ks//2] = 1

        x1 = torch.ones(size=(1, 1, ks, ks))
        x1[0, 0, ks//2, ks//2] = 0

        X00 = torch.concat((x0, xnut, xnut), dim=1)
        X01 = torch.concat((xnut, x0, xnut), dim=1)
        X02 = torch.concat((xnut, xnut, x0), dim=1)

        X10 = torch.concat((x1, xnut, xnut), dim=1)
        X11 = torch.concat((xnut, x1, xnut), dim=1)
        X12 = torch.concat((xnut, xnut, x1), dim=1)

        return (X00, X10), (X01, X11), (X02, X12)

    elif inch == 1:
        x0 = torch.zeros(size=(1, 1, ks, ks))
        x0[0, 0, ks//2, ks//2] = 1

        x1 = torch.ones(size=(1, 1, ks, ks))
        x1[0, 0, ks//2, ks//2] = 0
        return x0, x1



class ConstNet(nn.Module):

    def __init__(self, ks, inch, res_ch, num_cls, dev):
        super().__init__()
        self.ks = ks
        self.inch = inch
        self.dev = dev
        self.constlayer = nn.Conv2d(in_channels=inch, out_channels=res_ch, kernel_size=ks, stride=1, bias=False)
        self.blk1 = nn.Sequential(
            nn.Conv2d(in_channels=res_ch, out_channels=96, kernel_size=7, stride=2, padding=2), 
            nn.BatchNorm2d(96), nn.Tanh(), nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.blk2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm2d(64), nn.Tanh(), nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm2d(64), nn.Tanh(), nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.blk3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding='same'),
            nn.BatchNorm2d(128), nn.Tanh(), nn.MaxPool2d(kernel_size=3, stride=2), 
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(128), nn.Tanh(), nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(256), nn.Tanh(), nn.AvgPool2d(kernel_size=5, stride=2), nn.Flatten()
        )

        self.blk4 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=num_cls)#, nn.Tanh(), nn.Dropout(0.4),
            # nn.Linear(in_features=200, out_features=num_cls)
        )

        # self.base = nn.Sequential()

    def constrain_apply(self):
    
        (X00, X10), (X01, X11), (X02, X12) = create_dummy(ks=self.ks, inch=self.inch)
        # print(X00.shape)
        out00 = self.constlayer(X00.to(self.dev))
        out01 = self.constlayer(X01.to(self.dev))
        out02 = self.constlayer(X02.to(self.dev))

        out10 = self.constlayer(X10.to(self.dev))
        out11 = self.constlayer(X11.to(self.dev))
        out12 = self.constlayer(X12.to(self.dev))

        return (out00, out00+out10), (out01, out01+out11), (out02, out02+out12)
    
    def forward(self, x):
        out00, out01, out02 = self.constrain_apply()
        x = self.constlayer(x)
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        out = self.blk4(x)
        # out = self.fc(x)
        return out, out00, out01, out02

    




if __name__ == "__main__":
    print(__file__)

    net = ConstNet(ks=3, inch=3, res_ch=3, num_cls=33, dev='cpu')
    const_layer = net.constlayer
    constparam = ['constlayer.weight', 'constlayer.bias']
    params = list(filter(lambda kv:kv[0] in constparam, net.named_parameters()))
    base_params = list(filter(lambda kv:kv[0] not in constparam, net.named_parameters()))

    print(list(net.parameters()))

        


    

