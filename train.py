import os
import random
import numpy as np
import torch
from matplotlib import pyplot as plt
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
    lr = 3e-4
    ks = 3

    model = m.ConstNet(ks=ks, inch=3, res_ch=3, num_cls=33, dev=dev)
    model_state = torch.load(os.path.join(paths.data, f'ckpoint_{170}.pt'))
    # model.load_state_dict(model_state)
    constparam = ['constlayer.weight', 'constlayer.bias']
    params = list(filter(lambda kv:kv[0] in constparam, model.named_parameters()))
    base_params = list(filter(lambda kv:kv[0] not in constparam, model.named_parameters()))
    opt = optim.Adam([
        {'params': [temp[1] for temp in base_params]}, 
        {'params': [temp[1] for temp in params], 'lr':lr}
    ], lr=lr)
    #sch = torch.optim.lr_scheduler.LinearLR(optimizer=opt, start_factor=1, end_factor=0.0001, total_iters=10)
    criterion = nn.CrossEntropyLoss()

    data_loader, test_loader = dst.create_laoder(data_path=paths.server_data_path, train_precent=0.87, batch_size=128, nw=22)
    model.to(dev)
    num_batch = len(data_loader)
    test_batch = len(test_loader)
    for epoch in range(epochs):
        model.train()
        train_loss = engine.train_step(model=model, opt=opt, criterion=criterion, loader=data_loader, dev=dev)
        #sch.step()
        torch.save(model.state_dict(), os.path.join(paths.model, f'ckpoint_{epoch}.pt'))
        print(f"epoch={epoch} loss={train_loss/num_batch}")

        if epoch%5 == 0:
            acc, pic, residual = engine.eval_step(model=model, loader=test_loader, dev=dev)
            print(f'epoch={epoch} accuracy = {acc/test_batch} test_batch={test_batch}')

            res = residual.cpu().detach().squeeze().numpy()
            pix = pic.cpu().squeeze().permute(1,2,0).numpy()
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(24, 16))
            for i in range(2):
                for j in range(2):
                    if i==0 and j==0:
                        axs[i,j].imshow(pix)
                    else:
                        axs[i,j].imshow(res[i*2+j-1], cmap='gray')
            
            plt.subplots_adjust(wspace=0, hspace=0)
            save_path = os.path.join(paths.result, f'ckpoint_{epoch}')
            paths.create_dir(save_path)
            plt.savefig(os.path.join(save_path, 'res.png'))
