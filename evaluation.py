import os
import random
import argparse
from matplotlib import pyplot as plt
import torch
import numpy as np
import model as m
import datasetup as dst
from conf import Paths
import engine

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
paths = Paths()

parser = argparse.ArgumentParser(prog=os.path.basename(__file__))
parser.add_argument('--ckpoint_num', type=int, required=True)
parser.add_argument('--nw', type=int, required=True)

args = parser.parse_args()

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


if __name__ == '__main__':
    print(__file__)
    model = m.ConstNet(ks=5, inch=3, res_ch=10, num_cls=33, dev=dev)
    data_loader, test_loader = dst.create_laoder(data_path=paths.server_data_path, train_precent=0.8, batch_size=128, nw=args.nw)
    ckpoint = torch.load(os.path.join(paths.model, f'ckpoint_{args.ckpoint_num}.pt'), map_location=dev)
    model.load_state_dict(ckpoint)
    model.to(dev)
    acc, pic, residual = engine.eval_step(model=model, loader=test_loader, dev=dev)
    print(f'accuracu = {acc}')

    res = residual.cpu().detach().squeeze().numpy()
    pix = pic.cpu().squeeze().permute(1,2,0).numpy()
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(24, 16))
    for i in range(2):
        for j in range(3):
            if i==0 and j==0:
                axs[i,j].imshow(pix)
            else:
                axs[i,j].imshow(res[i*3+j], cmap='gray')
    
    plt.subplots_adjust(wspace=0, hspace=0)
    save_path = os.path.join(paths.result, f'ckpoint_{args.ckpoint_num}')
    paths.create_dir(save_path)
    plt.savefig(os.path.join(save_path, 'res.png'))
    



    