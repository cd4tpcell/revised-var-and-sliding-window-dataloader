import pickle
import os
import numpy as np
import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from utils import arg_util
from models import VAR, VQVAE, build_vae_var
from functools import partial
from torch.optim import AdamW
import torch.nn as nn
import torch.nn.functional as F
from discri import NLayerDiscriminator

import random
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_random_seed(56)


device = "cuda:1" if torch.cuda.is_available() else "cpu"

class MyDataset(Dataset):
    def __init__(self, dir,files):
        self.files = files
        self.dir = dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return {
            'pre': torch.from_numpy(np.load(self.dir+self.files[idx]).reshape(1,1024,1024).astype(np.float32)),

        }
dir='/Workspace_II/chenhm/wok/pre/npyf/'
myfiles=sorted(os.listdir(dir))
batch_size = 4
changdu=(len(myfiles)//batch_size)*batch_size
dataset = MyDataset(dir,myfiles[0:changdu])


dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

patch_nums = (1, 4,8 , 16,32,64)

num_classes=1000

vae, var = build_vae_var(
    V=2048, Cvae=16, ch=16, share_quant_resi=4,test_mode=False,    # hard-coded VQVAE hyperparameters
    device=device, patch_nums=patch_nums,
    num_classes=1000, depth=8, shared_aln=False,ch_mult=(1, 2,2,4,8),num_res_blocks=3,input_channel=1,using_sa=False,using_mid_sa=False
)

disc=NLayerDiscriminator(input_nc=1, ndf=64, n_layers=3, use_actnorm=True,init_method='kaiming')
disc.to(device)
# disc.eval()
trainable_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
print(trainable_params)
exit()
vae.to(device)
vae.train()

       
num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)  
print(f"模型参数数量: {num_params}")
# disc.apply(init_weights_normal)   
opt_clz = AdamW(vae.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
opt_disc = AdamW(disc.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)


import torch.optim as optim
from Scheduler import GradualWarmupScheduler
maxepo=100


cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer = opt_clz,
                        T_max = maxepo,
                        eta_min = 0,
                        last_epoch = -1
                    )
warmUpScheduler = GradualWarmupScheduler(
                            optimizer = opt_clz,
                            multiplier = 2.0,
                            warm_epoch = maxepo // 10,
                            after_scheduler = cosineScheduler,
                            last_epoch = 0
                        )

begin_ends = []
cur = 0
# from lpips import LPIPS
# perceptual_loss = LPIPS().eval().to(device=device)
def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor
use_dis=0
for epoch in range(0,maxepo):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        vae.train()
        for it,batch in enumerate(dataloader):
            
            original=batch['pre'].to(device)
            
            # original=original.reshape(original.shape[0],1,1024,1024)

            recons,_,vqloss=vae(original)
            
            # mseloss=F.mse_loss(recons,original)
            mseloss=F.l1_loss(recons, original)
            if use_dis:
                disc.eval()
                gen_fake = disc(recons)


                disc_factor = adopt_weight(1.0, epoch*changdu//batch_size+it, threshold=(maxepo+1)*changdu//batch_size)
                

                g_loss = -torch.mean(gen_fake)
                lam = vae.calculate_lambda(mseloss, g_loss)

                vae_loss =mseloss+vqloss*0.1+ g_loss* disc_factor *lam
            else:
                vae_loss=mseloss+vqloss*0.1

            opt_clz.zero_grad()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.)
            vae_loss.backward()
            opt_clz.step()



            if use_dis:
                disc.train()
                disc_real = disc(original)
                disc_fake = disc(recons.detach())
                d_loss_real = torch.mean(F.relu(1. - disc_real))
                d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                gan_loss = 0.5*(d_loss_real + d_loss_fake)
                gan_loss = disc_factor * 0.5*(d_loss_real + d_loss_fake)

                opt_disc.zero_grad()  
                gan_loss.backward()
                torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.)
                opt_disc.step() 


                print(f"epoch:{epoch},it:{it},loss:{mseloss.item(),vqloss.item(),g_loss.item()},lambda:{lam}")
            else:
                print(f"epoch:{epoch},it:{it},loss:{mseloss.item(),vqloss.item()}")
            # if it==2:
            #      break
        warmUpScheduler.step()
        if epoch ==0 or (epoch+1)%10==0:
            vae.eval()
            testdir='/Workspace_II/chenhm/wok/pre/var/testdata/'
            nfs=os.listdir(testdir)
            data=np.stack([np.load(testdir+nfs[i]) for i in range(2)])

            datavq=(torch.tensor(data).to(device)).reshape(2,1,1024,1024)
            jie,_,_=vae(datavq)
            del datavq
            jie=(jie.cpu().detach().numpy()).astype(float)
            import matplotlib.pyplot as plt

            import xarray as xr


            data=np.concatenate([data,jie[:,0]],axis=0)
            data=(data+1)*0.5*256.113
            fig = plt.figure(figsize=(16,8), constrained_layout=True)
            axes = [] 
            for i in range(1):
                for j in range(4):
                    ax = fig.add_subplot(1, 4, i * 4 + j + 1)
                    im = ax.imshow(data[j], cmap='viridis', origin='lower',vmin=0,vmax=50)
                    cbar = fig.colorbar(im, ax=ax, orientation='vertical')
                    cbar.set_label('pre')  # 设置色标标签
            # plt.tight_layout()
            plt.savefig('pre'+str(epoch)+'.png')


model_checkpoint_path = 'prevqvae.pth'
torch.save({
            'model_state_dict': vae.state_dict(),
            }, model_checkpoint_path)

# model_checkpoint_path = 'predis.pth'
# torch.save({
#             'model_state_dict': disc.state_dict(),
#             }, model_checkpoint_path)
