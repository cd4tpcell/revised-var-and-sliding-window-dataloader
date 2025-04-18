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
from seqdataset import MyDataset
import matplotlib.pyplot as plt
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


dir='/Workspace_II/chenhm/wok/pre/npyf/'
myfiles=sorted(os.listdir(dir))
train_dataset = MyDataset(dir,myfiles,3)
batch_size=16

def collate_fn(batch):
    # 过滤掉 None 值
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None  # 如果整个批次都无效，返回 None
    return torch.utils.data.dataloader.default_collate(batch)


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
predic_seq=1

# testdir='/Workspace_II/chenhm/wok/pre/testdata/'
# testmyfiles=sorted(os.listdir(testdir))
# test_dataset = MyDataset(testdir,testmyfiles,3)
# batch_size=4
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



def build_everything(args: arg_util.Args):
    num_classes=89
    patch_nums = (1, 4,8 , 16,32,64)
    vae_local, var_wo_ddp = build_vae_var(
    V=2048, Cvae=16, ch=16, share_quant_resi=4,test_mode=False,    # hard-coded VQVAE hyperparameters
    device=device, patch_nums=patch_nums,
    num_classes=num_classes, depth=6, shared_aln=False,ch_mult=(1, 2,2,4,8),num_res_blocks=3,input_channel=1,using_sa=False,using_mid_sa=False,
)
    vae_ckpt = 'prevqvae.pth'
    # trainable_params = sum(p.numel() for p in vae_local.parameters() if p.requires_grad)


    # vae_local.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    checkpoint = torch.load(vae_ckpt,map_location="cpu")
    vae_local.load_state_dict(checkpoint['model_state_dict'])
    vae_local.to(device)
    vae_local.eval()
    opt_clz = AdamW(var_wo_ddp.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
    # i=0
    # for param_group in opt_clz.param_groups:
    #     i=i+1
    #     print(i)
    #     print(f"Learning rate: {param_group['lr']}")
    # print(opt_clz.param_groups[0]['lr'])
    # exit()
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
    train_loss = nn.CrossEntropyLoss(label_smoothing=args.ls, reduction='none')

    begin_ends = []
    cur = 0


    for i, pn in enumerate(patch_nums):
        begin_ends.append((cur, cur + pn * pn))
        cur += pn*pn

    L=sum(pn * pn for pn in patch_nums)
    loss_weight = torch.ones(1, L, device=device) / L
    # prog_wp = max(min(args.pg0 /20, 1), 0.01)
    # iters_train = len(dataloader)
    for epoch in range(0,maxepo):
        # is_first_ep=(epoch==0)
        # g_it, max_it = epoch * iters_train, maxepo * iters_train
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        for it,batch in enumerate(train_dataloader):
            if batch is None:
                continue 
            # g_it = epoch * iters_train + it
            
            else:
                opt_clz.zero_grad()
                img = batch['pre'].to(device)

                


                with torch.no_grad():
                    B ,T,C,H,W=img.shape
                    gt_idx_Bl= vae_local.img_to_idxBl(img[:,-1,:,:,:])
                    gt_BL=torch.cat(gt_idx_Bl, dim=1)
                    var_ind=torch.full((B,),0).to(device)


                    cond=[]
                    for t_ind in range(0,T-predic_seq):
                        cond.append(vae_local.generate_minidx(img[:,t_ind,:,:,:]))
                    cond=torch.stack(cond,dim=1)
                    bs=cond.shape[0]
                    indd=np.where(np.random.rand(bs)<0.1)
                    cond[indd] = 0
                    V=vae_local.vocab_size


                    x_BLCv_wo_first_l= vae_local.quantize.idxBl_to_var_input(gt_idx_Bl)

    
                logits_BLV = var_wo_ddp(var_ind, x_BLCv_wo_first_l,cond)

                loss = train_loss(logits_BLV.view(-1,V), gt_BL.view(-1)).view(B, -1)
                loss = loss.mul(loss_weight).sum(dim=-1).mean()
                torch.autograd.set_detect_anomaly(True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(var_wo_ddp.parameters(), 1.)
                opt_clz.step()

                # if it%10==0:
                print(epoch,it,loss.item(),opt_clz.param_groups[0]['lr'],torch.max(img))

            #     if it==10:
            #         break
            # if epoch%1==0:
            #     for it,batch in enumerate(test_dataloader):
            #         img = batch['pre'].to(device)
            #         if torch.max(img)<=1 and it>5:
            #             test_img = batch['pre'].to(device)
        warmUpScheduler.step()
        model_checkpoint_path = 'myvar'+str(epoch)+'.pth'
        torch.save({
            'model_state_dict': var_wo_ddp.state_dict(),
                    }, model_checkpoint_path)

build_everything(arg_util.Args)