from models import  Llaja
from dataset_v1 import Dataset_novel_clean
import torch
from torchvision.transforms import v2
from torch.optim.lr_scheduler import StepLR
from torch import optim, nn, load, cat, tril
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader,random_split
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import numpy as np

batch_size=12
dataset=Dataset_novel_clean()

# 创建一个DataLoader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

#models=Llaja().cuda()
models=torch.load("model/model_v1_0.pth").cuda()
models.train()

criterion = nn.CrossEntropyLoss(ignore_index=0,label_smoothing=0.08)
scaler = GradScaler()
optimizer = optim.Adam(models.parameters(), lr=0.00005)
#scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
epoch=100
softmax=nn.Softmax()

for i in tqdm(range(epoch)):  # 一个epoch内
    epoch_loss = 0
    all_count=0
    true_count=0
    loop = tqdm(dataloader, total=len(dataloader),position=0)
    for tgt,tgt_mask,tgt_output,src,src_mask in loop:
        optimizer.zero_grad()
        this_target = tgt_output.squeeze(dim=1).cuda()
        with autocast():
            output = models(src=src.squeeze(dim=1).cuda(),tgt=tgt.squeeze(dim=1).cuda(),src_padmask=~(src_mask.bool().squeeze(dim=1)).cuda(),tgt_keymask=~(tgt_mask.bool().squeeze(dim=1)).cuda())
            output=output.transpose(-1,-2)
            loss = criterion(output, this_target)
            epoch_loss+=loss.item()
            _, predicted = torch.max(output, dim=1)
            correct_num = ((predicted == this_target) & (this_target!=0) ).sum().item()
            all_num=(this_target!=0).sum().item()
            all_count+=all_num
            true_count+=correct_num#计算正确率
            correct_rating = true_count / all_count
            #clip_grad_norm_(models.parameters(), max_norm=10,norm_type=2.0)
        # 反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_description(f'Epoch [{i}/{epoch}]')
        loop.set_postfix(accuracy_rate=correct_rating,loss=loss.item())
        torch.cuda.empty_cache()
    save_name = "model/model_v1_%d.pth" % (i % 3)#切换数据集时改这里
    torch.save(models, save_name)
