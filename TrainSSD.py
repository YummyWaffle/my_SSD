import Configs as cfg
from SSD300 import *
from SSD512 import *
from SSDLoader import ssd_voc_loader
from torch.utils.data import DataLoader
import numpy as np
import random
import math
from MultiBoxLoss import ssd_multibox_loss
import torch.optim as optim

assert(cfg.SSD_Version in (300,512))
dataset = ssd_voc_loader('D:/0426DIOR/DIOR/VOCdevkit',cfg.SSD_Version)
train_num = len(dataset)
model = SSD300() if (cfg.SSD_Version==300) else SSD512()
model = model.to(cfg.device)
optimizer = optim.SGD(model.parameters(),lr=0.001)
Epochs = cfg.Epochs
batch_size = cfg.Batch_Size
loss_func = ssd_multibox_loss(batch_size)
interval = 5

for Epoch in range(cfg.Epochs):
    index_list = np.arange(train_num)
    random.shuffle(index_list)
    counter = 0
    for i in range(0,train_num,batch_size):
        total_batch = math.ceil(train_num/batch_size)
        batch_indxs = index_list[i:min(i+batch_size,train_num)]
        real_batch_size = len(batch_indxs)     
        batch_box = []
        batch_cls = []
        in_tsr,bbox,cls = dataset[batch_indxs[0]]
        batch_box.append(bbox)
        batch_cls.append(cls)
        for batch_indx in range(1,real_batch_size):
            temp_img,temp_box,temp_cls = dataset[batch_indxs[batch_indx]]
            #print(temp_img.size())
            in_tsr = torch.cat((in_tsr,temp_img),0)
            batch_box.append(temp_box)
            batch_cls.append(temp_cls)
        cls_pred,reg_pred = model(in_tsr)
        loss = loss_func(cls_pred,reg_pred,batch_cls,batch_box)
        #print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        counter += 1
        if(counter%interval==0):
            print('Epoch:  (%d/%d)  -  Batch:  (%d/%d)  -  Loss : %.4f'%(Epoch+1,Epochs,counter,total_batch,loss))
            
        