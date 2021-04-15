import torch
import torch.nn as nn
from AnchorGenerator import PriorBox
from ssd_utils import *
import Configs as cfg
import torch.nn.functional as F

def Encoder(gt_box,gt_cls,priors,map_relation):
    target_reg = torch.zeros(size=(priors.size(0),4))
    target_cls = torch.zeros(size=(priors.size(0),))
    gt_box = to_cxcywh_form(gt_box)
    priors = to_cxcywh_form(priors)
    for i,prior in enumerate(priors):
        gt_index = map_relation[i]
        target_cls[i] = gt_cls[gt_index]
        # gt_box is corner format
        gt_box_i = gt_box[gt_index]
        tx = (gt_box_i[0] - prior[0]) / prior[2]
        ty = (gt_box_i[1] - prior[1]) / prior[3]
        tw = torch.log(gt_box_i[2]/prior[2])
        th = torch.log(gt_box_i[3]/prior[3])
        target_reg[i] = torch.tensor([tx,ty,tw,th])
    return target_reg,target_cls

class ssd_multibox_loss(nn.Module):
    def __init__(self,batch_size,device='cpu'):
        # set device 'cpu' or 'cuda:0', just advoid lack of gpu memory
        super(ssd_multibox_loss,self).__init__()
        self.box = PriorBox()
        self.device = device
        
    def forward(self,pred_cls,pred_reg,gt_cls,gt_box,max_num=256,neg_ratio=3.,iou_thr=0.5):
        """
          * Input Parameters
            pred_cls - tensor - (batch_size,prior_box_num,20)
            pred reg - tensor - (batch_size,prior_box_num,4)
            gt_cls - list(tensor) 
                    - list shape - (batch_size) 
                    - tensor shape - (gt_box_num,20)
            gt-box - list(tensor) -  [Tips: represent a box by (x,y,x',y')]
                    - list shape - (batch_size)
                    - tensor shape - (gt_box_num,4)
          * In-Class Parameters
            self.box() - tensor - (prior_box_num,4) [Tips: represnet a box by (xc,yc,w,h)]
        """
        Loss = torch.zeros(size=(1,))
        prior_box = to_corner_form(self.box()).to(self.device)
        for batch_index in range(len(gt_cls)):
            batch_pred_cls = pred_cls[batch_index,:,:].to(self.device)
            #print(batch_pred_cls.size())
            batch_pred_reg = pred_reg[batch_index,:,:].to(self.device)
            batch_gt_cls = gt_cls[batch_index].to(self.device)
            #print(batch_gt_cls)
            batch_gt_box = gt_box[batch_index].to(self.device)
            # Compute IoU Matrix with size (Anchors_Num,Gt_Num), iou_matrix[i,j] represent the iou between anchor-i & gt-j
            iou_matrix = iou_compute(prior_box,batch_gt_box)
            # Find the anchors index which fits the gts mostly [size - (1,num_gts)]
            best_gt_fits,best_gt_indx = iou_matrix.max(0,keepdim=True)
            best_gt_fits.squeeze_(0) # size [num_gts]
            best_gt_indx.squeeze_(0) # size [num_gts]
            
            # Find the best fit gts of each anchors
            best_anchor_fits,best_anchor_indx = iou_matrix.max(1,keepdim=True)
            best_anchor_fits.squeeze_(1) # size [num_anchors]
            best_anchor_indx.squeeze_(1) # size [num_anchors]
 
            best_anchor_fits.index_fill_(0,best_gt_indx,2) # fill 2 is to make sure it will be selected
            # j -> the gt index       best_gt_indx -> which anchor
            for j in range(best_gt_indx.size(0)):
                best_anchor_indx[best_gt_indx[j]] = j
            
            #print(best_anchor_indx.size())
            
            positive_mask = best_anchor_fits[:] > iou_thr
            negative_mask = best_anchor_fits[:] < iou_thr
            #print('-------')
            batch_pred_cls_pos = batch_pred_cls[positive_mask]
            pos_num = int(max_num * (1./(neg_ratio+1.)))
            neg_num = max_num - pos_num
            
            # Dealing with positive loss
            if(batch_pred_cls_pos.size(0) <= pos_num):
                pos_num = batch_pred_cls_pos.size(0)
                neg_num = int(pos_num * neg_ratio)
                
                batch_pred_reg_pos = batch_pred_reg[positive_mask]
                batch_prior_pos = prior_box[positive_mask]
                best_anchor_indx_pos = best_anchor_indx[positive_mask]
                target_reg,target_cls = Encoder(batch_gt_box,batch_gt_cls,batch_prior_pos,best_anchor_indx_pos)
                #print(target_cls)
                pos_cls_loss = F.cross_entropy(batch_pred_cls_pos,target_cls.long())
                pos_reg_loss = F.l1_loss(batch_pred_reg_pos,target_reg)
            else:
                # If too much switch top-k as postive
                #print(batch_pred_cls.size())
                #pos_max_p,_ = batch_pred_cls.max(1)
                #pos_top_k,pos_top_indx = torch.sort(pos_max_p,descending=True)
                #pos_top_k = pos_top_k[:pos_num]
                batch_pred_cls_pos = batch_pred_cls_pos[:pos_num]
                batch_pred_reg_pos = batch_pred_reg[positive_mask][:pos_num]
                batch_prior_pos = prior_box[positive_mask][:pos_num]
                best_anchor_indx_pos = best_anchor_indx[positive_mask][:pos_num]
                target_reg,target_cls = Encoder(batch_gt_box,batch_gt_cls,batch_prior_pos,best_anchor_indx_pos)
                pos_cls_loss = F.cross_entropy(batch_pred_cls_pos,target_cls.long())
                pos_reg_loss = F.l1_loss(batch_pred_reg_pos,target_reg) 
                #=torch.tensor([])
            
            # Dealing with negative loss
            batch_pred_cls_neg = batch_pred_cls[negative_mask]
            batch_pred_cls_neg = batch_pred_cls_neg[:neg_num]
            neg_target = torch.ones(size=(batch_pred_cls_neg.size(0),))
            neg_target *= 20
            #print(neg_target)
            neg_loss = F.cross_entropy(batch_pred_cls_neg,neg_target.long())
            #print(neg_loss)
            #best_anchor_indx_neg = best_anchor_indx[negative_mask]
            Loss += (neg_loss + pos_cls_loss + pos_reg_loss)
            
        return Loss/len(gt_cls)

if __name__ == '__main__':
    loss_func = ssd_multibox_loss(6)
    loss_func(None,None,None,None)