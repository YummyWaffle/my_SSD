import numpy as np
import torch

# input (xmin,ymin,xmax,ymax) - output (xc,yc,w,h)
def to_cxcywh_form(box):
    return torch.cat((box[:,:2] + (box[:,2:]-box[:,:2])/2.,box[:,2:]-box[:,:2]),1)
# input (xc,yc,w,h) - output (xmin,ymin,xmax,ymax)
def to_corner_form(box):
    return torch.cat((box[:,:2]-box[:,2:]/2.,box[:,:2]+box[:,2:]/2.),1)

def iou_compute(anchors,gts):
    A = anchors.size(0)
    B = gts.size(0)
    max_xy = torch.min(anchors[:,2:].unsqueeze(1).expand(A,B,2),
                       gts[:,2:].unsqueeze(0).expand(A,B,2))
    min_xy = torch.max(anchors[:,:2].unsqueeze(1).expand(A,B,2),
                       gts[:,:2].unsqueeze(0).expand(A,B,2))
    inter = torch.clamp((max_xy-min_xy),min=0) # size (A,B,2) inter[:,:,0] -> w  inter[:,:,1] -> h
    inter = inter[:,:,0] * inter[:,:,1] # size (A,B)
    area_anchors = ((anchors[:,2]-anchors[:,0])*(anchors[:,3]-anchors[:,1])).unsqueeze(1).expand_as(inter)
    area_gts = ((gts[:,2]-gts[:,0])*(gts[:,3]-gts[:,1])).unsqueeze(0).expand_as(inter)
    return inter / (area_anchors + area_gts - inter)
    
if __name__ == '__main__':
    box = torch.tensor([[0,0,3,3],[1,1,5,5],[1,6,7,9]],dtype=torch.float32)
    anchors = torch.tensor([[1,1,2,2],[3,3,4,4],[5,5,6,6]],dtype=torch.float32)
    iou = iou_compute(anchors,box)
    print(iou)