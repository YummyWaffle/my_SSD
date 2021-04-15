import torch
import torch.nn as nn
from SSDVGG import SSD_VGG

class SSD512(nn.Module):
    def __init__(self,box_settings=[4,6,6,6,4,4],
                      out_channels = [512,1024,512,256,256,256],
                      class_num=20):
        super(SSD512,self).__init__()
        self.class_num = class_num
        self.base_net = SSD_VGG()
        self.extra_layer1 = nn.Sequential(
            nn.Conv2d(1024,256,kernel_size=1,stride=1,padding=0,dilation=1),
            nn.Conv2d(256,512,kernel_size=3,stride=2,padding=1,dilation=1),
        )
        self.extra_layer2 = nn.Sequential(
            nn.Conv2d(512,128,kernel_size=1,stride=1,padding=0,dilation=1),
            nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1,dilation=1),
        )
        self.extra_layer3 = nn.Sequential(
            nn.Conv2d(256,128,kernel_size=1,stride=1,padding=0,dilation=1),
            nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1,dilation=1),
        )
        self.extra_layer4 = nn.Sequential(
            nn.Conv2d(256,128,kernel_size=1,stride=1,padding=0,dilation=1),
            nn.Conv2d(128,256,kernel_size=4,stride=1),
        )
        self.cls_convs = []
        self.reg_convs = []
        for i,box_setting in enumerate(box_settings):
            self.cls_convs.append(nn.Conv2d(out_channels[i],box_setting*(class_num+1),kernel_size=3,padding=1))
            #self.cls_convs.append(nn.Conv2d(out_channels[i],box_setting*class_num,kernel_size=3,padding=1))
            self.reg_convs.append(nn.Conv2d(out_channels[i],box_setting*4,kernel_size=3,padding=1))
        self.cls_convs = nn.ModuleList(self.cls_convs) 
        self.reg_convs = nn.ModuleList(self.reg_convs)
        
            
    def forward(self,x):
        # Get Feature Maps Here
        feats = []
        x = self.base_net(x)
        feats.extend(x)
        x = x[1]
        x = self.extra_layer1(x)
        feats.append(x)
        x = self.extra_layer2(x)
        feats.append(x)
        x = self.extra_layer3(x)
        feats.append(x)
        x = self.extra_layer4(x)
        feats.append(x)
        # Pass All The Reg & Cls Module
        cls_pred = []
        reg_pred = []
        for i,cls_conv in enumerate(self.cls_convs):
            cls_pred.append(cls_conv(feats[i]).permute(0,2,3,1).contiguous())
            reg_pred.append(self.reg_convs[i](feats[i]).permute(0,2,3,1).contiguous())
        cls_pred = torch.cat([c.view(c.size(0),-1,self.class_num+1) for c in cls_pred],1)
        reg_pred = torch.cat([r.view(r.size(0),-1,4) for r in reg_pred],1)
        return cls_pred,reg_pred
        
if __name__ == '__main__':
    print('Testing SSD 512 Fore-Propagation')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SSD512().to(device)
    print(model)
    tsr = torch.randn((1,3,512,512)).to(device)
    cls_pred,reg_pred = model(tsr)
    for i in range(len(cls_pred)):
        print(cls_pred[i].size(),reg_pred[i].size())