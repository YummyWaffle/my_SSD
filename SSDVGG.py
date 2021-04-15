import torch
import torch.nn as nn

class conv_bn_relu(nn.Module):
    def __init__(self,inplane,plane,kernel_size=3,stride=1,padding=1,dilation=1):
        super(conv_bn_relu,self).__init__()
        self.conv = nn.Conv2d(inplane,plane,kernel_size,stride,padding,dilation)
        self.bn = nn.BatchNorm2d(plane)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class conv_relu(nn.Module):
    def __init__(self,inplane,plane,kernel_size=3,stride=1,padding=1,dilation=1):
        super(conv_relu,self).__init__()
        self.conv = nn.Conv2d(inplane,plane,kernel_size,stride,padding,dilation)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.conv(x)
        x = self.relu(x)
        return x

"""
    SSD base network is a slightly modified VGG-16 accroding to [Arxiv:https://arxiv.org/abs/1512.02325]
    It mainly has three modifications:
      1) remove the classifier (FC/MLP Layers)
      2) adjust the parameters of block 5 max-pooling from (kernel_size=2,stride=2) to (kernel_size=3,stride=1,padding=1)
      3) add fc-6 & fc-7 layers (two convolution layers)
    The Output of this network is: 
      1) the output features map of 3rd convs in block 4 (conv4_3)
      2) fc7's feature map
"""
class SSD_VGG(nn.Module):
    arch_setting = {
        11: (1, 1, 2, 2, 2),
        13: (2, 2, 2, 2, 2),
        16: (2, 2, 3, 3, 3),
        19: (2, 2, 4, 4, 4)
    }
    def __init__(self,depth=16,use_bn=True):
        super(SSD_VGG,self).__init__()
        # Only Support VGG-16
        assert(depth==16)
        # Only Support RGB or 3 Channels Images
        self.inplane = 3
        self.plane = 64
        self.stage_settings = self.arch_setting[depth]
        self.features = []
        for i,stage_setting in enumerate(self.stage_settings):
            if(i != 4):
                layer = self._make_layers(conv_bn_relu,stage_setting,self.plane)
                self.features.extend(layer)
                self.plane = int(self.plane * 2)
            else:
                self.plane = int(self.plane / 2)
                layer =  self._make_layers(conv_bn_relu,stage_setting,self.plane,with_pool=False)
                self.features.extend(layer)
        self.features = nn.ModuleList(self.features)
        self.pool = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.fc6 = conv_bn_relu(self.inplane,1024,)
        self.fc7 = conv_bn_relu(1024,1024,kernel_size=1,stride=1,padding=0,dilation=1)
            
        
    def _make_layers(self,block_type,block_nums,plane,with_pool=True):
        layers = []
        layers.append(block_type(self.inplane,plane))
        self.inplane = plane
        for i in range(1,block_nums):
            layers.append(block_type(self.inplane,plane))
        if with_pool:
            layers.append(nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True))
        return layers
    
    def forward(self,x):
        # output conv4_3's feature map whose index is 12
        outs = []
        for i,layers in enumerate(self.features):
            x = layers(x)
            if(i == 12):
                outs.append(x)
        x = self.pool(x)
        x = self.fc6(x)
        # output fc7's feature map
        x = self.fc7(x)
        outs.append(x)
        return outs
