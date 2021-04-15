import torch
import torch.utils.data as data
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import torchvision.transforms as transforms


class ssd_voc_loader(data.Dataset):

    CLASSES = ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 'chimney', 'dam',
               'Expressway-Service-area', 'Expressway-toll-station', 'golffield', 'groundtrackfield', 
               'harbor', 'overpass','ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 
               'vehicle','windmill','background']
               
    def __init__(self,dataset_folder,ssd_version=300,train_loader=True,device='cuda:0'):
        self.ssd_version = ssd_version
        self.device = device
        self.image_folder = dataset_folder + '/VOC2007/JPEGImages'
        self.anno_folder = dataset_folder + '/VOC2007/Annotations'
        self.txt_folder = dataset_folder + '/VOC2007/ImageSets/Main'
        if train_loader:
            self.txt_folder += '/train.txt'
        else:
            self.txt_folder += '/test.txt'
        # Read sets
        self.file_sets = []
        f = open(self.txt_folder,'r')
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            self.file_sets.extend(line)
        self.file_num = len(self.file_sets)
        
    def __getitem__(self,index):
        # Parse Image
        img_path = self.image_folder + '/' + self.file_sets[index] + '.jpg'
        img = cv2.imread(img_path)
        img = cv2.resize(img,(self.ssd_version,self.ssd_version),interpolation=cv2.INTER_NEAREST)
        b,g,r = cv2.split(img)
        img = np.array(cv2.merge([r,g,b]))
        # Parse Annotations
        anno_path = self.anno_folder + '/' + self.file_sets[index] + '.xml'
        root = ET.parse(anno_path).getroot()
        img_width = float(root.find('size').find('width').text)
        img_height = float(root.find('size').find('height').text)
        objs = root.findall('object')
        gts_reg = []
        gts_cls = []
        para_w = float(self.ssd_version) / img_width
        para_h = float(self.ssd_version) / img_height
        for obj in objs:
            labels = self.CLASSES.index(obj.find('name').text)
            #labels_encode = np.zeros(len(self.CLASSES))
            #labels_encode[labels] = 1.
            gts_cls.append(labels)
            xmin = int(float(obj.find('bndbox').find('xmin').text) * para_w)
            xmax = int(float(obj.find('bndbox').find('xmax').text) * para_w)
            ymin = int(float(obj.find('bndbox').find('ymin').text) * para_h)
            ymax = int(float(obj.find('bndbox').find('ymax').text) * para_h)
            # normalize x,y is between 0~1
            gts_reg.append([xmin/self.ssd_version,ymin/self.ssd_version,xmax/self.ssd_version,ymax/self.ssd_version])
        gts_cls = torch.tensor(gts_cls)
        #print(gts_cls)
        gts_reg = torch.tensor(gts_reg)
        # Img Transform
        trans = transforms.Compose([transforms.ToTensor()])
        img = trans(img).to(self.device)
        img = img.unsqueeze(0)
        return img,gts_reg,gts_cls
        
    def __len__(self):
        return self.file_num