import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform, get_mask_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import pandas as pd
import numpy as np
import torch

import torchvision.transforms as transforms
class KeyDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_P = os.path.join(opt.dataroot, opt.phase+ '_in') #person images
        self.dir_K_in = os.path.join(opt.dataroot, opt.phase + '_inK') #keypoints
        self.dir_K_out = os.path.join(opt.dataroot, opt.phase + '_outK')
        self.init_categories(opt.pairLst)
        self.transform = get_transform(opt)


    def init_categories(self, pairLst):
        persons = os.listdir(self.dir_P)
        poses = os.listdir(self.dir_K_out)
        self.pairs = []
        print ('Loading data pairs')
        for person in persons:
            for pose in poses:
                self.pairs.append((person,pose))
        self.size = len(self.pairs)
        print('Loading data pairs finished ...')

    def __getitem__(self, index):
        if self.opt.phase == 'train':
            index = random.randint(0, self.size-1)

        P1_name, BP2_name = self.pairs[index]
        P1_path = os.path.join(self.dir_P, P1_name) # person 1
        BP1_path = os.path.join(self.dir_K_in, P1_name + '.npy') # bone of person 1

        # person 2 and its bone
        P2_path = "" # person 2 image not required, only pose is needed.
        BP2_path = os.path.join(self.dir_K_out, BP2_name) # bone of person 2


        P1_img = Image.open(P1_path).convert('RGB')
        BP1_img = np.load(BP1_path) # h, w, c
        BP2_img = np.load(BP2_path)
     
        BP1 = torch.from_numpy(BP1_img).float() #h, w, c
        BP1 = BP1.transpose(2, 0) #c,w,h
        BP1 = BP1.transpose(2, 1) #c,h,w 

        BP2 = torch.from_numpy(BP2_img).float()
        BP2 = BP2.transpose(2, 0) #c,w,h
        BP2 = BP2.transpose(2, 1) #c,h,w 
        P1 = self.transform(P1_img)
        P2 = torch.zeros(P1.shape)
        # P1_mask = self.transform_mask(P1_mask)

        return {'P1': P1, 'BP1': BP1, 'P2': P2, 'BP2': BP2,
                'P1_path': P1_name, 'P2_path': BP2_name}
                

    def __len__(self):
        if self.opt.phase == 'train':
            return 4000
        elif self.opt.phase == 'test':
            return self.size

    def name(self):
        return 'KeyDataset'
