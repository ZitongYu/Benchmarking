from __future__ import print_function, division
import os
import torch
import pandas as pd
#from skimage import io, transform
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
import os 
import imgaug.augmenters as iaa


 



class Normaliztion_valtest(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        image_x, spoofing_label, map_x1 = sample['image_x'],sample['spoofing_label'],sample['map_x1']
        new_image_x = (image_x - 127.5)/128     # [-1,1]
        return {'image_x': new_image_x, 'spoofing_label': spoofing_label, 'map_x1': map_x1}




class ToTensor_valtest(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image_x, spoofing_label, map_x1 = sample['image_x'],sample['spoofing_label'],sample['map_x1']
        
        # swap color axis because
        # numpy image: (batch_size) x T x H x W x C
        # torch image: (batch_size) x T x C X H X W
        image_x = image_x.transpose((2, 0, 1))
        image_x = np.array(image_x)
        
        map_x1 = np.array(map_x1)
                        
        spoofing_label_np = np.array([0],dtype=np.long)
        spoofing_label_np[0] = spoofing_label
        
        
        return {'image_x': torch.from_numpy(image_x.astype(np.float)).float(), 'spoofing_label': torch.from_numpy(spoofing_label_np.astype(np.long)).long(), 'map_x1': torch.from_numpy(map_x1.astype(np.float)).float()}


# /home/ztyu/FAS_dataset/OULU/Train_images/          6_3_20_5_121_scene.jpg        6_3_20_5_121_scene.dat
# /home/ztyu/FAS_dataset/OULU/IJCB_re/OULUtrain_images/        6_3_20_5_121_depth1D.jpg
class Spoofing_valtest(Dataset):

    def __init__(self, info_list, root_dir,  transform=None):

        self.landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    
    def __getitem__(self, idx):
        #print(self.landmarks_frame.iloc[idx, 0])
        videoname = str(self.landmarks_frame.iloc[idx, 0])
        image_path = os.path.join(self.root_dir, videoname)
             
        image_x = self.get_single_image_x(image_path)
		    
        spoofing_label = self.landmarks_frame.iloc[idx, 1]
        
        if spoofing_label == 1:            # real
            #spoofing_label = 0            # real
            #map_x1 = np.zeros((28, 28))   # real
            map_x1 = np.ones((28, 28))
        else:                              # fake
            #spoofing_label = 1
            #map_x1 = np.ones((28, 28))    # fake
            map_x1 = np.zeros((28, 28))
        

        sample = {'image_x': image_x, 'spoofing_label': spoofing_label, 'map_x1': map_x1}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_single_image_x(self, image_path):
        
        image_x = np.zeros((224, 224, 3))
        

        image_jpg = 'CWT_'+ str(63) + '.jpg'
            
        image_path_temp = os.path.join(image_path, image_jpg)
        image_x_temp = cv2.imread(image_path_temp)
        image_x = cv2.resize(image_x_temp, (224, 224))
        
   
        return image_x




