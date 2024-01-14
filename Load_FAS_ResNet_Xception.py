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


 


#face_scale = 0.9  #default for test, for training , can be set from [0.8 to 1.0]

# data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
seq = iaa.Sequential([
    iaa.Add(value=(-40,40), per_channel=True), # Add color 
    iaa.GammaContrast(gamma=(0.5,1.5)) # GammaContrast with a gamma of 0.5 to 1.5
])




def crop_face_from_scene(image,face_name_full, scale):
    f=open(face_name_full,'r')
    lines=f.readlines()
    y1,x1,w,h=[float(ele) for ele in lines[:4]]
    f.close()
    y2=y1+w
    x2=x1+h

    y_mid=(y1+y2)/2.0
    x_mid=(x1+x2)/2.0
    h_img, w_img = image.shape[0], image.shape[1]
    #w_img,h_img=image.size
    w_scale=scale*w
    h_scale=scale*h
    y1=y_mid-w_scale/2.0
    x1=x_mid-h_scale/2.0
    y2=y_mid+w_scale/2.0
    x2=x_mid+h_scale/2.0
    y1=max(math.floor(y1),0)
    x1=max(math.floor(x1),0)
    y2=min(math.floor(y2),w_img)
    x2=min(math.floor(x2),h_img)

    #region=image[y1:y2,x1:x2]
    region=image[x1:x2,y1:y2]
    return region




# Tensor
class Cutout(object):
    def __init__(self, length=30):
        self.length = length

    def __call__(self, sample):
        img, spoofing_label, map_x1 = sample['image_x'],sample['spoofing_label'],sample['map_x1']
        h, w = img.shape[1], img.shape[2]    # Tensor [1][2],  nparray [0][1]
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        length_new = np.random.randint(1, self.length)
        
        y1 = np.clip(y - length_new // 2, 0, h)
        y2 = np.clip(y + length_new // 2, 0, h)
        x1 = np.clip(x - length_new // 2, 0, w)
        x2 = np.clip(x + length_new // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return {'image_x': img, 'spoofing_label': spoofing_label, 'map_x1': map_x1}


class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        image_x, spoofing_label, map_x1 = sample['image_x'],sample['spoofing_label'],sample['map_x1']
        new_image_x = (image_x - 127.5)/128     # [-1,1]
        return {'image_x': new_image_x, 'spoofing_label': spoofing_label, 'map_x1': map_x1}



class RandomHorizontalFlip(object):
    """Horizontally flip the given Image randomly with a probability of 0.5."""
    def __call__(self, sample):
        image_x, spoofing_label, map_x1 = sample['image_x'],sample['spoofing_label'],sample['map_x1']
        
        new_image_x = np.zeros((224, 224, 3))
        #new_image_x = np.zeros((256, 256, 3))

        p = random.random()
        if p < 0.5:
            #print('Flip')

            new_image_x = cv2.flip(image_x, 1)

                
            return {'image_x': new_image_x, 'spoofing_label': spoofing_label, 'map_x1': map_x1}
        else:
            #print('no Flip')
            return {'image_x': image_x, 'spoofing_label': spoofing_label, 'map_x1': map_x1}



class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image_x, spoofing_label, map_x1 = sample['image_x'],sample['spoofing_label'],sample['map_x1']
        
        # swap color axis because
        # numpy image: (batch_size) x H x W x C
        # torch image: (batch_size) x C X H X W
        image_x = image_x[:,:,::-1].transpose((2, 0, 1))
        image_x = np.array(image_x)
        
        map_x1 = np.array(map_x1)
                        
        spoofing_label_np = np.array([0],dtype=np.long)
        spoofing_label_np[0] = spoofing_label
        
        
        return {'image_x': torch.from_numpy(image_x.astype(np.float)).float(), 'spoofing_label': torch.from_numpy(spoofing_label_np.astype(np.long)).long(), 'map_x1': torch.from_numpy(map_x1.astype(np.float)).float()}


# /home/ztyu/FAS_dataset/OULU/Train_images/          6_3_20_5_121_scene.jpg        6_3_20_5_121_scene.dat
# /home/ztyu/FAS_dataset/OULU/IJCB_re/OULUtrain_images/        6_3_20_5_121_depth1D.jpg
class Spoofing_train(Dataset):

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
        #image_x = np.zeros((256, 256, 3))

        # RGB
        image_id = np.random.randint(1, 8)
        image_jpg = 'Face_0'+ str(image_id) + '.jpg'
        
        image_path = os.path.join(image_path, image_jpg)
        image_x_temp = cv2.imread(image_path)

         
        image_x = cv2.resize(image_x_temp, (224, 224))
        #image_x = cv2.resize(image_x_temp, (256, 256))
        
        
        # data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
        image_x_aug = seq.augment_image(image_x) 
        
        
   
        return image_x_aug




