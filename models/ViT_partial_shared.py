import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch


import math

import torch
import torch.utils.model_zoo as model_zoo
from torch.nn import Parameter
import pdb
import numpy as np
import timm
import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):

    def forward_features_front_n(self, x, n):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks[:n](x)
        
        return x

    def forward_features_tail_n(self, x, n):

        x = self.blocks[n:](x)
        
        return x




timm.models.vision_transformer.VisionTransformer = VisionTransformer


class ViT_partial_shared(nn.Module):

    def __init__(self, pretrained=True, shared_layer_num = 1):
        super(ViT_partial_shared, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        self.vit2 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)

        num_ftrs = 768
        
        self.n = shared_layer_num
        
        #  binary CE
        self.fc1 = nn.Linear(num_ftrs, 2)
        self.fc2 = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        h_shared = self.vit.forward_features_front_n(x, self.n)  #[64, 197, 768]
        
        h1 = self.vit.forward_features_tail_n(h_shared, self.n)  #[64, 197, 768]
        h2 = self.vit2.forward_features_tail_n(h_shared, self.n)  #[64, 197, 768]
        
        logits1 = self.fc1(h1[:,0,:])  # class token [:,0,:]
        logits2 = self.fc2(h2[:,0,:])  # class token [:,0,:]
        
        #pdb.set_trace()
        
        return logits1, logits2


class ViT_partial_shared_tSNE(nn.Module):

    def __init__(self, pretrained=True, shared_layer_num = 1):
        super(ViT_partial_shared_tSNE, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        self.vit2 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)

        num_ftrs = 768
        
        self.n = shared_layer_num
        
        #  binary CE
        self.fc1 = nn.Linear(num_ftrs, 2)
        self.fc2 = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        h_shared = self.vit.forward_features_front_n(x, self.n)  #[64, 197, 768]
        
        h1 = self.vit.forward_features_tail_n(h_shared, self.n)  #[64, 197, 768]
        h2 = self.vit2.forward_features_tail_n(h_shared, self.n)  #[64, 197, 768]
        
        logits1 = self.fc1(h1[:,0,:])  # class token [:,0,:]
        logits2 = self.fc2(h2[:,0,:])  # class token [:,0,:]
        
        #pdb.set_trace()
        
        return h1[:,0,:], h2[:,0,:]




class ViT_STmap_LastCWT_partial_shared(nn.Module):

    def __init__(self, pretrained=True, shared_layer_num = 1):
        super(ViT_STmap_LastCWT_partial_shared, self).__init__()
        
        self.vit1 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        self.vit1_2 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        
        self.vit2 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        self.vit2_2 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)

        num_ftrs = 768
        self.n = shared_layer_num

        #  binary CE
        self.fc1_1 = nn.Linear(num_ftrs, 2)
        self.fc1_2 = nn.Linear(num_ftrs, 2)
        
        self.fc2_1 = nn.Linear(num_ftrs, 2)
        self.fc2_2 = nn.Linear(num_ftrs, 2)
        
        self.fc_fuse1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs*2, num_ftrs//2),
            nn.ReLU(),    
            nn.Dropout(0.5),
            #nn.Linear(num_ftrs//2, 2),   
        )
        
        self.fc_fuse2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs*2, num_ftrs//2),
            nn.ReLU(),    
            nn.Dropout(0.5),
            #nn.Linear(num_ftrs//2, 2),   
        )
        
        self.fc_final1 = nn.Linear(num_ftrs//2, 2)
        self.fc_final2 = nn.Linear(num_ftrs//2, 2)
        

    def forward(self, x1, x2):
        h1_shared = self.vit1.forward_features_front_n(x1, self.n)  #[64, 197, 768]
        h1_1 = self.vit1.forward_features_tail_n(h1_shared, self.n)  #[64, 197, 768]
        h1_2 = self.vit1_2.forward_features_tail_n(h1_shared, self.n)  #[64, 197, 768]
        logits1_1 = self.fc1_1(h1_1[:,0,:])
        logits1_2 = self.fc1_2(h1_2[:,0,:])
        
        h2_shared = self.vit2.forward_features_front_n(x2, self.n)  #[64, 197, 768]
        h2_1 = self.vit2.forward_features_tail_n(h2_shared, self.n)  #[64, 197, 768]
        h2_2 = self.vit2_2.forward_features_tail_n(h2_shared, self.n)  #[64, 197, 768]
        logits2_1 = self.fc2_1(h2_1[:,0,:])
        logits2_2 = self.fc2_2(h2_2[:,0,:])
        
        h_fuse1 = torch.cat((h1_1[:,0,:], h2_1[:,0,:]), dim=1) 
        logitsFused1 = self.fc_fuse1(h_fuse1)
        
        h_fuse2 = torch.cat((h2_1[:,0,:], h2_2[:,0,:]), dim=1) 
        logitsFused2 = self.fc_fuse2(h_fuse2)
        
        logits1_final = self.fc_final1(logitsFused1)
        logits2_final = self.fc_final2(logitsFused2)   
        
        #return logits, regmap8
        return logits1_1, logits1_2, logits2_1, logits2_2, logits1_final, logits2_final

