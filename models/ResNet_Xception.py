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






class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()
        #y = self.avg_pool(x).view(b, c)
        y = self.fc(x)
        return x * y



class ResNet50(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNet50, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)

        num_ftrs = resnet.fc.in_features
        
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        #  binary CE
        self.fc = nn.Linear(num_ftrs, 2)
        
        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x):
        h = self.features(x)

        regmap8 =  self.avgpool8(h)
        
        logits = self.fc(regmap8.squeeze(-1).squeeze(-1))
        
        #return logits, regmap8
        return logits



class ResNet50_partial_shared(nn.Module):

    def __init__(self, pretrained=True, shared_layer_num = 1):
        super(ResNet50_partial_shared, self).__init__()
        resnet1 = models.resnet50(pretrained=pretrained)
        resnet2 = models.resnet50(pretrained=pretrained)

        num_ftrs = resnet1.fc.in_features
        
        self.shared_feature = nn.Sequential(*list(resnet1.children())[:(-6+shared_layer_num)])
        self.features1 = nn.Sequential(*list(resnet1.children())[(-6+shared_layer_num):-2])
        self.features2 = nn.Sequential(*list(resnet2.children())[(-6+shared_layer_num):-2])

        #  binary CE
        self.fc1 = nn.Linear(num_ftrs, 2)
        self.fc2 = nn.Linear(num_ftrs, 2)
        
        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x):
        h_shared = self.shared_feature(x)
        
        h1 = self.features1(h_shared)
        h2 = self.features2(h_shared)

        regmap1 =  self.avgpool8(h1)
        logits1 = self.fc1(regmap1.squeeze(-1).squeeze(-1))
        
        regmap2 =  self.avgpool8(h2)
        logits2 = self.fc2(regmap2.squeeze(-1).squeeze(-1))
        
        return logits1, logits2





class ResNet50_2heads(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNet50_2heads, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)

        num_ftrs = resnet.fc.in_features
        
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        #  binary CE
        self.fc = nn.Linear(num_ftrs, 2)
        self.fc2 = nn.Linear(num_ftrs, 2)
        
        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x):
        h = self.features(x)

        regmap8 =  self.avgpool8(h)
        
        regmap8 = regmap8.squeeze(-1).squeeze(-1)
        
        logits = self.fc(regmap8)
        logits2 = self.fc2(regmap8)
        
        #return logits, regmap8
        return logits, logits2



class ResNet50_2heads_expression(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNet50_2heads_expression, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)

        num_ftrs = resnet.fc.in_features
        
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        #  binary CE
        self.fc = nn.Linear(num_ftrs, 2)
        self.fc2 = nn.Linear(num_ftrs, 7)
        
        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x):
        h = self.features(x)

        regmap8 =  self.avgpool8(h)
        
        regmap8 = regmap8.squeeze(-1).squeeze(-1)
        
        logits = self.fc(regmap8)
        logits2 = self.fc2(regmap8)
        
        #return logits, regmap8
        return logits, logits2




class ResNet50_2heads_race(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNet50_2heads_race, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)

        num_ftrs = resnet.fc.in_features
        
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        #  binary CE
        self.fc = nn.Linear(num_ftrs, 2)
        self.fc2 = nn.Linear(num_ftrs, 3)
        
        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x):
        h = self.features(x)

        regmap8 =  self.avgpool8(h)
        
        regmap8 = regmap8.squeeze(-1).squeeze(-1)
        
        logits = self.fc(regmap8)
        logits2 = self.fc2(regmap8)
        
        #return logits, regmap8
        return logits, logits2



class ResNet50_3class(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNet50_3class, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)

        num_ftrs = resnet.fc.in_features
        
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        #  binary CE
        self.fc = nn.Linear(num_ftrs, 3)
        
        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x):
        h = self.features(x)

        regmap8 =  self.avgpool8(h)
        
        regmap8 = regmap8.squeeze(-1).squeeze(-1)
        
        logits = self.fc(regmap8)
        
        return logits




class ViT_2heads(nn.Module):

    def __init__(self, pretrained=True):
        super(ViT_2heads, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)

        num_ftrs = 768
        
        #  binary CE
        self.fc1 = nn.Linear(num_ftrs, 2)
        self.fc2 = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        h1 = self.vit.forward_features(x)
        logits1 = self.fc1(h1)
        logits2 = self.fc2(h1)
        
        return logits1, logits2






class ViT_2heads_expression(nn.Module):

    def __init__(self, pretrained=True):
        super(ViT_2heads_expression, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)

        num_ftrs = 768
        
        #  binary CE
        self.fc1 = nn.Linear(num_ftrs, 2)
        self.fc2 = nn.Linear(num_ftrs, 7)

    def forward(self, x):
        h1 = self.vit.forward_features(x)
        logits1 = self.fc1(h1)
        logits2 = self.fc2(h1)
        
        return logits1, logits2



class ViT_2heads_race(nn.Module):

    def __init__(self, pretrained=True):
        super(ViT_2heads_race, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)

        num_ftrs = 768
        
        #  binary CE
        self.fc1 = nn.Linear(num_ftrs, 2)
        self.fc2 = nn.Linear(num_ftrs, 3)

    def forward(self, x):
        h1 = self.vit.forward_features(x)
        logits1 = self.fc1(h1)
        logits2 = self.fc2(h1)
        
        return logits1, logits2





class ResNet18(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNet18, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)

        num_ftrs = resnet.fc.in_features
        
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        #  binary CE
        self.fc = nn.Linear(num_ftrs, 2)
        
        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x):
        h = self.features(x)

        regmap8 =  self.avgpool8(h)
        
        logits = self.fc(regmap8.squeeze(-1).squeeze(-1))
        
        #return logits, regmap8
        return logits
        


class ResNet18_STmap_LastCWT(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNet18_STmap_LastCWT, self).__init__()
        resnet1 = models.resnet18(pretrained=pretrained)
        #resnet1 = models.resnet50(pretrained=pretrained)
        
        resnet2 = models.resnet18(pretrained=pretrained)
        #resnet2 = models.resnet50(pretrained=pretrained)

        num_ftrs = resnet1.fc.in_features
        
        self.features1 = nn.Sequential(*list(resnet1.children())[:-2])
        
        self.features2 = nn.Sequential(*list(resnet2.children())[:-2])

        #  binary CE
        self.fc1 = nn.Linear(num_ftrs, 2)
        self.fc2 = nn.Linear(num_ftrs, 2)
        
        self.fc_fuse = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs*2, num_ftrs//2),
            nn.ReLU(),    
            nn.Dropout(0.5),
            nn.Linear(num_ftrs//2, 2),   
        )
        
        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x1, x2):
        h1 = self.features1(x1)
        regmap1 =  self.avgpool8(h1).squeeze(-1).squeeze(-1)
        logits1 = self.fc1(regmap1)
        
        h2 = self.features2(x2)
        regmap2 =  self.avgpool8(h2).squeeze(-1).squeeze(-1)
        logits2 = self.fc2(regmap2)
        
        h_fuse = torch.cat((regmap1,regmap2), dim=1) 
        logitsFused = self.fc_fuse(h_fuse)   
        
        #return logits, regmap8
        return logits1, logits2, logitsFused
        



class ResNet18_STmap_LastCWT_2head(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNet18_STmap_LastCWT_2head, self).__init__()
        resnet1 = models.resnet18(pretrained=pretrained)
        #resnet1 = models.resnet50(pretrained=pretrained)
        
        resnet2 = models.resnet18(pretrained=pretrained)
        #resnet2 = models.resnet50(pretrained=pretrained)

        num_ftrs = resnet1.fc.in_features
        
        self.features1 = nn.Sequential(*list(resnet1.children())[:-2])
        
        self.features2 = nn.Sequential(*list(resnet2.children())[:-2])

        #  binary CE
        self.fc1_1 = nn.Linear(num_ftrs, 2)
        self.fc1_2 = nn.Linear(num_ftrs, 2)
        
        self.fc2_1 = nn.Linear(num_ftrs, 2)
        self.fc2_2 = nn.Linear(num_ftrs, 2)
        
        self.fc_fuse = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs*2, num_ftrs//2),
            nn.ReLU(),    
            nn.Dropout(0.5),
            #nn.Linear(num_ftrs//2, 2),   
        )
        
        self.fc_final1 = nn.Linear(num_ftrs//2, 2)
        self.fc_final2 = nn.Linear(num_ftrs//2, 2)
        
        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x1, x2):
        h1 = self.features1(x1)
        regmap1 =  self.avgpool8(h1).squeeze(-1).squeeze(-1)
        #logits1 = self.fc1(regmap1)
        logits1_1 = self.fc1_1(regmap1)
        logits1_2 = self.fc1_2(regmap1)
        
        h2 = self.features2(x2)
        regmap2 =  self.avgpool8(h2).squeeze(-1).squeeze(-1)
        #logits2 = self.fc2(regmap2)
        logits2_1 = self.fc2_1(regmap2)
        logits2_2 = self.fc2_2(regmap2)
        
        h_fuse = torch.cat((regmap1,regmap2), dim=1) 
        logitsFused = self.fc_fuse(h_fuse) 
        
        logits1_final = self.fc_final1(logitsFused)
        logits2_final = self.fc_final2(logitsFused)     
        
        #return logits1, logits2, logitsFused
        return logits1_1, logits1_2, logits2_1, logits2_2, logits1_final, logits2_final





class ResNet18_STmap_LastCWT_partial_shared(nn.Module):

    def __init__(self, pretrained=True, shared_layer_num = 1):
        super(ResNet18_STmap_LastCWT_partial_shared, self).__init__()
        resnet1 = models.resnet18(pretrained=pretrained)
        resnet1_2 = models.resnet18(pretrained=pretrained)
        
        resnet2 = models.resnet18(pretrained=pretrained)
        resnet2_2 = models.resnet18(pretrained=pretrained)

        num_ftrs = resnet1.fc.in_features
        
        #self.features1 = nn.Sequential(*list(resnet1.children())[:-2])
        
        #self.features2 = nn.Sequential(*list(resnet2.children())[:-2])
        
        self.shared_feature1 = nn.Sequential(*list(resnet1.children())[:(-6+shared_layer_num)])
        self.features1_1 = nn.Sequential(*list(resnet1.children())[(-6+shared_layer_num):-2])
        self.features1_2 = nn.Sequential(*list(resnet1_2.children())[(-6+shared_layer_num):-2])
        
        self.shared_feature2 = nn.Sequential(*list(resnet2.children())[:(-6+shared_layer_num)])
        self.features2_1 = nn.Sequential(*list(resnet2.children())[(-6+shared_layer_num):-2])
        self.features2_2 = nn.Sequential(*list(resnet2_2.children())[(-6+shared_layer_num):-2])
        

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
        
        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x1, x2):
        h1_shared = self.shared_feature1(x1)
        h1_1 = self.features1_1(h1_shared)
        h1_2 = self.features1_2(h1_shared)
        regmap1_1 =  self.avgpool8(h1_1).squeeze(-1).squeeze(-1)
        regmap1_2 =  self.avgpool8(h1_2).squeeze(-1).squeeze(-1)
        logits1_1 = self.fc1_1(regmap1_1)
        logits1_2 = self.fc1_2(regmap1_2)
        
        
        h2_shared = self.shared_feature2(x2)
        h2_1 = self.features2_1(h2_shared)
        h2_2 = self.features2_2(h2_shared)
        regmap2_1 =  self.avgpool8(h2_1).squeeze(-1).squeeze(-1)
        regmap2_2 =  self.avgpool8(h2_2).squeeze(-1).squeeze(-1)
        logits2_1 = self.fc2_1(regmap2_1)
        logits2_2 = self.fc2_2(regmap2_2)
        
        
        h_fuse1 = torch.cat((regmap1_1,regmap2_1), dim=1) 
        logitsFused1 = self.fc_fuse1(h_fuse1) 
        
        h_fuse2 = torch.cat((regmap1_2,regmap2_2), dim=1) 
        logitsFused2 = self.fc_fuse2(h_fuse2) 
        
        logits1_final = self.fc_final1(logitsFused1)
        logits2_final = self.fc_final2(logitsFused2)     
        
        #return logits1, logits2, logitsFused
        return logits1_1, logits1_2, logits2_1, logits2_2, logits1_final, logits2_final





class ResNet18_STmap_LastCWT_3class(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNet18_STmap_LastCWT_3class, self).__init__()
        resnet1 = models.resnet18(pretrained=pretrained)
        #resnet1 = models.resnet50(pretrained=pretrained)
        
        resnet2 = models.resnet18(pretrained=pretrained)
        #resnet2 = models.resnet50(pretrained=pretrained)

        num_ftrs = resnet1.fc.in_features
        
        self.features1 = nn.Sequential(*list(resnet1.children())[:-2])
        
        self.features2 = nn.Sequential(*list(resnet2.children())[:-2])

        #  binary CE
        self.fc1 = nn.Linear(num_ftrs, 3)
        self.fc2 = nn.Linear(num_ftrs, 3)
        
        self.fc_fuse = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs*2, num_ftrs//2),
            nn.ReLU(),    
            nn.Dropout(0.5),
            nn.Linear(num_ftrs//2, 3),   
        )
        
        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x1, x2):
        h1 = self.features1(x1)
        regmap1 =  self.avgpool8(h1).squeeze(-1).squeeze(-1)
        logits1 = self.fc1(regmap1)
        
        h2 = self.features2(x2)
        regmap2 =  self.avgpool8(h2).squeeze(-1).squeeze(-1)
        logits2 = self.fc2(regmap2)
        
        h_fuse = torch.cat((regmap1,regmap2), dim=1) 
        logitsFused = self.fc_fuse(h_fuse)   
        
        #return logits, regmap8
        return logits1, logits2, logitsFused




class ResNet18_RGB_STmap_LastCWT(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNet18_RGB_STmap_LastCWT, self).__init__()
        resnet1 = models.resnet18(pretrained=pretrained)
        
        resnet2 = models.resnet18(pretrained=pretrained)
        
        resnet3 = models.resnet18(pretrained=pretrained)

        num_ftrs = resnet1.fc.in_features
        
        self.features1 = nn.Sequential(*list(resnet1.children())[:-2])
        
        self.features2 = nn.Sequential(*list(resnet2.children())[:-2])
        
        self.features3 = nn.Sequential(*list(resnet2.children())[:-2])

        #  binary CE
        self.fc1 = nn.Linear(num_ftrs, 2)
        self.fc2 = nn.Linear(num_ftrs, 2)
        self.fc3 = nn.Linear(num_ftrs, 2)
        self.fc4 = nn.Linear(num_ftrs, 2)
        
        self.fc_fuse = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs*2, num_ftrs),
            nn.BatchNorm1d(num_ftrs),
        )
        
        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x1, x2, x3):
        # CWT
        h1 = self.features1(x1)
        regmap1 =  self.avgpool8(h1).squeeze(-1).squeeze(-1)
        logits1 = self.fc1(regmap1)
        
        # STmap
        h2 = self.features2(x2)
        regmap2 =  self.avgpool8(h2).squeeze(-1).squeeze(-1)
        logits2 = self.fc2(regmap2)
        
        # RGB
        h3 = self.features3(x3)
        regmap3 =  self.avgpool8(h3).squeeze(-1).squeeze(-1)
        logits3 = self.fc3(regmap3)
        
        # Concat fused CWT and STmap
        h_fuse = torch.cat((regmap1,regmap2), dim=1) 
        featureFused = self.fc_fuse(h_fuse) 
        
        # Cross-Atten CVT&STmap and RGB
        featureFused_temp = featureFused.unsqueeze(-1)
        featureFace_temp = regmap3.unsqueeze(-1)
        crossh1 = featureFused_temp @ featureFace_temp.transpose(-2, -1)
        crossh1 =F.softmax(crossh1, dim=-1)
        featureAtten = (crossh1 @ featureFace_temp).squeeze(-1)
          
        #logits_fuse2 = self.fc4(featureAtten)
        #pdb.set_trace()
        logits_fuse2 = self.fc4(regmap3+featureAtten)
        
        return logits1, logits2, logits3, logits_fuse2




class ResNet50_RGB_STmap_LastCWT(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNet50_RGB_STmap_LastCWT, self).__init__()
        resnet1 = models.resnet50(pretrained=pretrained)
        #resnet1 = models.resnet18(pretrained=pretrained)
        
        resnet2 = models.resnet50(pretrained=pretrained)
        #resnet2 = models.resnet18(pretrained=pretrained)
        
        resnet3 = models.resnet50(pretrained=pretrained)
        #resnet3 = models.resnet18(pretrained=pretrained)

        num_ftrs = resnet1.fc.in_features
        
        self.features1 = nn.Sequential(*list(resnet1.children())[:-2])
        
        self.features2 = nn.Sequential(*list(resnet2.children())[:-2])
        
        self.features3 = nn.Sequential(*list(resnet3.children())[:-2])

        #  binary CE
        self.fc1 = nn.Linear(num_ftrs, 2)
        self.fc2 = nn.Linear(num_ftrs, 2)
        self.fc3 = nn.Linear(num_ftrs, 2)
        self.fc4 = nn.Linear(num_ftrs, 2)
        self.fc5 = nn.Linear(num_ftrs, 2)      
        
        
        self.fc_fuse = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs*2, num_ftrs),
            nn.BatchNorm1d(num_ftrs),
        )
        
        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x1, x2, x3):
        # CWT
        h1 = self.features1(x1)
        regmap1 =  self.avgpool8(h1).squeeze(-1).squeeze(-1)
        logits1 = self.fc1(regmap1)
        
        # STmap
        h2 = self.features2(x2)
        regmap2 =  self.avgpool8(h2).squeeze(-1).squeeze(-1)
        logits2 = self.fc2(regmap2)
        
        # RGB
        h3 = self.features3(x3)
        regmap3 =  self.avgpool8(h3).squeeze(-1).squeeze(-1)
        logits3 = self.fc3(regmap3)
        
        # Concat fused CWT and STmap
        h_fuse = torch.cat((regmap1,regmap2), dim=1) 
        featureFused = self.fc_fuse(h_fuse) 
        logits_fuse = self.fc5(featureFused)
        
        # Cross-Atten CVT&STmap and RGB
        featureFused_temp = featureFused.unsqueeze(-1)
        featureFace_temp = regmap3.unsqueeze(-1)
        crossh1 = featureFused_temp @ featureFace_temp.transpose(-2, -1)
        crossh1 =F.softmax(crossh1, dim=-1)
        featureAtten = (crossh1 @ featureFace_temp).squeeze(-1)
          
        #logits_fuse2 = self.fc4(featureAtten)
        #pdb.set_trace()
        logits_fuse2 = self.fc4(regmap3+featureAtten)
        
        return logits1, logits2, logits_fuse, logits3, logits_fuse2




class ResNet50_18_RGB_STmap_LastCWT(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNet50_18_RGB_STmap_LastCWT, self).__init__()
        resnet1 = models.resnet18(pretrained=pretrained)
        #resnet1 = models.resnet18(pretrained=pretrained)
        
        resnet2 = models.resnet18(pretrained=pretrained)
        #resnet2 = models.resnet18(pretrained=pretrained)
        
        resnet3 = models.resnet50(pretrained=pretrained)
        #resnet3 = models.resnet18(pretrained=pretrained)

        num_ftrs = resnet1.fc.in_features
        num_ftrsR50 = resnet3.fc.in_features
        
        self.features1 = nn.Sequential(*list(resnet1.children())[:-2])
        
        self.features2 = nn.Sequential(*list(resnet2.children())[:-2])
        
        self.features3 = nn.Sequential(*list(resnet3.children())[:-2])

        #  binary CE
        self.fc1 = nn.Linear(num_ftrs, 2)
        self.fc2 = nn.Linear(num_ftrs, 2)
        self.fc3 = nn.Linear(num_ftrsR50, 2)
        self.fc4 = nn.Linear(num_ftrsR50, 2)
        self.fc5 = nn.Linear(num_ftrsR50, 2)      
        
        
        self.fc_fuse = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs*2, num_ftrsR50),
            nn.BatchNorm1d(num_ftrsR50),
        )
        
        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x1, x2, x3):
        # CWT
        h1 = self.features1(x1)
        regmap1 =  self.avgpool8(h1).squeeze(-1).squeeze(-1)
        logits1 = self.fc1(regmap1)
        
        # STmap
        h2 = self.features2(x2)
        regmap2 =  self.avgpool8(h2).squeeze(-1).squeeze(-1)
        logits2 = self.fc2(regmap2)
        
        # RGB
        h3 = self.features3(x3)
        regmap3 =  self.avgpool8(h3).squeeze(-1).squeeze(-1)
        logits3 = self.fc3(regmap3)
        
        # Concat fused CWT and STmap
        h_fuse = torch.cat((regmap1,regmap2), dim=1) 
        featureFused = self.fc_fuse(h_fuse) 
        logits_fuse = self.fc5(featureFused)
        
        # Cross-Atten CVT&STmap and RGB
        featureFused_temp = featureFused.unsqueeze(-1)
        featureFace_temp = regmap3.unsqueeze(-1)
        crossh1 = featureFused_temp @ featureFace_temp.transpose(-2, -1)
        crossh1 =F.softmax(crossh1, dim=-1)
        featureAtten = (crossh1 @ featureFace_temp).squeeze(-1)
          
        #logits_fuse2 = self.fc4(featureAtten)
        #pdb.set_trace()
        logits_fuse2 = self.fc4(regmap3+featureAtten)
        
        return logits1, logits2, logits_fuse, logits3, logits_fuse2
        

class ResNet50_18_Concat_RGB_STmap_LastCWT(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNet50_18_Concat_RGB_STmap_LastCWT, self).__init__()
        resnet1 = models.resnet18(pretrained=pretrained)
        #resnet1 = models.resnet18(pretrained=pretrained)
        
        resnet2 = models.resnet18(pretrained=pretrained)
        #resnet2 = models.resnet18(pretrained=pretrained)
        
        resnet3 = models.resnet50(pretrained=pretrained)
        #resnet3 = models.resnet18(pretrained=pretrained)

        num_ftrs = resnet1.fc.in_features
        num_ftrsR50 = resnet3.fc.in_features
        
        self.features1 = nn.Sequential(*list(resnet1.children())[:-2])
        
        self.features2 = nn.Sequential(*list(resnet2.children())[:-2])
        
        self.features3 = nn.Sequential(*list(resnet3.children())[:-2])

        #  binary CE
        self.fc1 = nn.Linear(num_ftrs, 2)
        self.fc2 = nn.Linear(num_ftrs, 2)
        self.fc3 = nn.Linear(num_ftrsR50, 2)
        #self.fc4 = nn.Linear(num_ftrsR50+num_ftrs, 2)
        self.fc5 = nn.Linear(num_ftrs, 2)      
        
        
        self.fc_fuse = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs*2, num_ftrs),
            nn.BatchNorm1d(num_ftrs),
        )
        
        self.fc_fuse2 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrsR50+num_ftrs, num_ftrs),
            nn.ReLU(), 
            nn.Linear(num_ftrs, 2),
        )
        
        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x1, x2, x3):
        # CWT
        h1 = self.features1(x1)
        regmap1 =  self.avgpool8(h1).squeeze(-1).squeeze(-1)
        logits1 = self.fc1(regmap1)
        
        # STmap
        h2 = self.features2(x2)
        regmap2 =  self.avgpool8(h2).squeeze(-1).squeeze(-1)
        logits2 = self.fc2(regmap2)
        
        # RGB
        h3 = self.features3(x3)
        regmap3 =  self.avgpool8(h3).squeeze(-1).squeeze(-1)
        logits3 = self.fc3(regmap3)
        
        # Concat fused CWT and STmap
        h_fuse = torch.cat((regmap1,regmap2), dim=1) 
        featureFused = self.fc_fuse(h_fuse) 
        logits_fuse = self.fc5(featureFused)
        
        # Concat fusion CVT&STmap and RGB
        featureAtten = torch.cat((featureFused, regmap3), dim=1) 
        logits_fuse2 = self.fc_fuse2(featureAtten)
        
        return logits1, logits2, logits_fuse, logits3, logits_fuse2




class ResNet50_18_SEConcat_RGB_STmap_LastCWT(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNet50_18_SEConcat_RGB_STmap_LastCWT, self).__init__()
        resnet1 = models.resnet18(pretrained=pretrained)
        
        resnet2 = models.resnet18(pretrained=pretrained)
        
        resnet3 = models.resnet50(pretrained=pretrained)

        num_ftrs = resnet1.fc.in_features
        num_ftrsR50 = resnet3.fc.in_features
        
        self.features1 = nn.Sequential(*list(resnet1.children())[:-2])
        
        self.features2 = nn.Sequential(*list(resnet2.children())[:-2])
        
        self.features3 = nn.Sequential(*list(resnet3.children())[:-2])

        #  binary CE
        self.fc1 = nn.Linear(num_ftrs, 2)
        self.fc2 = nn.Linear(num_ftrs, 2)
        self.fc3 = nn.Linear(num_ftrsR50, 2)
        #self.fc4 = nn.Linear(num_ftrsR50+num_ftrs, 2)
        self.fc5 = nn.Linear(num_ftrs, 2)      
        
        self.SEfuse1 = SELayer(num_ftrs)
        self.SEfuse2 = SELayer(num_ftrsR50)
        
        self.fc_fuse = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs*2, num_ftrs),
            nn.BatchNorm1d(num_ftrs),
        )
        
        self.fc_fuse2 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrsR50+num_ftrs, num_ftrs),
            nn.ReLU(), 
            nn.Linear(num_ftrs, 2),
        )
        
        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x1, x2, x3):
        # CWT
        h1 = self.features1(x1)
        regmap1 =  self.avgpool8(h1).squeeze(-1).squeeze(-1)
        logits1 = self.fc1(regmap1)
        
        # STmap
        h2 = self.features2(x2)
        regmap2 =  self.avgpool8(h2).squeeze(-1).squeeze(-1)
        logits2 = self.fc2(regmap2)
        
        # RGB
        h3 = self.features3(x3)
        regmap3 =  self.avgpool8(h3).squeeze(-1).squeeze(-1)
        logits3 = self.fc3(regmap3)
        
        # Concat fused CWT and STmap
        h_fuse = torch.cat((regmap1,regmap2), dim=1) 
        featureFused = self.fc_fuse(h_fuse) 
        logits_fuse = self.fc5(featureFused)
        
        # Concat fusion CVT&STmap and RGB
        featureAtten = torch.cat((self.SEfuse1(featureFused), self.SEfuse2(regmap3)), dim=1) 
        logits_fuse2 = self.fc_fuse2(featureAtten)
        
        return logits1, logits2, logits_fuse, logits3, logits_fuse2





# Best
# Try this one
# no need to fuse in the CWT and STmap
class ResNet50_18_Concat_AdapNorm2_RGB_STmap_LastCWT(nn.Module):

    def __init__(self, pretrained=True, theta=0.5):
        super(ResNet50_18_Concat_AdapNorm2_RGB_STmap_LastCWT, self).__init__()
        resnet1 = models.resnet18(pretrained=pretrained)
        
        resnet2 = models.resnet18(pretrained=pretrained)
        
        resnet3 = models.resnet50(pretrained=pretrained)

        num_ftrs = resnet1.fc.in_features
        num_ftrsR50 = resnet3.fc.in_features
        
        self.features1 = nn.Sequential(*list(resnet1.children())[:-2])
        
        self.features2 = nn.Sequential(*list(resnet2.children())[:-2])
        
        self.features3 = nn.Sequential(*list(resnet3.children())[:-2])

        #  binary CE
        self.fc1 = nn.Linear(num_ftrs, 2)
        self.fc2 = nn.Linear(num_ftrs, 2)
        self.fc3 = nn.Linear(num_ftrsR50, 2)
        #self.fc4 = nn.Linear(num_ftrsR50+num_ftrs, 2)
        self.fc5 = nn.Linear(num_ftrs, 2)      
        
        
        self.IN1 = nn.LayerNorm(num_ftrs, eps=1e-6)
        self.IN2 = nn.LayerNorm(num_ftrsR50, eps=1e-6)
        self.IN3 = nn.LayerNorm(num_ftrs, eps=1e-6)
        self.BN1 = nn.BatchNorm1d(num_ftrs)
        self.BN2 = nn.BatchNorm1d(num_ftrsR50)
        self.BN3 = nn.BatchNorm1d(num_ftrs)
        
        self.theta = theta
        
        self.fc_fuse = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs*2, num_ftrs),
            #nn.BatchNorm1d(num_ftrs),
        )
        
        self.fc_fuse2 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrsR50+num_ftrs, num_ftrs),
        )
        
        self.fc_fuse2_2 = nn.Sequential(
            nn.ReLU(), 
            nn.Linear(num_ftrs, 2),
        )
        
        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x1, x2, x3):
        # CWT
        h1 = self.features1(x1)
        regmap1 =  self.avgpool8(h1).squeeze(-1).squeeze(-1)
        logits1 = self.fc1(regmap1)
        
        # STmap
        h2 = self.features2(x2)
        regmap2 =  self.avgpool8(h2).squeeze(-1).squeeze(-1)
        logits2 = self.fc2(regmap2)
        
        # RGB
        h3 = self.features3(x3)
        regmap3 =  self.avgpool8(h3).squeeze(-1).squeeze(-1)
        logits3 = self.fc3(regmap3)
        
        # Concat fused CWT and STmap
        h_fuse = torch.cat((regmap1,regmap2), dim=1) 
        featureFused = self.fc_fuse(h_fuse) 
        logits_fuse = self.fc5(featureFused)
        
        # Concat fusion CVT&STmap and RGB
        #featureAtten = torch.cat((featureFused, h3), dim=1) 
        featureAtten = torch.cat((self.theta*self.IN1(featureFused) + (1-self.theta)*self.BN1(featureFused), self.theta*self.IN2(regmap3) + (1-self.theta)*self.BN2(regmap3)), dim=1) 
        logits_fuse2_temp = self.fc_fuse2(featureAtten)
        logits_fuse2 = self.fc_fuse2_2(self.theta*self.IN3(logits_fuse2_temp) + (1-self.theta)*self.BN3(logits_fuse2_temp))
        
        return logits1, logits2, logits_fuse, logits3, logits_fuse2



class ResNet50_18_Concat_Add_RGB_STmap_LastCWT(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNet50_18_Concat_Add_RGB_STmap_LastCWT, self).__init__()
        resnet1 = models.resnet18(pretrained=pretrained)
        
        resnet2 = models.resnet18(pretrained=pretrained)
        
        resnet3 = models.resnet50(pretrained=pretrained)

        num_ftrs = resnet1.fc.in_features
        num_ftrsR50 = resnet3.fc.in_features
        
        self.features1 = nn.Sequential(*list(resnet1.children())[:-2])
        
        self.features2 = nn.Sequential(*list(resnet2.children())[:-2])
        
        self.features3 = nn.Sequential(*list(resnet3.children())[:-2])

        #  binary CE
        self.fc1 = nn.Linear(num_ftrs, 2)
        self.fc2 = nn.Linear(num_ftrs, 2)
        self.fc3 = nn.Linear(num_ftrsR50, 2)
        #self.fc4 = nn.Linear(num_ftrsR50+num_ftrs, 2)
        self.fc5 = nn.Linear(num_ftrs, 2)      
        
        
        self.fc_fuse = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs*2, num_ftrs),
            nn.BatchNorm1d(num_ftrs),
        )
        
        self.fc_fuse2 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrsR50+num_ftrs, num_ftrsR50),
            #nn.ReLU(), 
        )
        
        self.fc_fuse3 = nn.Linear(num_ftrsR50, 2)
        
        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x1, x2, x3):
        # CWT
        h1 = self.features1(x1)
        regmap1 =  self.avgpool8(h1).squeeze(-1).squeeze(-1)
        logits1 = self.fc1(regmap1)
        
        # STmap
        h2 = self.features2(x2)
        regmap2 =  self.avgpool8(h2).squeeze(-1).squeeze(-1)
        logits2 = self.fc2(regmap2)
        
        # RGB
        h3 = self.features3(x3)
        regmap3 =  self.avgpool8(h3).squeeze(-1).squeeze(-1)
        logits3 = self.fc3(regmap3)
        
        # Concat fused CWT and STmap
        h_fuse = torch.cat((regmap1,regmap2), dim=1) 
        featureFused = self.fc_fuse(h_fuse) 
        logits_fuse = self.fc5(featureFused)
        
        # Concat fusion CVT&STmap and RGB
        featureAtten = torch.cat((featureFused, regmap3), dim=1) 
        logits_fuse2 = self.fc_fuse2(featureAtten)
        
        logits_fuse3 = self.fc_fuse3(logits_fuse2+regmap3)
        
        return logits1, logits2, logits_fuse, logits3, logits_fuse3



class ViT_STmap_LastCWT(nn.Module):

    def __init__(self, pretrained=True):
        super(ViT_STmap_LastCWT, self).__init__()
        
        self.vit1 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        self.vit2 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)

        num_ftrs = 768

        #  binary CE
        self.fc1 = nn.Linear(num_ftrs, 2)
        self.fc2 = nn.Linear(num_ftrs, 2)
        
        self.fc_fuse = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs*2, num_ftrs//2),
            nn.ReLU(),    
            nn.Dropout(0.5),
            nn.Linear(num_ftrs//2, 2),   
        )
        

    def forward(self, x1, x2):
        h1 = self.vit1.forward_features(x1)
        logits1 = self.fc1(h1)
        
        h2 = self.vit2.forward_features(x2)
        logits2 = self.fc2(h2)
        
        h_fuse = torch.cat((h1,h2), dim=1) 
        logitsFused = self.fc_fuse(h_fuse)   
        
        #return logits, regmap8
        return logits1, logits2, logitsFused




class ViT_STmap_LastCWT_3class(nn.Module):

    def __init__(self, pretrained=True):
        super(ViT_STmap_LastCWT_3class, self).__init__()
        
        self.vit1 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=3)
        self.vit2 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=3)

        num_ftrs = 768

        #  binary CE
        self.fc1 = nn.Linear(num_ftrs, 3)
        self.fc2 = nn.Linear(num_ftrs, 3)
        
        self.fc_fuse = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs*2, num_ftrs//2),
            nn.ReLU(),    
            nn.Dropout(0.5),
            nn.Linear(num_ftrs//2, 3),   
        )
        

    def forward(self, x1, x2):
        h1 = self.vit1.forward_features(x1)
        logits1 = self.fc1(h1)
        
        h2 = self.vit2.forward_features(x2)
        logits2 = self.fc2(h2)
        
        h_fuse = torch.cat((h1,h2), dim=1) 
        logitsFused = self.fc_fuse(h_fuse)   
        
        #return logits, regmap8
        return logits1, logits2, logitsFused



class ViT_STmap_LastCWT_2heads(nn.Module):

    def __init__(self, pretrained=True):
        super(ViT_STmap_LastCWT_2heads, self).__init__()
        
        self.vit1 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        self.vit2 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)

        num_ftrs = 768

        #  binary CE
        self.fc1_1 = nn.Linear(num_ftrs, 2)
        self.fc1_2 = nn.Linear(num_ftrs, 2)
        
        self.fc2_1 = nn.Linear(num_ftrs, 2)
        self.fc2_2 = nn.Linear(num_ftrs, 2)
        
        self.fc_fuse = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs*2, num_ftrs//2),
            nn.ReLU(),    
            nn.Dropout(0.5),
            #nn.Linear(num_ftrs//2, 2),   
        )
        
        self.fc_final1 = nn.Linear(num_ftrs//2, 2)
        self.fc_final2 = nn.Linear(num_ftrs//2, 2)
        

    def forward(self, x1, x2):
        h1 = self.vit1.forward_features(x1)
        logits1_1 = self.fc1_1(h1)
        logits1_2 = self.fc1_2(h1)
        
        h2 = self.vit2.forward_features(x2)
        logits2_1 = self.fc2_1(h2)
        logits2_2 = self.fc2_2(h2)
        
        h_fuse = torch.cat((h1,h2), dim=1) 
        logitsFused = self.fc_fuse(h_fuse)
        
        logits1_final = self.fc_final1(logitsFused)
        logits2_final = self.fc_final2(logitsFused)   
        
        #return logits, regmap8
        return logits1_1, logits1_2, logits2_1, logits2_2, logits1_final, logits2_final




class ViT_RGB_STmap_LastCWT(nn.Module):

    def __init__(self, pretrained=True):
        super(ViT_RGB_STmap_LastCWT, self).__init__()
        self.vit1 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        
        self.vit2 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        
        self.vit3 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)

        num_ftrs = 768
        

        #  binary CE
        self.fc1 = nn.Linear(num_ftrs, 2)
        self.fc2 = nn.Linear(num_ftrs, 2)
        self.fc3 = nn.Linear(num_ftrs, 2)
        self.fc4 = nn.Linear(num_ftrs, 2)
        self.fc5 = nn.Linear(num_ftrs, 2)      
        
        
        self.fc_fuse = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs*2, num_ftrs),
            nn.BatchNorm1d(num_ftrs),
        )
        
        #self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x1, x2, x3):
        # CWT
        h1 = self.vit1.forward_features(x1)
        logits1 = self.fc1(h1)
        
        # STmap
        h2 = self.vit2.forward_features(x2)
        logits2 = self.fc2(h2)
        
        # RGB
        h3 = self.vit3.forward_features(x3)
        logits3 = self.fc3(h3)
        
        # Concat fused CWT and STmap
        h_fuse = torch.cat((h1,h2), dim=1) 
        featureFused = self.fc_fuse(h_fuse) 
        logits_fuse = self.fc5(featureFused)
        
        # Cross-Atten CVT&STmap and RGB
        featureFused_temp = featureFused.unsqueeze(-1)
        featureFace_temp = h3.unsqueeze(-1)
        crossh1 = featureFused_temp @ featureFace_temp.transpose(-2, -1)
        crossh1 =F.softmax(crossh1, dim=-1)
        featureAtten = (crossh1 @ featureFace_temp).squeeze(-1)
          
        #logits_fuse2 = self.fc4(featureAtten)
        #pdb.set_trace()
        logits_fuse2 = self.fc4(h3+featureAtten)
        
        return logits1, logits2, logits_fuse, logits3, logits_fuse2
        

class ViT_Concat_RGB_STmap_LastCWT(nn.Module):

    def __init__(self, pretrained=True):
        super(ViT_Concat_RGB_STmap_LastCWT, self).__init__()
        self.vit1 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        
        self.vit2 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        
        self.vit3 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)

        num_ftrs = 768

        #  binary CE
        self.fc1 = nn.Linear(num_ftrs, 2)
        self.fc2 = nn.Linear(num_ftrs, 2)
        self.fc3 = nn.Linear(num_ftrs, 2)
        #self.fc4 = nn.Linear(num_ftrsR50+num_ftrs, 2)
        self.fc5 = nn.Linear(num_ftrs, 2)      
        
        
        self.fc_fuse = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs*2, num_ftrs),
            nn.BatchNorm1d(num_ftrs),
        )
        
        self.fc_fuse2 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs+num_ftrs, num_ftrs),
            nn.ReLU(), 
            nn.Linear(num_ftrs, 2),
        )
        
        #self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x1, x2, x3):
        # CWT
        h1 = self.vit1.forward_features(x1)
        logits1 = self.fc1(h1)
        
        # STmap
        h2 = self.vit2.forward_features(x2)
        logits2 = self.fc2(h2)
        
        # RGB
        h3 = self.vit3.forward_features(x3)
        logits3 = self.fc3(h3)
        
        # Concat fused CWT and STmap
        h_fuse = torch.cat((h1,h2), dim=1) 
        featureFused = self.fc_fuse(h_fuse) 
        logits_fuse = self.fc5(featureFused)
        
        # Concat fusion CVT&STmap and RGB
        featureAtten = torch.cat((featureFused, h3), dim=1) 
        logits_fuse2 = self.fc_fuse2(featureAtten)
        
        return logits1, logits2, logits_fuse, logits3, logits_fuse2



class ViT_SEConcat_RGB_STmap_LastCWT(nn.Module):

    def __init__(self, pretrained=True):
        super(ViT_SEConcat_RGB_STmap_LastCWT, self).__init__()
        self.vit1 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        
        self.vit2 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        
        self.vit3 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)

        num_ftrs = 768

        #  binary CE
        self.fc1 = nn.Linear(num_ftrs, 2)
        self.fc2 = nn.Linear(num_ftrs, 2)
        self.fc3 = nn.Linear(num_ftrs, 2)
        #self.fc4 = nn.Linear(num_ftrsR50+num_ftrs, 2)
        self.fc5 = nn.Linear(num_ftrs, 2)      
        
        
        self.fc_fuse = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs*2, num_ftrs),
            nn.BatchNorm1d(num_ftrs),
        )
        
        self.fc_fuse2 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs+num_ftrs, num_ftrs),
            nn.ReLU(), 
            nn.Linear(num_ftrs, 2),
        )
        
        self.SEfuse1 = SELayer(num_ftrs)
        self.SEfuse2 = SELayer(num_ftrs)
        
        #self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x1, x2, x3):
        # CWT
        h1 = self.vit1.forward_features(x1)
        logits1 = self.fc1(h1)
        
        # STmap
        h2 = self.vit2.forward_features(x2)
        logits2 = self.fc2(h2)
        
        # RGB
        h3 = self.vit3.forward_features(x3)
        logits3 = self.fc3(h3)
        
        # Concat fused CWT and STmap
        h_fuse = torch.cat((h1,h2), dim=1) 
        featureFused = self.fc_fuse(h_fuse) 
        logits_fuse = self.fc5(featureFused)
        
        # Concat fusion CVT&STmap and RGB
        featureAtten = torch.cat((self.SEfuse1(featureFused), self.SEfuse2(h3)), dim=1) 
        logits_fuse2 = self.fc_fuse2(featureAtten)
        
        return logits1, logits2, logits_fuse, logits3, logits_fuse2


class ViT_Concat_AdapNorm_RGB_STmap_LastCWT(nn.Module):

    def __init__(self, pretrained=True, theta=0.5):
        super(ViT_Concat_AdapNorm_RGB_STmap_LastCWT, self).__init__()
        self.vit1 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        
        self.vit2 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        
        self.vit3 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)

        num_ftrs = 768

        #  binary CE
        self.fc1 = nn.Linear(num_ftrs, 2)
        self.fc2 = nn.Linear(num_ftrs, 2)
        self.fc3 = nn.Linear(num_ftrs, 2)
        #self.fc4 = nn.Linear(num_ftrsR50+num_ftrs, 2)
        self.fc5 = nn.Linear(num_ftrs, 2)      
        
        self.INh1 = nn.LayerNorm(num_ftrs, eps=1e-6)
        self.INh2 = nn.LayerNorm(num_ftrs, eps=1e-6)
        self.IN1 = nn.LayerNorm(num_ftrs, eps=1e-6)
        self.IN2 = nn.LayerNorm(num_ftrs, eps=1e-6)
        self.IN3 = nn.LayerNorm(num_ftrs, eps=1e-6)
        self.BNh1 = nn.BatchNorm1d(num_ftrs)
        self.BNh2 = nn.BatchNorm1d(num_ftrs)
        self.BN1 = nn.BatchNorm1d(num_ftrs)
        self.BN2 = nn.BatchNorm1d(num_ftrs)
        self.BN3 = nn.BatchNorm1d(num_ftrs)
        
        self.theta = theta
        
        self.fc_fuse = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs*2, num_ftrs),
            #nn.BatchNorm1d(num_ftrs),
        )
        
        self.fc_fuse2 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs+num_ftrs, num_ftrs),
        )
        
        self.fc_fuse2_2 = nn.Sequential(
            nn.ReLU(), 
            nn.Linear(num_ftrs, 2),
        )
        
        #self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x1, x2, x3):
        # CWT
        h1 = self.vit1.forward_features(x1)
        logits1 = self.fc1(h1)
        
        # STmap
        h2 = self.vit2.forward_features(x2)
        logits2 = self.fc2(h2)
        
        # RGB
        h3 = self.vit3.forward_features(x3)
        logits3 = self.fc3(h3)
        
        # Concat fused CWT and STmap
        h1_norm = self.theta*self.INh1(h1) + (1-self.theta)*self.BNh1(h1)
        h2_norm = self.theta*self.INh2(h2) + (1-self.theta)*self.BNh2(h2)
        h_fuse = torch.cat((h1_norm,h2_norm), dim=1) 
        featureFused = self.fc_fuse(h_fuse) 
        logits_fuse = self.fc5(featureFused)
        
        # Concat fusion CVT&STmap and RGB
        #featureAtten = torch.cat((featureFused, h3), dim=1) 
        featureAtten = torch.cat((self.theta*self.IN1(featureFused) + (1-self.theta)*self.BN1(featureFused), self.theta*self.IN2(h3) + (1-self.theta)*self.BN2(h3)), dim=1) 
        logits_fuse2_temp = self.fc_fuse2(featureAtten)
        logits_fuse2 = self.fc_fuse2_2(self.theta*self.IN3(logits_fuse2_temp) + (1-self.theta)*self.BN3(logits_fuse2_temp))
        
        return logits1, logits2, logits_fuse, logits3, logits_fuse2



# Best
# Try this one
# no need to fuse in the CWT and STmap
class ViT_Concat_AdapNorm2_RGB_STmap_LastCWT(nn.Module):

    def __init__(self, pretrained=True, theta=0.5):
        super(ViT_Concat_AdapNorm2_RGB_STmap_LastCWT, self).__init__()
        self.vit1 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        
        self.vit2 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        
        self.vit3 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)

        num_ftrs = 768

        #  binary CE
        self.fc1 = nn.Linear(num_ftrs, 2)
        self.fc2 = nn.Linear(num_ftrs, 2)
        self.fc3 = nn.Linear(num_ftrs, 2)
        #self.fc4 = nn.Linear(num_ftrsR50+num_ftrs, 2)
        self.fc5 = nn.Linear(num_ftrs, 2)      
        
        
        self.IN1 = nn.LayerNorm(num_ftrs, eps=1e-6)
        self.IN2 = nn.LayerNorm(num_ftrs, eps=1e-6)
        self.IN3 = nn.LayerNorm(num_ftrs, eps=1e-6)
        self.BN1 = nn.BatchNorm1d(num_ftrs)
        self.BN2 = nn.BatchNorm1d(num_ftrs)
        self.BN3 = nn.BatchNorm1d(num_ftrs)
        
        self.theta = theta
        
        self.fc_fuse = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs*2, num_ftrs),
            #nn.BatchNorm1d(num_ftrs),
        )
        
        self.fc_fuse2 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs+num_ftrs, num_ftrs),
        )
        
        self.fc_fuse2_2 = nn.Sequential(
            nn.ReLU(), 
            nn.Linear(num_ftrs, 2),
        )
        
        #self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x1, x2, x3):
        # CWT
        h1 = self.vit1.forward_features(x1)
        logits1 = self.fc1(h1)
        
        # STmap
        h2 = self.vit2.forward_features(x2)
        logits2 = self.fc2(h2)
        
        # RGB
        h3 = self.vit3.forward_features(x3)
        logits3 = self.fc3(h3)
        
        # Concat fused CWT and STmap
        h_fuse = torch.cat((h1,h2), dim=1) 
        featureFused = self.fc_fuse(h_fuse) 
        logits_fuse = self.fc5(featureFused)
        
        # Concat fusion CVT&STmap and RGB
        #featureAtten = torch.cat((featureFused, h3), dim=1) 
        featureAtten = torch.cat((self.theta*self.IN1(featureFused) + (1-self.theta)*self.BN1(featureFused), self.theta*self.IN2(h3) + (1-self.theta)*self.BN2(h3)), dim=1) 
        logits_fuse2_temp = self.fc_fuse2(featureAtten)
        logits_fuse2 = self.fc_fuse2_2(self.theta*self.IN3(logits_fuse2_temp) + (1-self.theta)*self.BN3(logits_fuse2_temp))
        
        return logits1, logits2, logits_fuse, logits3, logits_fuse2


class ViT_Concat_AdapNorm4_RGB_STmap_LastCWT(nn.Module):

    def __init__(self, pretrained=True):
        super(ViT_Concat_AdapNorm4_RGB_STmap_LastCWT, self).__init__()
        self.vit1 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        
        self.vit2 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        
        self.vit3 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)

        num_ftrs = 768

        #  binary CE
        self.fc1 = nn.Linear(num_ftrs, 2)
        self.fc2 = nn.Linear(num_ftrs, 2)
        self.fc3 = nn.Linear(num_ftrs, 2)
        #self.fc4 = nn.Linear(num_ftrsR50+num_ftrs, 2)
        self.fc5 = nn.Linear(num_ftrs, 2)      
        
        
        self.IN1 = nn.LayerNorm(num_ftrs, eps=1e-6)
        self.IN2 = nn.LayerNorm(num_ftrs, eps=1e-6)
        self.IN3 = nn.LayerNorm(num_ftrs, eps=1e-6)
        self.BN1 = nn.BatchNorm1d(num_ftrs)
        self.BN2 = nn.BatchNorm1d(num_ftrs)
        self.BN3 = nn.BatchNorm1d(num_ftrs)
        
        self.INBN1_weights = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs//16),
            nn.ReLU(), 
            nn.Linear(num_ftrs//16, num_ftrs),
            nn.Sigmoid(),
        )
        
        self.INBN2_weights = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs//16),
            nn.ReLU(), 
            nn.Linear(num_ftrs//16, num_ftrs),
            nn.Sigmoid(),
        )
        
        self.INBN3_weights = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs//16),
            nn.ReLU(), 
            nn.Linear(num_ftrs//16, num_ftrs),
            nn.Sigmoid(),
        )
        
        
        self.fc_fuse = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs*2, num_ftrs),
            #nn.BatchNorm1d(num_ftrs),
        )
        
        self.fc_fuse2 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs+num_ftrs, num_ftrs),
        )
        
        self.fc_fuse2_2 = nn.Sequential(
            nn.ReLU(), 
            nn.Linear(num_ftrs, 2),
        )
        
        #self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x1, x2, x3):
        # CWT
        h1 = self.vit1.forward_features(x1)
        logits1 = self.fc1(h1)
        
        # STmap
        h2 = self.vit2.forward_features(x2)
        logits2 = self.fc2(h2)
        
        # RGB
        h3 = self.vit3.forward_features(x3)
        logits3 = self.fc3(h3)
        
        # Concat fused CWT and STmap
        h_fuse = torch.cat((h1,h2), dim=1) 
        featureFused = self.fc_fuse(h_fuse) 
        logits_fuse = self.fc5(featureFused)
        
        # Concat fusion CVT&STmap and RGB
        #featureAtten = torch.cat((featureFused, h3), dim=1) 
        
        wIN1 = self.INBN1_weights(featureFused)
        wBN1 = 1.0-wIN1
        wIN2 = self.INBN2_weights(h3)
        wBN2 = 1.0-wIN2
        
        featureAtten = torch.cat((wIN1*self.IN1(featureFused)+wBN1*self.BN1(featureFused), wIN2*self.IN2(h3)+wBN2*self.BN2(h3)), dim=1) 
        
        logits_fuse2_temp = self.fc_fuse2(featureAtten)
        
        wIN3 = self.INBN3_weights(logits_fuse2_temp)
        wBN3 = 1.0-wIN3
        
        logits_fuse2 = self.fc_fuse2_2(wIN3*self.IN3(logits_fuse2_temp)+wBN3*self.BN3(logits_fuse2_temp))
        
        return logits1, logits2, logits_fuse, logits3, logits_fuse2



# no improvement
class ViT_Concat_CA_RGB_STmap_LastCWT(nn.Module):

    def __init__(self, pretrained=True):
        super(ViT_Concat_CA_RGB_STmap_LastCWT, self).__init__()
        self.vit1 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        
        self.vit2 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        
        self.vit3 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)

        num_ftrs = 768
        

        #  binary CE
        self.fc1 = nn.Linear(num_ftrs, 2)
        self.fc2 = nn.Linear(num_ftrs, 2)
        self.fc3 = nn.Linear(num_ftrs, 2)
        self.fc4 = nn.Linear(num_ftrs, 2)
        self.fc5 = nn.Linear(num_ftrs, 2)      
        
        
        self.fc_fuse = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs*2, num_ftrs),
            #nn.BatchNorm1d(num_ftrs),
            nn.LayerNorm(num_ftrs, eps=1e-6),
        )
        
        self.fc_fuse2 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs+num_ftrs, num_ftrs),
            #nn.BatchNorm1d(num_ftrs),
            nn.LayerNorm(num_ftrs, eps=1e-6),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, num_ftrs),
            #nn.BatchNorm1d(num_ftrs),
            nn.LayerNorm(num_ftrs, eps=1e-6),
        )
        
        #self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x1, x2, x3):
        # CWT
        h1 = self.vit1.forward_features(x1)
        logits1 = self.fc1(h1)
        
        # STmap
        h2 = self.vit2.forward_features(x2)
        logits2 = self.fc2(h2)
        
        # RGB
        h3 = self.vit3.forward_features(x3)
        logits3 = self.fc3(h3)
        
        # Concat fused CWT and STmap
        h_fuse = torch.cat((h1,h2), dim=1) 
        featureFused = self.fc_fuse(h_fuse) 
        logits_fuse = self.fc5(featureFused)
        
        # Concat fusion CVT&STmap and RGB
        featureCon = torch.cat((featureFused, h3), dim=1) 
        logits_fuse2 = self.fc_fuse2(featureCon)
        
        # Cross-Atten CVT&STmap and RGB
        featureFused_temp = logits_fuse2.unsqueeze(-1)
        featureFace_temp = h3.unsqueeze(-1)
        crossh1 = featureFused_temp @ featureFace_temp.transpose(-2, -1)
        crossh1 =F.softmax(crossh1, dim=-1)
        featureAtten = (crossh1 @ featureFace_temp).squeeze(-1)
          
        #logits_fuse2 = self.fc4(featureAtten)
        #pdb.set_trace()
        logits_fuse3 = self.fc4(h3+featureAtten)
        
        return logits1, logits2, logits_fuse, logits3, logits_fuse3



# slight changes improvement
class ViT_Concat_shrink_RGB_STmap_LastCWT(nn.Module):

    def __init__(self, pretrained=True, shrink=0.3):
        super(ViT_Concat_shrink_RGB_STmap_LastCWT, self).__init__()
        self.vit1 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        
        self.vit2 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        
        self.vit3 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)

        num_ftrs = 768
        num_ftrs_shrink = math.floor(768*shrink)

        #  binary CE
        self.fc1 = nn.Linear(num_ftrs, 2)
        self.fc2 = nn.Linear(num_ftrs, 2)
        self.fc3 = nn.Linear(num_ftrs, 2)
        #self.fc4 = nn.Linear(num_ftrsR50+num_ftrs, 2)
        self.fc5 = nn.Linear(num_ftrs_shrink, 2)      
        
        
        self.fc_fuse = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs*2, num_ftrs_shrink),
            nn.BatchNorm1d(num_ftrs_shrink),
        )
        
        self.fc_fuse2 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs+num_ftrs_shrink, (num_ftrs+num_ftrs_shrink)//2),
            nn.ReLU(), 
            nn.Linear((num_ftrs+num_ftrs_shrink)//2, 2),
        )
        
        #self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x1, x2, x3):
        # CWT
        h1 = self.vit1.forward_features(x1)
        logits1 = self.fc1(h1)
        
        # STmap
        h2 = self.vit2.forward_features(x2)
        logits2 = self.fc2(h2)
        
        # RGB
        h3 = self.vit3.forward_features(x3)
        logits3 = self.fc3(h3)
        
        # Concat fused CWT and STmap
        h_fuse = torch.cat((h1,h2), dim=1) 
        featureFused = self.fc_fuse(h_fuse) 
        logits_fuse = self.fc5(featureFused)
        
        # Concat fusion CVT&STmap and RGB
        featureAtten = torch.cat((featureFused, h3), dim=1) 
        logits_fuse2 = self.fc_fuse2(featureAtten)
        
        return logits1, logits2, logits_fuse, logits3, logits_fuse2


# no improvements
class ViT_Concat_Add_RGB_STmap_LastCWT(nn.Module):

    def __init__(self, pretrained=True):
        super(ViT_Concat_Add_RGB_STmap_LastCWT, self).__init__()
        self.vit1 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        
        self.vit2 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        
        self.vit3 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)

        num_ftrs = 768

        #  binary CE
        self.fc1 = nn.Linear(num_ftrs, 2)
        self.fc2 = nn.Linear(num_ftrs, 2)
        self.fc3 = nn.Linear(num_ftrs, 2)
        #self.fc4 = nn.Linear(num_ftrsR50+num_ftrs, 2)
        self.fc5 = nn.Linear(num_ftrs, 2)      
        
        
        self.fc_fuse = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs*2, num_ftrs),
            nn.BatchNorm1d(num_ftrs),
        )
        
        self.fc_fuse2 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs+num_ftrs, num_ftrs),
            #nn.ReLU(), 
        )
        
        self.fc_fuse3 = nn.Linear(num_ftrs, 2)

    def forward(self, x1, x2, x3):
        # CWT
        h1 = self.vit1.forward_features(x1)
        logits1 = self.fc1(h1)
        
        # STmap
        h2 = self.vit2.forward_features(x2)
        logits2 = self.fc2(h2)
        
        # RGB
        h3 = self.vit3.forward_features(x3)
        logits3 = self.fc3(h3)
        
        # Concat fused CWT and STmap
        h_fuse = torch.cat((h1,h2), dim=1) 
        featureFused = self.fc_fuse(h_fuse) 
        logits_fuse = self.fc5(featureFused)
        
        # Concat fusion CVT&STmap and RGB
        featureAtten = torch.cat((featureFused, h3), dim=1) 
        logits_fuse2 = self.fc_fuse2(featureAtten)
        
        logits_fuse3 = self.fc_fuse3(logits_fuse2+h3)
        
        return logits1, logits2, logits_fuse, logits3, logits_fuse3


# no improvement
class ViT_Concat_Add_Uncertainty_RGB_STmap_LastCWT(nn.Module):

    def __init__(self, pretrained=True):
        super(ViT_Concat_Add_Uncertainty_RGB_STmap_LastCWT, self).__init__()
        self.vit1 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        
        self.vit2 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        
        self.vit3 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)

        num_ftrs = 768

        #  binary CE
        self.fc1 = nn.Linear(num_ftrs, 2)
        self.fc2 = nn.Linear(num_ftrs, 2)
        self.fc3 = nn.Linear(num_ftrs, 2)
        #self.fc4 = nn.Linear(num_ftrsR50+num_ftrs, 2)
        self.fc5 = nn.Linear(num_ftrs, 2)      
        
        
        self.fc_fuse = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs*2, num_ftrs),
            nn.BatchNorm1d(num_ftrs),
        )
        
        self.fc_fuse2 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs+num_ftrs, num_ftrs),
            #nn.ReLU(), 
        )
        
        self.fc_fuse3 = nn.Linear(num_ftrs, 2)

    def forward(self, x1, x2, x3):
        # CWT
        h1 = self.vit1.forward_features(x1)
        logits1 = self.fc1(h1)
        
        # STmap
        h2 = self.vit2.forward_features(x2)
        logits2 = self.fc2(h2)
        
        # RGB
        h3 = self.vit3.forward_features(x3)
        logits3 = self.fc3(h3)
        
        # Concat fused CWT and STmap
        h_fuse = torch.cat((h1,h2), dim=1) 
        featureFused = self.fc_fuse(h_fuse) 
        logits_fuse = self.fc5(featureFused)
        
        # Concat fusion CVT&STmap and RGB
        featureAtten = torch.cat((featureFused, h3), dim=1) 
        logits_fuse2 = self.fc_fuse2(featureAtten)
        
        # Uncertainty
        logits1_softmax = F.softmax(logits1)
        logits2_softmax = F.softmax(logits2)
        logits_fuse_softmax = F.softmax(logits_fuse)
        Uncert = (torch.abs(logits1_softmax[:,0]-logits1_softmax[:,1])+torch.abs(logits2_softmax[:,0]-logits2_softmax[:,1])+torch.abs(logits_fuse_softmax[:,0]-logits_fuse_softmax[:,1]))/3.0
        
        #pdb.set_trace()
        logits_fuse3 = self.fc_fuse3(Uncert.unsqueeze(-1)*logits_fuse2+h3)
        
        return logits1, logits2, logits_fuse, logits3, logits_fuse3



# Input 224x224
class ResNet18_Input63_224(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNet18_Input63_224, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(63*3, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(),    
        )
        
        num_ftrs = resnet.fc.in_features
        
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        #  binary CE
        self.fc = nn.Linear(num_ftrs, 2)
        
        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x):
        [B, T, C, H, W]=x.shape
        x = x.view(B,-1,H,W)
        
        x = self.conv1(x)
        
        h = self.features(x)

        regmap8 =  self.avgpool8(h)
        
        logits = self.fc(regmap8.squeeze(-1).squeeze(-1))
        
        #return logits, regmap8
        return logits



class ResNet18_Input63_224_add(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNet18_Input63_224_add, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(63*3, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(),    
        )
        
        num_ftrs = resnet.fc.in_features
        
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        #  binary CE
        self.fc = nn.Linear(num_ftrs, 2)
        
        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x):
        [B, T, C, H, W]=x.shape
        x_temp = x.view(B,-1,H,W)
        
        x_temp = self.conv1(x_temp) + x[:,62,:,:,:]
        
        h = self.features(x_temp)

        regmap8 =  self.avgpool8(h)
        
        logits = self.fc(regmap8.squeeze(-1).squeeze(-1))
        
        #return logits, regmap8
        return logits



# Input 224x224
class ViT_Input63_224(nn.Module):

    def __init__(self, pretrained=True):
        super(ViT_Input63_224, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(63*3, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(),    
        )
        

    def forward(self, x):
        [B, T, C, H, W]=x.shape
        x = x.view(B,-1,H,W)
        
        x = self.conv1(x)
        
        logits = self.vit(x)

        return logits


class ViT_Input63_224_add(nn.Module):

    def __init__(self, pretrained=True):
        super(ViT_Input63_224_add, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(63*3, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(),    
        )
        

    def forward(self, x):
        [B, T, C, H, W]=x.shape
        x_temp = x.view(B,-1,H,W)
        
        x_temp = self.conv1(x_temp) + x[:,62,:,:,:]
        
        logits = self.vit(x_temp)

        return logits



# Input 112x112
class ResNet18_Input63(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNet18_Input63, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(63*3, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),    
        )
        
        num_ftrs = resnet.fc.in_features
        
        self.features = nn.Sequential(*list(resnet.children())[1:-2])

        #  binary CE
        self.fc = nn.Linear(num_ftrs, 2)
        
        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x):
        [B, T, C, H, W]=x.shape
        x = x.view(B,-1,H,W)
        
        x = self.conv1(x)
        
        h = self.features(x)

        regmap8 =  self.avgpool8(h)
        
        logits = self.fc(regmap8.squeeze(-1).squeeze(-1))
        
        #return logits, regmap8
        return logits




# Input 112x112
class ResNet18_Input63_32x96(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNet18_Input63_32x96, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(63*3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),    
        )
        
        num_ftrs = resnet.fc.in_features
        
        self.features = nn.Sequential(*list(resnet.children())[1:-2])

        #  binary CE
        self.fc = nn.Linear(num_ftrs, 2)
        
        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x):
        [B, T, C, H, W]=x.shape
        x = x.view(B,-1,H,W)
        
        x = self.conv1(x)
        
        h = self.features(x)

        regmap8 =  self.avgpool8(h)
        
        logits = self.fc(regmap8.squeeze(-1).squeeze(-1))
        
        #return logits, regmap8
        return logits



# Input 112x112
class ResNet18_Input63_64x192(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNet18_Input63_64x192, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(63*3, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),    
        )
        
        num_ftrs = resnet.fc.in_features
        
        self.features = nn.Sequential(*list(resnet.children())[1:-2])

        #  binary CE
        self.fc = nn.Linear(num_ftrs, 2)
        
        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x):
        [B, T, C, H, W]=x.shape
        x = x.view(B,-1,H,W)
        
        x = self.conv1(x)
        
        h = self.features(x)

        regmap8 =  self.avgpool8(h)
        
        logits = self.fc(regmap8.squeeze(-1).squeeze(-1))
        
        #return logits, regmap8
        return logits



class DenseNet161(nn.Module):

    def __init__(self, pretrained=True):
        super(DenseNet161, self).__init__()
        densenet = models.densenet161(pretrained=pretrained)

        #num_ftrs = densenet.fc.in_features
        
        self.features = densenet.features

        self.regressmap = nn.Conv2d(2208, 2, kernel_size=1, stride=1, padding=0, bias=False)
        
        #  binary CE
        #self.fc = nn.Linear(num_ftrs, 2)
        
        #self.avgpool7 = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x):
        h = self.features(x)
        
        
        
        map_binary = self.regressmap(h)
        
        #regmap7 =  self.avgpool7(h)
        
        #logits = self.fc(regmap7.squeeze(-1).squeeze(-1))
        
        #return logits, regmap8
        return map_binary





class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, dilation=self.conv.dilation, groups=self.conv.groups)

            return out_normal - self.theta * out_diff




 
        
class SpatialAttention(nn.Module):
    def __init__(self, kernel = 3):
        super(SpatialAttention, self).__init__()


        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel, padding=kernel//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        
        return self.sigmoid(x)



		

class CDCNpp(nn.Module):

    def __init__(self, basic_conv=Conv2d_cd, theta=0.7 ):   
        super(CDCNpp, self).__init__()
        
        
        self.conv1 = nn.Sequential(
            basic_conv(3, 80, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(80),
            nn.ReLU(),    
            
        )
        
        self.Block1 = nn.Sequential(
            basic_conv(80, 160, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),  
            
            basic_conv(160, int(160*1.6), kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(int(160*1.6)),
            nn.ReLU(),  
            basic_conv(int(160*1.6), 160, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(160),
            nn.ReLU(), 
            
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
        )
        
        self.Block2 = nn.Sequential(
            basic_conv(160, int(160*1.2), kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(int(160*1.2)),
            nn.ReLU(),  
            basic_conv(int(160*1.2), 160, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),  
            basic_conv(160, int(160*1.4), kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(int(160*1.4)),
            nn.ReLU(),  
            basic_conv(int(160*1.4), 160, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),  
            
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.Block3 = nn.Sequential(
            basic_conv(160, 160, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(160),
            nn.ReLU(), 
            basic_conv(160, int(160*1.2), kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(int(160*1.2)),
            nn.ReLU(),  
            basic_conv(int(160*1.2), 160, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(160),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        # Original
        
        self.lastconv1 = nn.Sequential(
            basic_conv(160*3, 160, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            basic_conv(160, 1, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.ReLU(),    
        )
        
      
        self.sa1 = SpatialAttention(kernel = 7)
        self.sa2 = SpatialAttention(kernel = 5)
        self.sa3 = SpatialAttention(kernel = 3)
        self.downsample28x28 = nn.Upsample(size=(28, 28), mode='bilinear')

 
    def forward(self, x):	    	# x [3, 256, 256]
        
        x_input = x
        x = self.conv1(x)		   
        
        x_Block1 = self.Block1(x)	    	    	
        attention1 = self.sa1(x_Block1)
        x_Block1_SA = attention1 * x_Block1
        x_Block1_32x32 = self.downsample28x28(x_Block1_SA)   
        
        x_Block2 = self.Block2(x_Block1)	    
        attention2 = self.sa2(x_Block2)  
        x_Block2_SA = attention2 * x_Block2
        x_Block2_32x32 = self.downsample28x28(x_Block2_SA)  
        
        x_Block3 = self.Block3(x_Block2)	    
        attention3 = self.sa3(x_Block3)  
        x_Block3_SA = attention3 * x_Block3	
        x_Block3_32x32 = self.downsample28x28(x_Block3_SA)   
        
        x_concat = torch.cat((x_Block1_32x32,x_Block2_32x32,x_Block3_32x32), dim=1)    
        
        #pdb.set_trace()
        
        map_x = self.lastconv1(x_concat)
        
        map_x = map_x.squeeze(1)
        
        return map_x



class MesoInception4(nn.Module):
	"""
	# input size 256x256
	"""
	def __init__(self, num_classes=2):
		super(MesoInception4, self).__init__()
		self.num_classes = num_classes
		#InceptionLayer1
		self.Incption1_conv1 = nn.Conv2d(3, 1, 1, padding=0, bias=False)
		self.Incption1_conv2_1 = nn.Conv2d(3, 4, 1, padding=0, bias=False)
		self.Incption1_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
		self.Incption1_conv3_1 = nn.Conv2d(3, 4, 1, padding=0, bias=False)
		self.Incption1_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
		self.Incption1_conv4_1 = nn.Conv2d(3, 2, 1, padding=0, bias=False)
		self.Incption1_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
		self.Incption1_bn = nn.BatchNorm2d(11)


		#InceptionLayer2
		self.Incption2_conv1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
		self.Incption2_conv2_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
		self.Incption2_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
		self.Incption2_conv3_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
		self.Incption2_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
		self.Incption2_conv4_1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
		self.Incption2_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
		self.Incption2_bn = nn.BatchNorm2d(12)

		#Normal Layer
		self.conv1 = nn.Conv2d(12, 16, 5, padding=2, bias=False)
		self.relu = nn.ReLU(inplace=True)
		self.leakyrelu = nn.LeakyReLU(0.1)
		self.bn1 = nn.BatchNorm2d(16)
		self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2))

		self.conv2 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
		self.maxpooling2 = nn.MaxPool2d(kernel_size=(4, 4))

		self.dropout = nn.Dropout2d(0.5)
		self.fc1 = nn.Linear(16*8*8, 16)
		self.fc2 = nn.Linear(16, num_classes)


	#InceptionLayer
	def InceptionLayer1(self, input):
		x1 = self.Incption1_conv1(input)
		x2 = self.Incption1_conv2_1(input)
		x2 = self.Incption1_conv2_2(x2)
		x3 = self.Incption1_conv3_1(input)
		x3 = self.Incption1_conv3_2(x3)
		x4 = self.Incption1_conv4_1(input)
		x4 = self.Incption1_conv4_2(x4)
		y = torch.cat((x1, x2, x3, x4), 1)
		y = self.Incption1_bn(y)
		y = self.maxpooling1(y)

		return y

	def InceptionLayer2(self, input):
		x1 = self.Incption2_conv1(input)
		x2 = self.Incption2_conv2_1(input)
		x2 = self.Incption2_conv2_2(x2)
		x3 = self.Incption2_conv3_1(input)
		x3 = self.Incption2_conv3_2(x3)
		x4 = self.Incption2_conv4_1(input)
		x4 = self.Incption2_conv4_2(x4)
		y = torch.cat((x1, x2, x3, x4), 1)
		y = self.Incption2_bn(y)
		y = self.maxpooling1(y)

		return y

	def forward(self, inputs):
		x = self.InceptionLayer1(inputs) #(Batch, 11, 128, 128)
		x = self.InceptionLayer2(x) #(Batch, 12, 64, 64)

		x = self.conv1(x) #(Batch, 16, 64 ,64)
		x = self.relu(x)
		x = self.bn1(x)
		x = self.maxpooling1(x) #(Batch, 16, 32, 32)

		x = self.conv2(x) #(Batch, 16, 32, 32)
		x = self.relu(x)
		x = self.bn1(x)
		x = self.maxpooling2(x) #(Batch, 16, 8, 8)

		x = x.view(x.size(0), -1) #(Batch, 16*8*8)
		x = self.dropout(x)
		x = self.fc1(x) #(Batch, 16)
		x = self.leakyrelu(x)
		x = self.dropout(x)
		x = self.fc2(x)

		return x


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


class xception(nn.Module):
    def __init__(self,num_classes=2,escape=''):
        super(xception, self).__init__()
        self.escape=escape
        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)
        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.relu4=nn.ReLU(inplace=True)
        self.last_linear = nn.Linear(2048, num_classes)
        self.seq=[]
        self.seq.append(('b0',[self.conv1,lambda x:self.bn1(x),self.relu1,self.conv2,lambda x:self.bn2(x)]))
        self.seq.append(('b1',[self.relu2,self.block1]))
        self.seq.append(('b2',[self.block2]))
        self.seq.append(('b3',[self.block3]))
        self.seq.append(('b4',[self.block4]))
        self.seq.append(('b5',[self.block5]))
        self.seq.append(('b6',[self.block6]))
        self.seq.append(('b7',[self.block7]))
        self.seq.append(('b8',[self.block8]))
        self.seq.append(('b9',[self.block9]))
        self.seq.append(('b10',[self.block10]))
        self.seq.append(('b11',[self.block11]))
        self.seq.append(('b12',[self.block12]))
        self.seq.append(('final',[self.conv3,lambda x:self.bn3(x),self.relu3,self.conv4,lambda x:self.bn4(x)]))
        self.seq.append(('logits',[self.relu4,lambda x:F.adaptive_avg_pool2d(x, (1, 1)),lambda x:x.view(x.size(0), -1),self.last_linear]))
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #-----------------------------

    def forward(self, x):
        layers={}
        for stage in self.seq:
            for f in stage[1]:
                x=f(x)
            layers[stage[0]]=x
            if stage[0]==self.escape:
                break
        return layers

class AttentionMap(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(AttentionMap, self).__init__()
        self.register_buffer('mask',torch.zeros([1,1,24,24]))
        self.mask[0,0,2:-2,2:-2]=1
        self.num_attentions=out_channels
        self.conv_extract = nn.Conv2d(in_channels, in_channels, kernel_size=3,padding=1) #extracting feature map from backbone
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        if self.num_attentions==0:
            return torch.ones([x.shape[0],1,1,1],device=x.device)
        x = self.conv_extract(x)
        x = self.bn1(x)
        x = F.relu(x,inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)+1
        mask=F.interpolate(self.mask,(x.shape[2],x.shape[3]),mode='nearest')
        return x*mask


class AttentionPooling(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, features, attentions,norm=2):
        H, W = features.size()[-2:]
        B, M, AH, AW = attentions.size()
        if AH != H or AW != W:
            attentions=F.interpolate(attentions,size=(H,W), mode='bilinear', align_corners=True)
        if norm==1:
            attentions=attentions+1e-8
        if len(features.shape)==4:
            feature_matrix=torch.einsum('imjk,injk->imn', attentions, features)
        else:
            feature_matrix=torch.einsum('imjk,imnjk->imn', attentions, features)
        if norm==1:
            w=torch.sum(attentions,dim=(2,3)).unsqueeze(-1)
            feature_matrix/=w
        if norm==2:
            feature_matrix = F.normalize(feature_matrix,p=2,dim=-1)
        if norm==3:
            w=torch.sum(attentions,dim=(2,3)).unsqueeze(-1)+1e-8
            feature_matrix/=w
        return feature_matrix


class Texture_Enhance_v2(nn.Module):
    def __init__(self,num_features,num_attentions):
        super().__init__()
        self.output_features=num_features
        self.output_features_d=num_features
        self.conv_extract=nn.Conv2d(num_features,num_features,3,padding=1)
        self.conv0=nn.Conv2d(num_features*num_attentions,num_features*num_attentions,5,padding=2,groups=num_attentions)
        self.conv1=nn.Conv2d(num_features*num_attentions,num_features*num_attentions,3,padding=1,groups=num_attentions)
        self.bn1=nn.BatchNorm2d(num_features*num_attentions)
        self.conv2=nn.Conv2d(num_features*2*num_attentions,num_features*num_attentions,3,padding=1,groups=num_attentions)
        self.bn2=nn.BatchNorm2d(2*num_features*num_attentions)
        self.conv3=nn.Conv2d(num_features*3*num_attentions,num_features*num_attentions,3,padding=1,groups=num_attentions)
        self.bn3=nn.BatchNorm2d(3*num_features*num_attentions)
        self.conv_last=nn.Conv2d(num_features*4*num_attentions,num_features*num_attentions,1,groups=num_attentions)
        self.bn4=nn.BatchNorm2d(4*num_features*num_attentions)
        self.bn_last=nn.BatchNorm2d(num_features*num_attentions)
        
        self.M=num_attentions
    def cat(self,a,b):
        B,C,H,W=a.shape
        c=torch.cat([a.reshape(B,self.M,-1,H,W),b.reshape(B,self.M,-1,H,W)],dim=2).reshape(B,-1,H,W)
        return c

    def forward(self,feature_maps,attention_maps=(1,1)):
        B,N,H,W=feature_maps.shape
        if type(attention_maps)==tuple:
            attention_size=(int(H*attention_maps[0]),int(W*attention_maps[1]))
        else:
            attention_size=(attention_maps.shape[2],attention_maps.shape[3])
        feature_maps=self.conv_extract(feature_maps)
        feature_maps_d=F.adaptive_avg_pool2d(feature_maps,attention_size)
        if feature_maps.size(2)>feature_maps_d.size(2):
            feature_maps=feature_maps-F.interpolate(feature_maps_d,(feature_maps.shape[2],feature_maps.shape[3]),mode='nearest')
        attention_maps=(torch.tanh(F.interpolate(attention_maps.detach(),(H,W),mode='bilinear',align_corners=True))).unsqueeze(2) if type(attention_maps)!=tuple else 1
        feature_maps=feature_maps.unsqueeze(1)
        feature_maps=(feature_maps*attention_maps).reshape(B,-1,H,W)
        feature_maps0=self.conv0(feature_maps)
        feature_maps1=self.conv1(F.relu(self.bn1(feature_maps0),inplace=True))
        feature_maps1_=self.cat(feature_maps0,feature_maps1)
        feature_maps2=self.conv2(F.relu(self.bn2(feature_maps1_),inplace=True))
        feature_maps2_=self.cat(feature_maps1_,feature_maps2)
        feature_maps3=self.conv3(F.relu(self.bn3(feature_maps2_),inplace=True))
        feature_maps3_=self.cat(feature_maps2_,feature_maps3)
        feature_maps=F.relu(self.bn_last(self.conv_last(F.relu(self.bn4(feature_maps3_),inplace=True))),inplace=True)
        feature_maps=feature_maps.reshape(B,-1,N,H,W)
        return feature_maps,feature_maps_d

 
class MultiAtten(nn.Module):
    def __init__(self, net='xception',feature_layer='b3',num_classes=2,dropout_rate=0.5):
        super().__init__()
        self.num_classes = num_classes

        self.net=xception(num_classes,escape=feature_layer)
        
        self.feature_layer=feature_layer
        with torch.no_grad():
            layers = self.net(torch.zeros(1,3,100,100))
        num_features=layers[self.feature_layer].shape[1]
        
        self.pooling=nn.AdaptiveAvgPool2d(1)
        self.texture_enhance=Texture_Enhance_v2(num_features,1)
        self.num_features=self.texture_enhance.output_features
        self.fc=nn.Linear(self.num_features,self.num_classes)
        self.dropout=nn.Dropout(dropout_rate)

    def forward(self, x):
        layers = self.net(x)
        feature_maps = layers[self.feature_layer]
        feature_maps, _=self.texture_enhance(feature_maps,(0.2,0.2))
        x=self.pooling(feature_maps)
        x = x.flatten(start_dim=1)
        x=self.dropout(x)
        x=self.fc(x)
        return x


