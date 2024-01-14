from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
import argparse,os
import pandas as pd
import cv2
import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

#from models.ResNet_Xception import Xception


from Load_FAS_ResNet_Xception import Spoofing_train, Normaliztion, ToTensor, RandomHorizontalFlip, Cutout
from Load_FAS_ResNet_Xception_test import Spoofing_valtest, Normaliztion_valtest, ToTensor_valtest


import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import copy
import pdb

from utils_FAS_Deepfake_new import AvgrageMeter, accuracy, performances_FAS_Separate, performances_Deepfake_Separate




##########    Dataset root    ##########

# root_dir    SiW + 3DMAD + HKBU ;   MSU + 3DMask
root_FAS_Deepfake_dir = '/scratch/project_2004030/'

# train_list     SiW + 3DMAD + HKBU
train_FAS_list = '/users/yuzitong/Protocols/FAS_Deepfake_TIFS_Protocols/Joint_FAS_Deepfake/SiW_3DMAD_HKBU_train.txt'
train_Deepfake_list = '/users/yuzitong/Protocols/FAS_Deepfake_TIFS_Protocols/Joint_FAS_Deepfake/FF_STmap_CWT_train.txt'

# Intra-test_list      SiW + 3DMAD + HKBU
test_SiW_list = '/users/yuzitong/Protocols/FAS_Deepfake_TIFS_Protocols/Joint_FAS_Deepfake/SiW_test.txt'
test_3DMAD_list = '/users/yuzitong/Protocols/FAS_Deepfake_TIFS_Protocols/Joint_FAS_Deepfake/3DMAD_test.txt' 
test_HKBU_list = '/users/yuzitong/Protocols/FAS_Deepfake_TIFS_Protocols/Joint_FAS_Deepfake/HKBU_test.txt' 

# Cross-test      MSU + 3DMask 
test_MSU_list = '/users/yuzitong/Protocols/FAS_Deepfake_TIFS_Protocols/Joint_FAS_Deepfake/MSU_test.txt' 
test_3DMask_list = '/users/yuzitong/Protocols/FAS_Deepfake_TIFS_Protocols/Joint_FAS_Deepfake/3DMask_test.txt' 
test_ROSE_list = '/users/yuzitong/Protocols/FAS_Deepfake_TIFS_Protocols/Joint_FAS_Deepfake/ROSE_train_test_all.txt'

# Intra-test_list      FF++
test_FF_list = '/users/yuzitong/Protocols/FAS_Deepfake_TIFS_Protocols/Joint_FAS_Deepfake/FF_STmap_CWT_test.txt'

# Cross-test      DFDC + Celeb-DF 
test_DFDC_list = '/users/yuzitong/Protocols/FAS_Deepfake_TIFS_Protocols/Joint_FAS_Deepfake/DFDC_test.txt' 
test_CelebDF_list = '/users/yuzitong/Protocols/FAS_Deepfake_TIFS_Protocols/Joint_FAS_Deepfake/Celeb-DF_test.txt' 



# feature  -->   [ batch, channel, height, width ]
def FeatureMap2Heatmap(x):
    ## initial images 
    ## initial images 
    org_img = x[0,:,:,:].cpu()  
    org_img = org_img.data.numpy()*128+127.5
    org_img = org_img.transpose((1, 2, 0))
    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(args.log+'/'+args.log + '_x_visual.jpg', org_img)
    

def get_inf_iterator(data_loader):
    """Inf data iterator."""
    while True:
        for i, sample_batched in enumerate(data_loader):
            yield (sample_batched['image_x'], sample_batched['spoofing_label'])



# main function
def train_test():
    # GPU  & log file  -->   if use DataParallel, please comment this command
    #os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (args.gpu)

    isExists = os.path.exists(args.log)
    if not isExists:
        os.makedirs(args.log)
    log_file = open(args.log+'/'+ args.log+'_log.txt', 'w')
    
    echo_batches = args.echo_batches

    print("Separate FAS!!!:\n ")

    log_file.write('Separate FAS!!!:\n ')
    log_file.flush()

    # load the network, load the pre-trained model in UCF101?
    finetune = args.finetune
    if finetune==True:
        print('finetune!\n')
        log_file.write('finetune!\n')
        log_file.flush()
            
        model = DepthNet_FAS()
        #model = model.cuda()
        model = model.to(device[0])
        model = nn.DataParallel(model, device_ids=device, output_device=device[0])
        model.load_state_dict(torch.load('Depth_FAS_absolute_contrastive_P1_ft/Depth_FAS_absolute_contrastive_P1_ft_100.pkl'))

        lr = args.lr
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        

    else:
        print('train from scratch!\n')
        log_file.write('train from scratch!\n')
        log_file.flush()
        
        
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        #model = Xception(pretrained=True)
        #model = ResNet18_FineGrained_PyramidSupervision(pretrained=True, num_class=3)
        
        

        model = model.cuda()
        #model = model.to(device[0])
        #model = nn.DataParallel(model, device_ids=device, output_device=device[0])

        lr = args.lr
        #optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.00005)
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    print(model) 
    
    
    #criterion_absolute_loss = nn.MSELoss().cuda()
    #criterion_contrastive_loss = Contrast_depth_loss().cuda() 
    
    criterion = nn.CrossEntropyLoss()

    #bandpass_filter_numpy = build_bandpass_filter_numpy(30, 30)  # fs, order  # 61, 64 

    ACER_save = 1.0
    
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        scheduler.step()
        if (epoch + 1) % args.step_size == 0:
            lr *= args.gamma

        
        loss_absolute = AvgrageMeter()
        loss_contra =  AvgrageMeter()
        loss_absolute_RGB = AvgrageMeter()
        
        
        
        ###########################################
        '''                train             '''
        ###########################################
        model.train()
        
        train_data_FAS = Spoofing_train(train_FAS_list, root_FAS_Deepfake_dir, transform=transforms.Compose([RandomHorizontalFlip(), ToTensor(), Cutout(), Normaliztion()]))
        dataloader_train_FAS = DataLoader(train_data_FAS, batch_size=args.batchsize//2, shuffle=True, num_workers=4)
        
        train_data_DF = Spoofing_train(train_Deepfake_list, root_FAS_Deepfake_dir, transform=transforms.Compose([RandomHorizontalFlip(), ToTensor(), Cutout(), Normaliztion()]))
        dataloader_train_DF = DataLoader(train_data_DF, batch_size=args.batchsize//2, shuffle=True, num_workers=4)
        
        iternum = min(len(dataloader_train_FAS),len(dataloader_train_DF))
        
        
        data1_FAS = get_inf_iterator(dataloader_train_FAS)
        data1_DF = get_inf_iterator(dataloader_train_DF)
        
        for i in range(iternum):
            image_x_FAS, spoofing_label_FAS = next(data1_FAS)
            image_x_DF, spoofing_label_DF = next(data1_DF)
             
            inputs = torch.cat([image_x_FAS, image_x_DF],0).cuda()
            spoof_label = torch.cat([spoofing_label_FAS, spoofing_label_DF],0).cuda()

            optimizer.zero_grad()
            

            logits =  model(inputs)
            
            loss_global =  criterion(logits, spoof_label.squeeze(-1))
 
             
            loss =  loss_global
             
            loss.backward()
            
            optimizer.step()
            
            n = inputs.size(0)
            loss_absolute.update(loss_global.data, n)
            loss_contra.update(loss_global.data, n)
            loss_absolute_RGB.update(loss_global.data, n)
        

            if i % echo_batches == echo_batches-1:    # print every 50 mini-batches
                
                # visualization
                FeatureMap2Heatmap(inputs)

                # log written
                print('epoch:%d, mini-batch:%3d, lr=%f, CE_global= %.4f , CE1= %.4f , CE2= %.4f \n' % (epoch + 1, i + 1, lr,  loss_absolute.avg,  loss_contra.avg,  loss_absolute_RGB.avg))
        
        # whole epoch average
        log_file.write('epoch:%d, mini-batch:%3d, lr=%f, CE_global= %.4f , CE1= %.4f , CE2= %.4f \n' % (epoch + 1, i + 1, lr,  loss_absolute.avg,  loss_contra.avg,  loss_absolute_RGB.avg))
        log_file.flush()
    
                    
        #### validation/test
        #if epoch <260:
        #    epoch_test = 260    # DPC =  200
        #else:
        #    epoch_test = 20    # 20
        
        epoch_test = 2
        if epoch % epoch_test == epoch_test-1:    # test every 5 epochs  
            model.eval()
            
            with torch.no_grad():
                
                ###########################################
                '''                test             '''
                ##########################################
                # Intra-test for SiW
                test_data = Spoofing_valtest(test_SiW_list, root_FAS_Deepfake_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    #log_file.write('test SiW i= %d \n' % (i))
                    
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    #test_maps = sample_batched['val_map_x'].cuda()   # binary map from PRNet 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    
                    for frame_t in range(inputs.shape[1]):
                        logits  =  model(inputs[:,frame_t,:,:,:])
                        if frame_t==0:
                            logits_accumulate = F.softmax(logits,-1)
                        else:
                            logits_accumulate += F.softmax(logits,-1)
                    logits_accumulate = logits_accumulate/inputs.shape[1]
                    
                    for test_batch in range(inputs.shape[0]):
                        map_score_list.append('{} {}\n'.format(logits_accumulate[test_batch][1], spoof_label[test_batch][0]))
                    
                    #for test_batch in range(inputs.shape[0]):
                    #    map_score = 0.0
                    #    for frame_t in range(inputs.shape[1]):
                    #        logits  =  model(inputs[test_batch,frame_t,:,:,:].unsqueeze(0))
                    #        map_score += F.softmax(logits)[0][1]
                    #    map_score = map_score/inputs.shape[1]
                    #        
                    #    map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                
                #SiW_test_filename = args.log+'/'+ args.log+'_SiW_test_%d.txt' % (epoch + 1)
                SiW_test_filename = args.log+'/'+ args.log+'_SiW_test.txt' 
                with open(SiW_test_filename, 'w') as file:
                    file.writelines(map_score_list)    
                
                ##########################################
                # Intra-test for 3DMAD
                test_data = Spoofing_valtest(test_3DMAD_list, root_FAS_Deepfake_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    #test_maps = sample_batched['val_map_x'].cuda()   # binary map from PRNet 
        
                    optimizer.zero_grad()
                    
                    for frame_t in range(inputs.shape[1]):
                        logits  =  model(inputs[:,frame_t,:,:,:])
                        if frame_t==0:
                            logits_accumulate = F.softmax(logits,-1)
                        else:
                            logits_accumulate += F.softmax(logits,-1)
                    logits_accumulate = logits_accumulate/inputs.shape[1]
                    
                    for test_batch in range(inputs.shape[0]):
                        map_score_list.append('{} {}\n'.format(logits_accumulate[test_batch][1], spoof_label[test_batch][0]))
                    
                    #for test_batch in range(inputs.shape[0]):
                    #    map_score = 0.0
                    #    for frame_t in range(inputs.shape[1]):
                    #        logits  =  model(inputs[test_batch,frame_t,:,:,:].unsqueeze(0))
                    #        map_score += F.softmax(logits)[0][1]
                    #    map_score = map_score/inputs.shape[1]
                    #        
                    #    map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                
                #test_3DMAD_filename = args.log+'/'+ args.log+'_3DMAD_test_%d.txt' % (epoch + 1)
                test_3DMAD_filename = args.log+'/'+ args.log+'_3DMAD_test.txt' 
                with open(test_3DMAD_filename, 'w') as file:
                    file.writelines(map_score_list)    
                    
                
                ##########################################    
                # Intra-test for HKBU
                test_data = Spoofing_valtest(test_HKBU_list, root_FAS_Deepfake_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    #test_maps = sample_batched['val_map_x'].cuda()   # binary map from PRNet 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    for frame_t in range(inputs.shape[1]):
                        logits  =  model(inputs[:,frame_t,:,:,:])
                        if frame_t==0:
                            logits_accumulate = F.softmax(logits,-1)
                        else:
                            logits_accumulate += F.softmax(logits,-1)
                    logits_accumulate = logits_accumulate/inputs.shape[1]
                    
                    for test_batch in range(inputs.shape[0]):
                        map_score_list.append('{} {}\n'.format(logits_accumulate[test_batch][1], spoof_label[test_batch][0]))
                    
                    #for test_batch in range(inputs.shape[0]):
                    #    map_score = 0.0
                    #    for frame_t in range(inputs.shape[1]):
                    #        logits  =  model(inputs[test_batch,frame_t,:,:,:].unsqueeze(0))
                    #        map_score += F.softmax(logits)[0][1]
                    #    map_score = map_score/inputs.shape[1]
                    #        
                    #    map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                
                #HKBU_test_filename = args.log+'/'+ args.log+'_HKBU_test_%d.txt' % (epoch + 1)
                HKBU_test_filename = args.log+'/'+ args.log+'_HKBU_test.txt' 
                with open(HKBU_test_filename, 'w') as file:
                    file.writelines(map_score_list)    
                    
                
                ##########################################    
                # Inter-test for MSU
                test_data = Spoofing_valtest(test_MSU_list, root_FAS_Deepfake_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    #test_maps = sample_batched['val_map_x'].cuda()   # binary map from PRNet 
        
                    optimizer.zero_grad()
                    
                    for frame_t in range(inputs.shape[1]):
                        logits  =  model(inputs[:,frame_t,:,:,:])
                        if frame_t==0:
                            logits_accumulate = F.softmax(logits,-1)
                        else:
                            logits_accumulate += F.softmax(logits,-1)
                    logits_accumulate = logits_accumulate/inputs.shape[1]
                    
                    for test_batch in range(inputs.shape[0]):
                        map_score_list.append('{} {}\n'.format(logits_accumulate[test_batch][1], spoof_label[test_batch][0]))
                    
                    #for test_batch in range(inputs.shape[0]):
                    #    map_score = 0.0
                    #    for frame_t in range(inputs.shape[1]):
                    #        logits  =  model(inputs[test_batch,frame_t,:,:,:].unsqueeze(0))
                    #        map_score += F.softmax(logits)[0][1]
                    #    map_score = map_score/inputs.shape[1]
                    #        
                    #    map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                
                #MSU_test_filename = args.log+'/'+ args.log+'_MSU_test_%d.txt' % (epoch + 1)
                MSU_test_filename = args.log+'/'+ args.log+'_MSU_test.txt' 
                with open(MSU_test_filename, 'w') as file:
                    file.writelines(map_score_list)    
                    
                
                ##########################################    
                # Inter-test for 3DMask
                test_data = Spoofing_valtest(test_3DMask_list, root_FAS_Deepfake_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    #test_maps = sample_batched['val_map_x'].cuda()   # binary map from PRNet 
        
                    optimizer.zero_grad()
                    
                    for frame_t in range(inputs.shape[1]):
                        logits  =  model(inputs[:,frame_t,:,:,:])
                        if frame_t==0:
                            logits_accumulate = F.softmax(logits,-1)
                        else:
                            logits_accumulate += F.softmax(logits,-1)
                    logits_accumulate = logits_accumulate/inputs.shape[1]
                    
                    for test_batch in range(inputs.shape[0]):
                        map_score_list.append('{} {}\n'.format(logits_accumulate[test_batch][1], spoof_label[test_batch][0]))
                    
                    #for test_batch in range(inputs.shape[0]):
                    #    map_score = 0.0
                    #    for frame_t in range(inputs.shape[1]):
                    #        logits  =  model(inputs[test_batch,frame_t,:,:,:].unsqueeze(0))
                    #        map_score += F.softmax(logits)[0][1]
                    #    map_score = map_score/inputs.shape[1]
                    #        
                    #    map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                
                #test_3DMask_filename = args.log+'/'+ args.log+'_3DMask_test_%d.txt' % (epoch + 1)
                test_3DMask_filename = args.log+'/'+ args.log+'_3DMask_test.txt'
                with open(test_3DMask_filename, 'w') as file:
                    file.writelines(map_score_list)    
                    
                
                
                ##########################################    
                # Inter-test for ROSE
                test_data = Spoofing_valtest(test_ROSE_list, root_FAS_Deepfake_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    #test_maps = sample_batched['val_map_x'].cuda()   # binary map from PRNet 
        
                    optimizer.zero_grad()
                    
                    for frame_t in range(inputs.shape[1]):
                        logits  =  model(inputs[:,frame_t,:,:,:])
                        if frame_t==0:
                            logits_accumulate = F.softmax(logits,-1)
                        else:
                            logits_accumulate += F.softmax(logits,-1)
                    logits_accumulate = logits_accumulate/inputs.shape[1]
                    
                    for test_batch in range(inputs.shape[0]):
                        map_score_list.append('{} {}\n'.format(logits_accumulate[test_batch][1], spoof_label[test_batch][0]))
                    
                    #for test_batch in range(inputs.shape[0]):
                    #    map_score = 0.0
                    #    for frame_t in range(inputs.shape[1]):
                    #        logits  =  model(inputs[test_batch,frame_t,:,:,:].unsqueeze(0))
                    #        map_score += F.softmax(logits)[0][1]
                    #    map_score = map_score/inputs.shape[1]
                    #        
                    #    map_score_list.append('{} {}\n'.format(map_score, spoof_label[test_batch][0]))
                
                #test_3DMask_filename = args.log+'/'+ args.log+'_3DMask_test_%d.txt' % (epoch + 1)
                test_ROSE_filename = args.log+'/'+ args.log+'_ROSE_test.txt'
                with open(test_ROSE_filename, 'w') as file:
                    file.writelines(map_score_list)   
                
                # Intra-test for FF++
                test_data = Spoofing_valtest(test_FF_list, root_FAS_Deepfake_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    #log_file.write('test SiW i= %d \n' % (i))
                    
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    #test_maps = sample_batched['val_map_x'].cuda()   # binary map from PRNet 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    for frame_t in range(inputs.shape[1]):
                        logits  =  model(inputs[:,frame_t,:,:,:])
                        if frame_t==0:
                            logits_accumulate = F.softmax(logits,-1)
                        else:
                            logits_accumulate += F.softmax(logits,-1)
                    logits_accumulate = logits_accumulate/inputs.shape[1]
                    
                    for test_batch in range(inputs.shape[0]):
                        map_score_list.append('{} {}\n'.format(logits_accumulate[test_batch][1], spoof_label[test_batch][0]))
                
                FF_test_filename = args.log+'/'+ args.log+'_FF_test_%d.txt' % (epoch + 1)
                with open(FF_test_filename, 'w') as file:
                    file.writelines(map_score_list)    
                
                
                
                ##########################################    
                # Inter-test for DFDC
                test_data = Spoofing_valtest(test_DFDC_list, root_FAS_Deepfake_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    #test_maps = sample_batched['val_map_x'].cuda()   # binary map from PRNet 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    for frame_t in range(inputs.shape[1]):
                        logits  =  model(inputs[:,frame_t,:,:,:])
                        if frame_t==0:
                            logits_accumulate = F.softmax(logits,-1)
                        else:
                            logits_accumulate += F.softmax(logits,-1)
                    logits_accumulate = logits_accumulate/inputs.shape[1]
                    
                    for test_batch in range(inputs.shape[0]):
                        map_score_list.append('{} {}\n'.format(logits_accumulate[test_batch][1], spoof_label[test_batch][0]))
                
                DFDC_test_filename = args.log+'/'+ args.log+'_DFDC_test_%d.txt' % (epoch + 1)
                with open(DFDC_test_filename, 'w') as file:
                    file.writelines(map_score_list)    
                    
                
                ##########################################    
                # Inter-test for CelebDF
                test_data = Spoofing_valtest(test_CelebDF_list, root_FAS_Deepfake_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    #test_maps = sample_batched['val_map_x'].cuda()   # binary map from PRNet 
        
                    optimizer.zero_grad()
                    
                    #pdb.set_trace()
                    for frame_t in range(inputs.shape[1]):
                        logits  =  model(inputs[:,frame_t,:,:,:])
                        if frame_t==0:
                            logits_accumulate = F.softmax(logits,-1)
                        else:
                            logits_accumulate += F.softmax(logits,-1)
                    logits_accumulate = logits_accumulate/inputs.shape[1]
                    
                    for test_batch in range(inputs.shape[0]):
                        map_score_list.append('{} {}\n'.format(logits_accumulate[test_batch][1], spoof_label[test_batch][0]))
                
                test_CelebDF_filename = args.log+'/'+ args.log+'_CelebDF_test_%d.txt' % (epoch + 1)
                with open(test_CelebDF_filename, 'w') as file:
                    file.writelines(map_score_list)    
                
 
                
                ##########################################################################   
                #       performance measurement for both intra- and inter-testings
                ##########################################################################   
                AUC_SiW, EER_SiW, AUC_3DMAD, EER_3DMAD, AUC_HKBU, EER_HKBU, AUC_MSU, EER_MSU, AUC_3DMask, EER_3DMask, AUC_ROSE, EER_ROSE, tpr_fpr_intra, tpr_fpr_cross = performances_FAS_Separate(SiW_test_filename, test_3DMAD_filename, HKBU_test_filename, MSU_test_filename, test_3DMask_filename, test_ROSE_filename)
                
                print('epoch:%d, AUC_SiW= %.4f, EER_SiW= %.4f' % (epoch + 1, AUC_SiW, EER_SiW))
                log_file.write('\n epoch:%d, AUC_SiW= %.4f, EER_SiW= %.4f\n' % (epoch + 1, AUC_SiW, EER_SiW))
              
                print('epoch:%d, AUC_3DMAD= %.4f, EER_3DMAD= %.4f' % (epoch + 1, AUC_3DMAD, EER_3DMAD))
                log_file.write('epoch:%d, AUC_3DMAD= %.4f, EER_3DMAD= %.4f\n' % (epoch + 1, AUC_3DMAD, EER_3DMAD))
                
                print('epoch:%d, AUC_HKBU= %.4f, EER_HKBU= %.4f' % (epoch + 1, AUC_HKBU, EER_HKBU))
                log_file.write('epoch:%d, AUC_HKBU= %.4f, EER_HKBU= %.4f\n' % (epoch + 1, AUC_HKBU, EER_HKBU))
                
                print('epoch:%d, AUC_MSU= %.4f, EER_MSU= %.4f' % (epoch + 1, AUC_MSU, EER_MSU))
                log_file.write('epoch:%d, AUC_MSU= %.4f, EER_MSU= %.4f\n' % (epoch + 1, AUC_MSU, EER_MSU))
                
                print('epoch:%d, AUC_3DMask= %.4f, EER_3DMask= %.4f' % (epoch + 1, AUC_3DMask, EER_3DMask))
                log_file.write('epoch:%d, AUC_3DMask= %.4f, EER_3DMask= %.4f\n\n' % (epoch + 1, AUC_3DMask, EER_3DMask))
                
                print('epoch:%d, AUC_ROSE= %.4f, EER_ROSE= %.4f' % (epoch + 1, AUC_ROSE, EER_ROSE))
                log_file.write('epoch:%d, AUC_ROSE= %.4f, EER_ROSE= %.4f\n\n' % (epoch + 1, AUC_ROSE, EER_ROSE))
                
                print('epoch:%d, Intra-TPR = %.4f, %.4f, %.4f' % (epoch + 1, tpr_fpr_intra[0], tpr_fpr_intra[1], tpr_fpr_intra[2]))
                log_file.write('epoch:%d, Intra-TPR = %.4f, %.4f, %.4f\n\n' % (epoch + 1, tpr_fpr_intra[0], tpr_fpr_intra[1], tpr_fpr_intra[2]))
                
                print('epoch:%d, Cross-TPR = %.4f, %.4f, %.4f' % (epoch + 1, tpr_fpr_cross[0], tpr_fpr_cross[1], tpr_fpr_cross[2]))
                log_file.write('epoch:%d, Cross-TPR = %.4f, %.4f, %.4f\n\n' % (epoch + 1, tpr_fpr_cross[0], tpr_fpr_cross[1], tpr_fpr_cross[2]))
                
                
                # Deepfake
                AUC_FF, EER_FF, AUC_DFDC, EER_DFDC, AUC_CelebDF, EER_CelebDF, tpr_fpr_intra, tpr_fpr_cross = performances_Deepfake_Separate(FF_test_filename, DFDC_test_filename, test_CelebDF_filename)
                
                print('epoch:%d, AUC_FF= %.4f, EER_FF= %.4f' % (epoch + 1, AUC_FF, EER_FF))
                log_file.write('\n epoch:%d, AUC_FF= %.4f, EER_FF= %.4f\n' % (epoch + 1, AUC_FF, EER_FF))
                
                print('epoch:%d, AUC_DFDC= %.4f, EER_DFDC= %.4f' % (epoch + 1, AUC_DFDC, EER_DFDC))
                log_file.write('epoch:%d, AUC_DFDC= %.4f, EER_DFDC= %.4f\n' % (epoch + 1, AUC_DFDC, EER_DFDC))
                
                print('epoch:%d, AUC_CelebDF= %.4f, EER_CelebDF= %.4f' % (epoch + 1, AUC_CelebDF, EER_CelebDF))
                log_file.write('epoch:%d, AUC_CelebDF= %.4f, EER_CelebDF= %.4f\n\n' % (epoch + 1, AUC_CelebDF, EER_CelebDF))
                
                print('epoch:%d, Intra-TPR = %.4f, %.4f, %.4f' % (epoch + 1, tpr_fpr_intra[0], tpr_fpr_intra[1], tpr_fpr_intra[2]))
                log_file.write('epoch:%d, Intra-TPR = %.4f, %.4f, %.4f\n\n' % (epoch + 1, tpr_fpr_intra[0], tpr_fpr_intra[1], tpr_fpr_intra[2]))
                
                print('epoch:%d, Cross-TPR = %.4f, %.4f, %.4f' % (epoch + 1, tpr_fpr_cross[0], tpr_fpr_cross[1], tpr_fpr_cross[2]))
                log_file.write('epoch:%d, Cross-TPR = %.4f, %.4f, %.4f\n\n' % (epoch + 1, tpr_fpr_cross[0], tpr_fpr_cross[1], tpr_fpr_cross[2]))
                
                log_file.flush()
                
        #if epoch <1:    
        # save the model until the next improvement     
        #    torch.save(model.state_dict(), args.log+'/'+args.log+'_%d.pkl' % (epoch + 1))


    print('Finished Training')
    log_file.close()
  

  
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=int, default=3, help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')  #default=0.0001   0.01
    parser.add_argument('--batchsize', type=int, default=64, help='initial batchsize')  #default=7  | sgd = 32
    parser.add_argument('--step_size', type=int, default=20, help='how many epochs lr decays once')  # 500  | DPC = 400
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=10, help='how many batches display once')  # 50
    parser.add_argument('--epochs', type=int, default=30, help='total training epochs')
    parser.add_argument('--log', type=str, default="ViT_FAS_Deepfake_Joint_Simult", help='log and save model name')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')

    args = parser.parse_args()
    train_test()
