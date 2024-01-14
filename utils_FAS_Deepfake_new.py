import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import sklearn
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import pdb

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res






def performances_tpr_fpr_3datasets(map_score_val_filename, map_score_val_filename2, map_score_val_filename3):
    with open(map_score_val_filename, 'r') as file:
        lines = file.readlines()
    with open(map_score_val_filename2, 'r') as file2:
        lines2 = file2.readlines()
    with open(map_score_val_filename3, 'r') as file3:
        lines3 = file3.readlines()
    scores = []
    labels = []
    for line in lines:
        record = line.split()
        scores.append(float(record[0]))
        labels.append(float(record[1]))
    for line in lines2:
        record = line.split()
        scores.append(float(record[0]))
        labels.append(float(record[1]))
    for line in lines3:
        record = line.split()
        scores.append(float(record[0]))
        labels.append(float(record[1]))

    fpr_list = [0.1, 0.01, 0.001]
    threshold_list = get_thresholdtable_from_fpr(scores,labels, fpr_list)
    tpr_list = get_tpr_from_threshold(scores,labels, threshold_list)
    return tpr_list







def performances_tpr_fpr_2datasets(map_score_val_filename, map_score_val_filename2):
    with open(map_score_val_filename, 'r') as file:
        lines = file.readlines()
    with open(map_score_val_filename2, 'r') as file2:
        lines2 = file2.readlines()
    scores = []
    labels = []
    for line in lines:
        record = line.split()
        scores.append(float(record[0]))
        labels.append(float(record[1]))
    for line in lines2:
        record = line.split()
        scores.append(float(record[0]))
        labels.append(float(record[1]))

    fpr_list = [0.1, 0.01, 0.001]
    threshold_list = get_thresholdtable_from_fpr(scores,labels, fpr_list)
    tpr_list = get_tpr_from_threshold(scores,labels, threshold_list)
    return tpr_list



def performances_tpr_fpr(map_score_val_filename):
    with open(map_score_val_filename, 'r') as file:
        lines = file.readlines()
    scores = []
    labels = []
    for line in lines:
        try:
            record = line.split()
            scores.append(float(record[0]))
            labels.append(float(record[1]))
        except:
            continue

    fpr_list = [0.1, 0.01, 0.001]
    threshold_list = get_thresholdtable_from_fpr(scores,labels, fpr_list)
    tpr_list = get_tpr_from_threshold(scores,labels, threshold_list)
    return tpr_list


def get_thresholdtable_from_fpr(scores, labels, fpr_list):
    threshold_list = []
    live_scores = []
    for score, label in zip(scores,labels):
        if label == 1:
            live_scores.append(float(score))
    live_scores.sort()
    live_nums = len(live_scores)
    for fpr in fpr_list:
        i_sample = int(fpr * live_nums)
        i_sample = max(1, i_sample)
        if not live_scores:
            return [0.5]*10
        threshold_list.append(live_scores[i_sample - 1])
    return threshold_list

# Get the threshold under thresholds
def get_tpr_from_threshold(scores,labels, threshold_list):
    tpr_list = []
    hack_scores = []
    for score, label in zip(scores,labels):
        if label == 0:
            hack_scores.append(float(score))
    hack_scores.sort()
    hack_nums = len(hack_scores)
    for threshold in threshold_list:
        hack_index = 0
        while hack_index < hack_nums:
            if hack_scores[hack_index] >= threshold:
                break
            else:
                hack_index += 1
        if hack_nums != 0:
            tpr = hack_index * 1.0 / hack_nums
        else:
            tpr = 0
        tpr_list.append(tpr)
    return tpr_list


def get_threshold(score_file):
    with open(score_file, 'r') as file:
        lines = file.readlines()

    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        angle = float(tokens[0])
        #pdb.set_trace()
        type = int(tokens[1])
        data.append({'map_score': angle, 'label': type})
        if type==1:
            num_real += 1
        else:
            num_fake += 1

    min_error = count    # account ACER (or ACC)
    min_threshold = 0.0
    min_ACC = 0.0
    min_ACER = 0.0
    min_APCER = 0.0
    min_BPCER = 0.0
    
    
    for d in data:
        threshold = d['map_score']
        
        type1 = len([s for s in data if s['map_score'] <= threshold and s['label'] == 1])
        type2 = len([s for s in data if s['map_score'] > threshold and s['label'] == 0])
        
        ACC = 1-(type1 + type2) / count
        APCER = type2 / num_fake
        BPCER = type1 / num_real
        ACER = (APCER + BPCER) / 2.0
        
        if ACER < min_error:
            min_error = ACER
            min_threshold = threshold
            min_ACC = ACC
            min_ACER = ACER
            min_APCER = APCER
            min_BPCER = min_BPCER

    # print(min_error, min_threshold)
    return min_threshold, min_ACC, min_APCER, min_BPCER, min_ACER



def test_threshold_based(threshold, score_file):
    with open(score_file, 'r') as file:
        lines = file.readlines()

    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        angle = float(tokens[0])
        type = int(tokens[1])
        data.append({'map_score': angle, 'label': type})
        if type==1:
            num_real += 1
        else:
            num_fake += 1
    
 
    type1 = len([s for s in data if s['map_score'] <= threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > threshold and s['label'] == 0])
    
    ACC = 1-(type1 + type2) / count
    APCER = type2 / num_fake
    BPCER = type1 / num_real
    ACER = (APCER + BPCER) / 2.0
    
    return ACC, APCER, BPCER, ACER









def get_err_threhold(fpr, tpr, threshold):
    RightIndex=(tpr+(1-fpr)-1); 
    right_index = np.argmax(RightIndex)
    best_th = threshold[right_index]
    err = fpr[right_index]

    differ_tpr_fpr_1=tpr+fpr-1.0
  
    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    err = fpr[right_index]    

    #print(err, best_th)
    return err, best_th



#def performances(dev_scores, dev_labels, test_scores, test_labels):
def performances_FAS_Separate(SiW_test_filename, test_3DMAD_filename, HKBU_test_filename, MSU_test_filename, test_3DMask_filename, test_ROSE_filename):
    
    # SiW
    with open(SiW_test_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1
    
    fpr,tpr,threshold = roc_curve(val_labels, val_scores, pos_label=1)
    EER_SiW, val_threshold = get_err_threhold(fpr, tpr, threshold)
    AUC_SiW = auc(fpr, tpr)
    
    
    
    # 3DMAD
    with open(test_3DMAD_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1
    
    fpr,tpr,threshold = roc_curve(val_labels, val_scores, pos_label=1)
    EER_3DMAD, val_threshold = get_err_threhold(fpr, tpr, threshold)
    AUC_3DMAD = auc(fpr, tpr)
    
    
    # HKBU
    with open(HKBU_test_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1
    
    fpr,tpr,threshold = roc_curve(val_labels, val_scores, pos_label=1)
    EER_HKBU, val_threshold = get_err_threhold(fpr, tpr, threshold)
    AUC_HKBU = auc(fpr, tpr)
    
    
    # MSU
    with open(MSU_test_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1
    
    fpr,tpr,threshold = roc_curve(val_labels, val_scores, pos_label=1)
    EER_MSU, val_threshold = get_err_threhold(fpr, tpr, threshold)
    AUC_MSU = auc(fpr, tpr)
    
    
    # 3DMask
    with open(test_3DMask_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1
    
    fpr,tpr,threshold = roc_curve(val_labels, val_scores, pos_label=1)
    EER_3DMask, val_threshold = get_err_threhold(fpr, tpr, threshold)
    AUC_3DMask = auc(fpr, tpr)
    
    # ROSE
    with open(test_ROSE_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1
    
    fpr,tpr,threshold = roc_curve(val_labels, val_scores, pos_label=1)
    EER_ROSE, val_threshold = get_err_threhold(fpr, tpr, threshold)
    AUC_ROSE = auc(fpr, tpr)
    
    # tpr @ fpr = 0.1, 0.01, 0.001
    tpr_fpr_intra = performances_tpr_fpr_3datasets(SiW_test_filename, test_3DMAD_filename, HKBU_test_filename)
    tpr_fpr_cross = performances_tpr_fpr_3datasets(MSU_test_filename, test_3DMask_filename, test_ROSE_filename)
    
    
    
    #return val_threshold, best_test_threshold, val_ACC, val_ACER, test_ACC, test_APCER, test_BPCER, test_ACER, test_threshold_ACER
    return AUC_SiW, EER_SiW, AUC_3DMAD, EER_3DMAD, AUC_HKBU, EER_HKBU, AUC_MSU, EER_MSU, AUC_3DMask, EER_3DMask, AUC_ROSE, EER_ROSE, tpr_fpr_intra, tpr_fpr_cross









#def performances(dev_scores, dev_labels, test_scores, test_labels):
def performances_Deepfake_Separate(FF_test_filename, DFDC_test_filename, test_CelebDF_filename):
    
    # FF
    with open(FF_test_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1
    
    fpr,tpr,threshold = roc_curve(val_labels, val_scores, pos_label=1)
    EER_FF, val_threshold = get_err_threhold(fpr, tpr, threshold)

    #val_ACC = 1-(type1 + type2) / count
    AUC_FF = auc(fpr, tpr)

    
    
    # DFDC
    with open(DFDC_test_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1
    
    fpr,tpr,threshold = roc_curve(val_labels, val_scores, pos_label=1)
    EER_DFDC, val_threshold = get_err_threhold(fpr, tpr, threshold)

    #val_ACC = 1-(type1 + type2) / count
    AUC_DFDC = auc(fpr, tpr)
    
    
    
    # CelebDF
    with open(test_CelebDF_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1
    
    fpr,tpr,threshold = roc_curve(val_labels, val_scores, pos_label=1)
    EER_CelebDF, val_threshold = get_err_threhold(fpr, tpr, threshold)

    #val_ACC = 1-(type1 + type2) / count
    AUC_CelebDF = auc(fpr, tpr)
    
    # tpr @ fpr = 0.1, 0.01, 0.001
    tpr_fpr_intra = performances_tpr_fpr(FF_test_filename)
    tpr_fpr_cross = performances_tpr_fpr_2datasets(DFDC_test_filename, test_CelebDF_filename)
    
    #return val_threshold, best_test_threshold, val_ACC, val_ACER, test_ACC, test_APCER, test_BPCER, test_ACER, test_threshold_ACER
    return AUC_FF, EER_FF, AUC_DFDC, EER_DFDC, AUC_CelebDF, EER_CelebDF, tpr_fpr_intra, tpr_fpr_cross








def performances_SiW_EER(map_score_val_filename):

    # val 
    with open(map_score_val_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1
    
    fpr,tpr,threshold = roc_curve(val_labels, val_scores, pos_label=1)
    val_err, val_threshold = get_err_threhold(fpr, tpr, threshold)
    
    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])
    
    val_ACC = 1-(type1 + type2) / count
    val_APCER = type2 / num_fake
    val_BPCER = type1 / num_real
    val_ACER = (val_APCER + val_BPCER) / 2.0
    
    
    
    return val_threshold, val_ACC, val_APCER, val_BPCER, val_ACER







def performances_SiWM_EER(map_score_val_filename):

    # val 
    with open(map_score_val_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1
    
    fpr,tpr,threshold = roc_curve(val_labels, val_scores, pos_label=1)
    val_err, val_threshold = get_err_threhold(fpr, tpr, threshold)
    
    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])
    
    val_ACC = 1-(type1 + type2) / count
    val_APCER = type2 / num_fake
    val_BPCER = type1 / num_real
    val_ACER = (val_APCER + val_BPCER) / 2.0
    
    
    
    return val_threshold, val_err, val_ACC, val_APCER, val_BPCER, val_ACER




def get_err_threhold_CASIA_Replay(fpr, tpr, threshold):
    RightIndex=(tpr+(1-fpr)-1); 
    right_index = np.argmax(RightIndex)
    best_th = threshold[right_index]
    err = fpr[right_index]

    differ_tpr_fpr_1=tpr+fpr-1.0
  
    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    err = fpr[right_index]    

    #print(err, best_th)
    return err, best_th, right_index



def performances_CASIA_Replay(map_score_val_filename):

    # val 
    with open(map_score_val_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])  # int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1
    
    fpr,tpr,threshold = roc_curve(val_labels, val_scores, pos_label=1)
    val_err, val_threshold, right_index = get_err_threhold_CASIA_Replay(fpr, tpr, threshold)
    
    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])
    
    val_ACC = 1-(type1 + type2) / count
    
    FRR = 1- tpr    # FRR = 1 - TPR
    
    HTER = (fpr+FRR)/2.0    # error recognition rate &  reject recognition rate
    
    return val_ACC, fpr[right_index], FRR[right_index], HTER[right_index]





def performances_ZeroShot(map_score_val_filename):

    # val 
    with open(map_score_val_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1
    
    fpr,tpr,threshold = roc_curve(val_labels, val_scores, pos_label=1)
    auc_val = metrics.auc(fpr, tpr)
    
    val_err, val_threshold, right_index = get_err_threhold_CASIA_Replay(fpr, tpr, threshold)
    
    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])
    
    val_ACC = 1-(type1 + type2) / count
    
    FRR = 1- tpr    # FRR = 1 - TPR
    
    HTER = (fpr+FRR)/2.0    # error recognition rate &  reject recognition rate
    
    return val_ACC, auc_val, HTER[right_index]






def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

