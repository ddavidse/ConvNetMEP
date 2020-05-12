# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:31:45 2020
@author: ddavidse

code for elephant set
"""


# %% Importing libraries

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

import time
import os
import random
import copy

import loss_landscapes
import loss_landscapes.metrics

from mpl_toolkits.mplot3d import axes3d, Axes3D # ignore warning, we need this 
from sklearn.decomposition import PCA

from CNN_functions import files_tensor, getListOfFiles, batch_generator, string_tensor, \
        sec_max, weights_init, plot_confusion_matrix, weights_to_list, list_to_weights
        
import pandas as pd
import seaborn as sb

from scipy.special import softmax

from evt_fitting_mod import weibull_tailfitting
from compute_openmax_mod import recalibrate_scores, compute_open_max_probability

# %%  ----- CUDA initialization ------------------------------------------------------------------------------------------

#torch.set_default_tensor_type('torch.cuda.FloatTensor')

SeedVal = 523594
random.seed(SeedVal)

# %% ----- Select dir name etc -------------------------------------------------------------------------------------------

#dirName = r'D:\ddavidse\Desktop\test folder\data2'

DT = r'C:\Users\ddavidse\Desktop'
output_dir = r'D:\ddavidse\Desktop\Network_runs'

dataset_orig_dir = r'C:\SharedData\Datasets for CNN\100x100 downsampled from 150x150'
#dataset_fool_dir = r'C:\SharedData\Datasets for CNN\elephant set'
#dataset_fool_dir = r'C:\SharedData\Datasets for CNN\mirrored set'
dataset_fool_dir = r'C:\SharedData\Datasets for CNN\NoiseMultiplier = 1.5'

saved_run_dir = r'C:\Users\ddavidse\Desktop\Network_runs\0 - noise MAV - 51.9'
saved_weights = r'{}\final_weights.pth'.format(saved_run_dir)

#os.chdir(dirName)
#fileNames = os.listdir()

# %% ----- Select features ------------------------------------------------------------------------------------------------

input_size = 100
BS = 50

fig_dpi = 150       # dpi for saved figures
fig_font_size = 12  # font size for figures

batchnorm_flag = True

baseline_flag = True
threshold_flag = True
threshold = 2.1

MAV_flag = False
thresh_flag = True
thresh = 4.8

openmax_flag = False



# %% make folder

os.chdir(DT)
mfname = 'Network_runs'

if not os.path.exists(mfname):
        os.mkdir(mfname)
        
fdir = '{}\\{}'.format(DT,mfname)

counter = 0
tv = 0

while tv == 0:
    
    counter += 1
    fname = '{}\\Network_pass_{}'.format(fdir, counter)  

    if not os.path.exists(fname):
        os.mkdir(fname)
        tv = 1
        

# %% ----- detecting number of images per class --------------------------------------------------------------------------

os.chdir(dataset_orig_dir)

DirListTop = os.listdir()
names = DirListTop

N = []

for x in names:
    
    os.chdir(x)
    DirListSub = os.listdir()
    N.append(len(DirListSub))
    os.chdir(dataset_orig_dir)
    
NC = np.cumsum(N)
    

# %% ----- detecting file size -------------------------------------------------------------------------------------------

os.chdir(names[0])
filelist = os.listdir()
testfile = filelist[0]

testimage = sio.loadmat(testfile)
testimage2 = list(testimage.values())
testimage3 = testimage2[3]

input_size = len(testimage3)

# %% ----- Obtaining and storing data ------------------------------------------------------------------------------------

# Data
listOfFiles = getListOfFiles(dataset_orig_dir)
listOfImages = files_tensor(listOfFiles)

# Labels 
label = np.zeros(sum(N))
labels = [x for x in range(len(names))]

for i in range(0,NC[0]):
    label[i] = labels[0]
    
for k in range(1,len(N)):
    for i in range(NC[k-1],NC[k]):
        label[i] = labels[k]

# Data and labels    
dataset_orig = [listOfImages, label]


# %% ----- detecting number of images per class --------------------------------------------------------------------------

os.chdir(dataset_fool_dir)

DirListTop0 = os.listdir()
names0 = DirListTop0

N0 = []

for x in names0:
    
    os.chdir(x)
    DirListSub = os.listdir()
    N0.append(len(DirListSub))
    os.chdir(dataset_fool_dir)
    
NC0 = np.cumsum(N0)
    

# %% ----- Obtaining and storing data ------------------------------------------------------------------------------------

# Data
listOfFiles0 = getListOfFiles(dataset_fool_dir)
listOfImages0 = files_tensor(listOfFiles0)

# Labels 
label0 = np.zeros(sum(N0))
labels0 = [x for x in range(len(names0))]

for i in range(0,NC0[0]):
    label0[i] = labels0[0]
    
for k in range(1,len(N0)):
    for i in range(NC0[k-1],NC0[k]):
        label0[i] = labels0[k]

# Data and labels    
dataset_fool = [listOfImages0, label0]

# %% ----- Soft coding input size ----------------------------------------------------------------------------------------
    
if input_size % 2 == 0:
    Size_1 = input_size / 2
else:
    Size_1 = (input_size - 1) / 2
    
if Size_1 % 2 == 0:
    Size_2 = Size_1 / 2
else:
    Size_2 = (Size_1 - 1) / 2
    
Size_2 = int(Size_2)


# %% ----- Defining the neural net --------------------------------------------------------------------------------------

#note: nn.Conv2D(in, out, kernel, stride=1, padding)
#note: nn.MaxPool2d(kernel, stride, padding)

class BN_Net(nn.Module):
    
    def __init__(self):
        super(BN_Net, self).__init__() 
        

        self.features = nn.Sequential(
                                      nn.Conv2d(1,5,5,1,2),
                                      nn.MaxPool2d(2,2),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(5),
                                      nn.Conv2d(5,8,5,1,2),
                                      nn.MaxPool2d(2,2),
                                      nn.ReLU(inplace=True), 
                                      nn.BatchNorm2d(8)
                                         )
        
        self.classifier = nn.Sequential(
                                        nn.Linear(8*Size_2*Size_2, 120),
                                        nn.ReLU(inplace=True),
                                        nn.BatchNorm1d(120),
                                        nn.Linear(120,84),
                                        nn.ReLU(inplace=True),
                                        nn.BatchNorm1d(84),
                                        nn.Linear(84,5)
                                        )
                      
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 8*Size_2*Size_2)
        x = self.classifier(x)
        
        return x
    
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__() 
        

        self.features = nn.Sequential(
                                      nn.Conv2d(1,5,5,1,2),
                                      nn.MaxPool2d(2,2),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(5,8,5,1,2),
                                      nn.MaxPool2d(2,2),
                                      nn.ReLU(inplace=True)
                                         )
        
        self.classifier = nn.Sequential(
                                        nn.Linear(8*Size_2*Size_2, 120),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(120,84),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(84,5)
                                        )
                      
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 8*Size_2*Size_2)
        x = self.classifier(x)
        
        return x
    
if batchnorm_flag:
    net = BN_Net()
else:
    net = Net()

#net.cuda()

checkpoint = torch.load(saved_weights)
net.load_state_dict(checkpoint['model_state_dict'])


# %% ----- baseline uncertainty -------------------------------------------------------------------------------------------

if baseline_flag:
    
    data_loader_fool = batch_generator(BS, dataset_fool)
    
    unc = []
    
    with torch.no_grad():    
        for data in data_loader_fool:
            images, labels = data
            images = torch.stack(images)
            images = torch.unsqueeze(images, 1)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            
            correct_index = predicted == string_tensor(labels).long()
    
            for x in range(len(outputs)):
                
                probs = softmax(outputs[x].cpu().numpy())
                prob = np.max(probs)
                uncert = 100*(1 - prob)
                
                
                unc.append(uncert)
                           
    ftl1 = r'{}\unc_correct.mat'.format(saved_run_dir)
    ftl2 = r'{}\unc_wrong.mat'.format(saved_run_dir)
    
    loaded_1 = sio.loadmat(ftl1)
    saved_unc_correct = loaded_1['data']
    suc_list = list(saved_unc_correct[0])
    
    loaded_2 = sio.loadmat(ftl2)
    saved_unc_wrong = loaded_2['data']
    suw_list = list(saved_unc_wrong[0])
    
    tot_list = suc_list + suw_list
    
    
    
    
    df1 = pd.DataFrame({'1':suc_list})
    df2 = pd.DataFrame({'2':suw_list})
    df3 = pd.DataFrame({'3':unc})
    df4 = pd.DataFrame({'4':tot_list})
    
    df = pd.concat([df4, df3], ignore_index=True, axis=1)
    df.columns = ['original dataset','fooling dataset']        
    
    plt.rcParams.update({'font.size': fig_font_size})
    
    baseline_plot = plt.figure(figsize=(9,7))
    plt.grid(1, which='major')
    plt.grid(1, which='minor', color='k', linestyle='-', alpha=0.08)
    plt.minorticks_on()
    sbplot = sb.stripplot(data=df)
    plt.ylabel('uncertainty in %') 
    plt.title('baseline uncertainty plot')
     
    
    if threshold_flag:
        x = plt.gca().axes.get_xlim()
        plt.plot(x, len(x)*[threshold],'r')
        
        L1C = len(suc_list)
        L1W = len(suw_list)
        L1T = L1C + L1W
        L1_acc = round(100 * L1C / L1T, 1)
        
        L2T = len(unc)
        
        suc_list_thr = [x for x in suc_list if x < threshold]
        suw_list_thr = [x for x in suw_list if x < threshold]
        L1C_thr = len(suc_list_thr)
        L1W_thr = len(suw_list_thr)
        L1T_thr = L1C_thr + L1W_thr
        L1_acc_thr = round(100 * L1C_thr / L1T_thr,1)
        
        unc_thr = [x for x in unc if x < threshold]
        L2T_thr = len(unc_thr)
            
        throw_1 = round(100 * (1 - L1T_thr / L1T),1)
        throw_2 = round(100 * (1 - L2T_thr / L2T),1)
    
        
        print('\n-------------------------------------------------------------------')
        print('\nAccuracy on the original dataset: \t{} %'.format(L1_acc))
        print('\nThreshold: {} %'.format(threshold))
        print('\nAccuracy on thresholded original dataset: \t{} %'.format(L1_acc_thr))
        print('\nPart of the original dataset thrown away: \t{} %'.format(throw_1))
        print('Part of the secondary dataset thrown away: \t{} %'.format(throw_2))
        print('\n-------------------------------------------------------------------')
        
        os.chdir(fname)
        
        f = open('baseline_results.txt','w+')
        f.write('\n\n-------------------------------------------------------------------')
        f.write('\n\nAccuracy on the original dataset: \t{} %'.format(L1_acc))
        f.write('\n\nThreshold: {} %'.format(threshold))
        f.write('\n\nAccuracy on thresholded original dataset: \t{} %'.format(L1_acc_thr))
        f.write('\n\nPart of the original dataset thrown away: \t{} %'.format(throw_1))
        f.write('\nPart of the secondary dataset thrown away: \t{} %'.format(throw_2))
        f.write('\n\n-------------------------------------------------------------------')
        f.close()
        
    
# %% ----- MAV distance ---------------------------------------------------------------------------------------------------

if MAV_flag:
    
    ftl1 = r'{}\MAV.mat'.format(saved_run_dir)
    ftl2 = r'{}\MAV_dist_cor.mat'.format(saved_run_dir)
    ftl3 = r'{}\MAV_dist_inc.mat'.format(saved_run_dir)
    
    loaded_1 = sio.loadmat(ftl1)
    MAV = loaded_1['data']
    
    loaded_2 = sio.loadmat(ftl2)
    dist_cor = loaded_2['data']
    
    dist_0 = dist_cor[0][0][0]
    dist_1 = dist_cor[0][1][0]
    dist_2 = dist_cor[0][2][0]
    dist_3 = dist_cor[0][3][0]
    dist_4 = dist_cor[0][4][0]
    
    max_0 = max(dist_0)
    max_1 = max(dist_1)
    max_2 = max(dist_2)
    max_3 = max(dist_3)
    max_4 = max(dist_4)
    
    maxlist = [max_0, max_1, max_2, max_3, max_4]
    maxarray = np.array(maxlist)
    meanarray = np.array([np.mean(np.array(dist_0)), np.mean(np.array(dist_1)), np.mean(np.array(dist_2)), np.mean(np.array(dist_3)), np.mean(np.array(dist_4))])
    
    dists_list = [dist_0, dist_1, dist_2, dist_3, dist_4]
    dist_tot = [x for i in dists_list for x in i]
    
    list_1 = list(dist_tot)
    
    loaded_3 = sio.loadmat(ftl3)
    dist_inc = loaded_3['data']
    list_2 = list(dist_inc[0])
    
    data_loader_fool = batch_generator(BS, dataset_fool)
    
    dist_test = []
    
    with torch.no_grad():    
        for data in data_loader_fool:
            images, labels = data
            images = torch.stack(images)
            images = torch.unsqueeze(images, 1)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            
            correct_index = predicted == string_tensor(labels).long()
            
            for x in range(len(outputs)):
                avdiff = outputs[x].cpu().numpy() - MAV[predicted[x]]
                avdist = np.linalg.norm(avdiff)
                
                dist_test.append(avdist)
                
    df1 = pd.DataFrame({'1':list_1})
    df2 = pd.DataFrame({'2':list_2})
    df3 = pd.DataFrame({'3':dist_test})
    df = pd.concat([df1, df2, df3], ignore_index=True, axis=1)
    df.columns = ['base_correct','base_incorrect','test']        
    
    
    MAV_dist_plot = plt.figure(figsize=(9,7))
    plt.grid(1, which='major')
    plt.grid(1, which='minor', color='k', linestyle='-', alpha=0.08)
    plt.minorticks_on()
    sbplot = sb.stripplot(data=df)
    plt.ylabel('distance to MAV')        

    if thresh_flag:
        
        x = plt.gca().axes.get_xlim()
        plt.plot(x, len(x)*[thresh],'r')
        
        list_1_thr = [x for x in list_1 if x < thresh]
        list_2_thr = [x for x in list_2 if x < thresh]
        dist_test_thr = [x for x in dist_test if x < thresh]
        
        L1C = len(list_1)
        L1W = len(list_2)
        L1T = len(dist_test)
        
        L2C = len(list_1_thr)
        L2W = len(list_2_thr)
        L2T = len(dist_test_thr)
        
        A1 = round(100 * L1C / (L1C + L1W), 1)
        A2 = round(100 * L2C / (L2C + L2W), 1)
        
        P1 = round(100 * (1 - (L2C + L2W) / (L1C + L1W)), 1)
        P2 = round(100 * (1 - L2T / L1T), 1)
        
        
        print('\n-------------------------------------------------------------------')
        print('\nAccuracy on the original dataset: \t{} %'.format(A1))
        print('\nThreshold: {}'.format(thresh))
        print('\nAccuracy on thresholded original dataset: \t{} %'.format(A2))
        print('\nPart of the original dataset thrown away: \t{} %'.format(P1))
        print('Part of the fooling dataset thrown away: \t{} %'.format(P2))
        print('\n-------------------------------------------------------------------')
        
        
        
        
        
# %% ----- OPENMAX ------------------------------------------------------------------------------------------------------------
        
if MAV_flag:
    if openmax_flag:
        
        tail = 2
        open_thresh = 0.007
        alpha = 4
        
        
        weibull_models = weibull_tailfitting(maxarray, MAV, taillength=tail)
        
        data_loader_fool = batch_generator(BS, dataset_fool)
    
        n_open_set = 0
        n_set = 0
        SMM_fool = []
        SMM_orig = []
        
        with torch.no_grad():    
            for data in data_loader_fool:
                images, labels = data
                images = torch.stack(images)
                images = torch.unsqueeze(images, 1)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                
                correct_index = predicted == string_tensor(labels).long()
                
                for x in outputs:
                    openmax_probs = recalibrate_scores(weibull_models, x.numpy(), alpharank=alpha)
                    n_set += 1
                    
                    SMM = max(openmax_probs[:-1])
                    SMM_fool.append(SMM)
                    
                    if SMM < open_thresh:
                        n_open_set += 1
        
        
        
        open_throw = round(100 * n_open_set / n_set, 1)
        
        print('\nOpenMax threw out {} % of the fooling dataset'.format(open_throw))
        
        
        data_loader_orig = batch_generator(BS, dataset_orig)
        
        with torch.no_grad():    
            for data in data_loader_orig:
                images, labels = data
                images = torch.stack(images)
                images = torch.unsqueeze(images, 1)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                
                correct_index = predicted == string_tensor(labels).long()
                
                for x in outputs:
                    openmax_probs = recalibrate_scores(weibull_models, x.numpy(), alpharank=alpha)
                    n_set += 1
                    
                    SMM = max(openmax_probs[:-1])
                    SMM_orig.append(SMM)
                    
                    if SMM < open_thresh:
                        n_open_set += 1
        
        
        
        open_throw_2 = round(100 * n_open_set / n_set, 1)
        
        print('OpenMax threw out {} % of the original dataset'.format(open_throw_2))
        print('\nDifference: {:.1f} %'.format(open_throw - open_throw_2))
        
        df1 = pd.DataFrame({'1':SMM_orig})
        df2 = pd.DataFrame({'2':SMM_fool})
        df = pd.concat([df1, df2], ignore_index=True, axis=1)
        df.columns = ['original','fool']        
        
        
        openmax_plot = plt.figure(figsize=(9,7))
        plt.grid(1, which='major')
        plt.grid(1, which='minor', color='k', linestyle='-', alpha=0.08)
        plt.minorticks_on()
        sbplot = sb.stripplot(data=df)
        plt.ylabel('max prob 1:5')    
        x = plt.gca().axes.get_xlim()
        plt.plot(x, len(x)*[open_thresh],'r')
        
        
        





# %% ----- Saving results -------------------------------------------------------------------------------------------------


    
os.chdir(fname)

if baseline_flag:
    path = '{}\\baseline_uncertainty_plot.png'.format(fname)
    baseline_plot.savefig(path, dpi=fig_dpi)
    
if MAV_flag:
    path = '{}\\MAV_distance_plot.png'.format(fname)
    MAV_dist_plot.savefig(path, dpi=fig_dpi) 
    
if openmax_flag:
    f = open('openmax info.txt','w+')
    f.write('-----------------------------------------------------------------------------------------')
    f.write('\n\nparameters used:')
    f.write('\n\ntail = {}'.format(tail))
    f.write('\nopen_thresh = {}'.format(open_thresh))
    f.write('\nalpha = {}'.format(alpha))
    f.write('\n\n-----------------------------------------------------------------------------------------')
    f.write('\n\nResults:')
    f.write('\n\nOpenMax threw out {} % of the fooling dataset'.format(open_throw))
    f.write('\nOpenMax threw out {} % of the original dataset'.format(open_throw_2))
    f.write('\n\nDifference: {:.1f} %'.format(open_throw - open_throw_2))
    f.write('\n\n-----------------------------------------------------------------------------------------')
    f.write('\n\nValue of the random seed: {}'.format(SeedVal))
    f.close()
    
    path = '{}\\openmax_plot.png'.format(fname)
    openmax_plot.savefig(path, dpi=fig_dpi)




# %% 

os.chdir(DT)
os.startfile(fname)

del net


