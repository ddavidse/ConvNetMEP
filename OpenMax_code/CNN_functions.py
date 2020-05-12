# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 11:17:17 2019

@author: ddavidse
"""

import scipy
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import random
import numpy as np
import itertools
import copy

"""
-----------------------------------------------------------------------------------------
Functions for handling data
-----------------------------------------------------------------------------------------
"""

def files_tensor(images):
    
    img = []    
    for i in range(len(images)):        
        A1 = scipy.io.loadmat(images[i])
        A2 = list(A1.values())
        A3 = A2[3]
        pix_val = torch.tensor(data = A3, dtype=torch.float)
        img.append(pix_val)        
    return torch.stack(img)



def getListOfFiles(dirName):
    
    listOfFiles = os.listdir(dirName)
    allFiles = list()
    
    for entry in listOfFiles:
        
        fullPath = os.path.join(dirName, entry)        
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)             
    return allFiles

"""
-----------------------------------------------------------------------------------------
Batch generator
-----------------------------------------------------------------------------------------
"""


def batch_generator(batch_size, dataset): 
        
    batch = [[],[]]
    counter = 0
    
    shuffle_list = list(zip(dataset[0], dataset[1]))
    random.shuffle(shuffle_list) 
    data, labels = zip(*shuffle_list) 

    while True:

        batch[0].append(data[counter])
        batch[1].append(labels[counter])
           
        counter += 1
    
        if counter % batch_size == 0:            
            yield batch # Note: 'yield' makes this a generator instead of a function
            del batch
            batch = [[],[]]
            
        if len(data) < counter + 1:
            break            

"""
-----------------------------------------------------------------------------------------
Some utility
-----------------------------------------------------------------------------------------
"""
        
def string_tensor(labels):
    lbl=[]
    for i in range(len(labels)):
        float_label = float(labels[i])
        lbl.append(float_label)
    return torch.tensor(lbl)


          
def sec_max(numbers):
    count = 0
    m1 = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1            
            else:
                m2 = x
    return m2 if count >= 2 else None



def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.uniform_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.uniform_(m.bias)
        
    
        
"""
-----------------------------------------------------------------------------------------
Confusion matrix
-----------------------------------------------------------------------------------------
"""
        
def plot_confusion_matrix(cm, classes, normalize='none', cmap=plt.cm.Blues, percentage=False):
    
    if normalize == 'row':
        normalize_flag = True
        cmnl = []
        for i in range(len(cm)):
            cmnl.append(cm[i] / float(sum(cm[i])))
        cm = torch.stack(cmnl)
        if percentage:
            cm = 100*cm
            title = 'Confusion matrix, normalize = row, %'
        else:
            title = 'Confusion matrix, normalize = row'
        
    elif normalize == 'full':
        normalize_flag = True
        cmnl = []
        for i in range(len(cm)):
            cmnl.append(cm[i] / float(sum(sum(cm))))
        cm = torch.stack(cmnl)
        if percentage:
            cm = 100*cm
            title = 'Confusion matrix, normalize = full, %'
        else:
            title = 'Confusion matrix, normalize = full'
    else:
        title = 'Confusion matrix'
        normalize_flag = False
                
    #print('\n',cm)
      
    plt.imshow(cm, interpolation='none', cmap=cmap)
    
    plt.title(title)
    plt.colorbar(shrink=0.9)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
     
    if normalize_flag and not percentage:
        fmt = '.2f'
    elif normalize_flag and percentage:
        fmt = '.1f'
    else:
        fmt = 'd'
        
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
"""
-----------------------------------------------------------------------------------------
Weight transform functions
-----------------------------------------------------------------------------------------
"""


def weights_to_list(net):
    
    A = net.state_dict()
    B = A.values()
    C = [x.detach().cpu().numpy() for x in B]
    
    X11 = C[0].tolist()
    X1F = [x for list in X11 for sub1 in list for sub2 in sub1 for x in sub2]
    X2F = C[1].tolist()
    X31 = C[2].tolist()
    X3F = [x for list in X31 for sub1 in list for sub2 in sub1 for x in sub2]
    X4F = C[3].tolist()
    X51 = C[4].tolist()
    X5F = [x for list in X51 for x in list]
    X6F = C[5].tolist()
    X71 = C[6].tolist()
    X7F = [x for list in X71 for x in list]
    X8F = C[7].tolist()
    X91 = C[8].tolist()
    X9F = [x for list in X91 for x in list]
    X10F = C[9].tolist()
    
    weight_list = X1F + X2F + X3F + X4F + X5F + X6F + X7F + X8F + X9F + X10F
    
    return weight_list



def list_to_weights(wvec, net, deepcopy=False):
    
    A = net.state_dict()
    S = [x.size() for x in A.values()]

    SL = [list(x) for x in S]
        
    L = len(SL)
    Sizes = []
    WeightTensors = []
    ite = 0
    
    for x in range(L):
        ite += 1
        xsize = 1
        for number in SL[x]:
            xsize = xsize * number
        Sizes.append(xsize)
        Sizes_C = np.cumsum(Sizes)
        
        if x == 0:
            weights = wvec[0:Sizes[x]]
        else:
            weights = wvec[Sizes_C[x-1]:Sizes_C[x]]
            
        weights_array = np.array(weights)
        weights_reshaped = weights_array.reshape(SL[x])
        weights_tensor = torch.tensor(weights_reshaped, dtype=torch.float32)
        
        WeightTensors.append(weights_tensor)
    
    Keys = net.state_dict().keys()
    KeyList = list(Keys)
    
    if deepcopy:
        net2 = copy.deepcopy(net)
    
    for k in range(len(WeightTensors)):
        for i in range(WeightTensors[k].size()[0]): 
            if ~deepcopy:
                net.state_dict()[KeyList[k]][i] = WeightTensors[k][i]
            elif deepcopy:
                net2.state_dict()[KeyList[k]][i] = WeightTensors[k][i]
    
    if deepcopy:
        return net2
    else: print('weights replaced')