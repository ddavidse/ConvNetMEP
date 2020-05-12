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

from evt_fitting_V2 import weibull_tailfitting
from compute_openmax_V2 import recalibrate_scores, computeOpenMaxProbability


# %%  ----- Delete old net -----------------------------------------------------------------------------------------------

if 'net' in locals():
    del net

# %%  ----- Set Parameters and Folders -----------------------------------------------------------------------------------

CUDA_flag = False

# Place where output folders are created:
DT = r'C:\Users\ddavidse\Desktop'

# Location of original data:
dirName = r'C:\SharedData\Datasets for CNN\100x100 downsampled from 150x150'

# Location of fooling data:
dataset_fool_dir = r'C:\SharedData\Datasets for CNN\elephant set'
#dataset_fool_dir = r'C:\SharedData\Datasets for CNN\mirrored set'

BS = 50             # batch size
epochs = 12         # number of epochs for training
fig_dpi = 150       # dpi for saved figures
fig_font_size = 12  # font size for figures

# %% ----- Control randomness -----------------------------------------------------------------------------------

random_initialization = True        # set False to use saved_weights
saved_weights = r'D:\ddavidse\Desktop\5 class network runs\network runs - background - full data\Network_run_16\initial_weights.pth'

set_random_seed = False             # set True to use seed_value
seed_value = 784410

# %% ----- Configure dataset split ---------------------------------------------------------------------------------------

ValFrac = 1/6       # set fraction of total dataset used for validation
TestFrac = 1/6      # set fraction of total dataset used for testing

# %% ----- Select features -----------------------------------------------------------------------------------------------

batchnorm_flag = True                   # make True to use batch normalization
misclassified_images_flag = False       # make True to output misclassified images
misclassified_outputs_flag = False      # make True to output misclassified net output values to run info.txt
loss_landscape_flag = False             # make True to create a loss landscape based on random planes
PCA_flag = False                        # make True to create a loss landscape based on PCA

baseline_flag = False                   # make True to use probability thresholding
MAV_flag = True                         # make True to do all the MAV magic
openmax_flag = True                     # make True to use OpenMax (requires MAV_flag to be true as well)

# %% ----- Loss landscapes package settings ------------------------------------------------------------------------------

STEPS = 40      # resolution of landscape, higher = more detail          

# %% ----- PCA settings --------------------------------------------------------------------------------------------------

FilterSteps = 40
filter_normalization_flag = True
distance_multiplier = 2
bowl = False
contour_center = False

# %%  ----- CUDA initialization ------------------------------------------------------------------------------------------

if CUDA_flag:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
                
# %% ----- Create output folder ------------------------------------------------------------------------------------------
    
os.chdir(DT)
mfname = 'Network_runs'

if not os.path.exists(mfname):
    os.mkdir(mfname)
        
fdir = '{}\\{}'.format(DT,mfname)

counter = 0
tv = 0

while tv == 0:
    
    counter += 1
    fname = '{}\\Network_run_{}'.format(fdir, counter)  

    if not os.path.exists(fname):
        os.mkdir(fname)
        tv = 1
        

# %% ----- detecting number of images per class --------------------------------------------------------------------------

os.chdir(dirName)

DirListTop = os.listdir()
names = DirListTop

N = []

for x in names:
    
    os.chdir(x)
    DirListSub = os.listdir()
    N.append(len(DirListSub))
    os.chdir(dirName)
    
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
listOfFiles = getListOfFiles(dirName)
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
shuffle_list = list(zip(listOfImages, label))
dataset_orig = [listOfImages, label]


# %% ----- Creating sets with equal class distribution -------------------------------------------------------------------

if set_random_seed == True:
    SeedVal = seed_value
else:
    SeedVal = random.randrange(1,1000000)
    
random.seed(SeedVal)

zipZ = [[] for x in range(len(N))]
zip_train = [[] for x in range(len(N))]
zip_val = [[] for x in range(len(N))]
zip_test = [[] for x in range(len(N))]

zipZ[0] = shuffle_list[0:NC[0]]

for k in range(1,len(N)):
    zipZ[k] = shuffle_list[NC[k-1]:NC[k]]

for k in range(len(N)):
    random.shuffle(zipZ[k])


TrainFrac = 1 - ValFrac - TestFrac

for k in range(len(N)):
    zip_train[k] = zipZ[k][0:int(TrainFrac*N[k])]
    zip_val[k] = zipZ[k][int(TrainFrac*N[k]):int((TrainFrac+ValFrac)*N[k])]
    zip_test[k] = zipZ[k][int((TrainFrac+ValFrac)*N[k]):N[k]]

    
train = [x for s in zip_train for x in s]
val = [x for s in zip_val for x in s]
test = [x for s in zip_test for x in s]

files1, labels1 = zip(*test)
test_set = [files1, labels1]

files2, labels2 = zip(*train)
train_set = [files2, labels2]

files3, labels3 = zip(*val)
val_set = [files3, labels3]


# %% ----- Adjusting sets to fit in batches (optional code, not important) -----------------------------------------------

diff_val = BS - (len(val_set[0]) % BS);

if diff_val == 2:   
    
    c_images = []
    c_labels = []
    
    c_images.append(train_set[0][0])
    c_labels.append(train_set[1][0])
  
    c_images.append(train_set[0][len(train_set[0]) - 1])
    c_labels.append(train_set[1][len(train_set[1]) - 1])
        
    c_images = tuple(c_images)
    c_labels = tuple(c_labels)
    
    val_set[0] = val_set[0] + c_images
    val_set[1] = val_set[1] + c_labels
    
    train_set[0] = train_set[0][1:len(train_set[0])-1]
    train_set[1] = train_set[1][1:len(train_set[1])-1]
    
elif diff_val == 1:
    
    c_image = train_set[0][len(train_set[0]) - 1]
    c_label = train_set[1][len(train_set[1]) - 1]
    
    t0list = list(val_set[0])
    t0list.append(c_image)
    val_set[0] = tuple(t0list)
    
    t1list = list(val_set[1])
    t1list.append(c_label)
    val_set[1] = tuple(t1list)
    
    train_set[0] = train_set[0][0:len(train_set[0])-2]
    train_set[1] = train_set[1][0:len(train_set[1])-2]
    
    
diff_test = BS - (len(test_set[0]) % BS);

if diff_test == 2:   
    
    c_images = []
    c_labels = []
    
    c_images.append(train_set[0][0])
    c_labels.append(train_set[1][0])
  
    c_images.append(train_set[0][len(train_set[0]) - 1])
    c_labels.append(train_set[1][len(train_set[1]) - 1])
        
    c_images = tuple(c_images)
    c_labels = tuple(c_labels)
    
    test_set[0] = test_set[0] + c_images
    test_set[1] = test_set[1] + c_labels
    
    train_set[0] = train_set[0][1:len(train_set[0])-1]
    train_set[1] = train_set[1][1:len(train_set[1])-1]
    
elif diff_test == 1:
    
    c_image = train_set[0][len(train_set[0]) - 1]
    c_label = train_set[1][len(train_set[1]) - 1]
    
    t0list = list(test_set[0])
    t0list.append(c_image)
    test_set[0] = tuple(t0list)
    
    t1list = list(test_set[1])
    t1list.append(c_label)
    test_set[1] = tuple(t1list)
    
    train_set[0] = train_set[0][0:len(train_set[0])-1]
    train_set[1] = train_set[1][0:len(train_set[1])-1]
    
diff_train = len(test_set[0]) % BS

if diff_train < 5:
    
    c_image = train_set[0][len(test_set[0]) - 1]
    c_label = train_set[1][len(test_set[1]) - 1]
    
    t0list = list(train_set[0])
    t0list.append(c_image)
    train_set[0] = tuple(t0list)
    
    t1list = list(train_set[1])
    t1list.append(c_label)
    train_set[1] = tuple(t1list)
    
    test_set[0] = test_set[0][0:len(test_set[0])-1]
    test_set[1] = test_set[1][0:len(test_set[1])-1]
    

# %% ----- Batch generator and loaders ----------------------------------------------------------------------------------
    
train_loader = batch_generator (BS, train_set)
test_loader = batch_generator (BS, test_set)
val_loader = batch_generator (BS, val_set)


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


# %% ----- Defining the neural nets --------------------------------------------------------------------------------------

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
      

# %% ----- Initialize or load weights -----------------------------------------------------------------------------

if batchnorm_flag:
    net = BN_Net()
else:
    net = Net()

if CUDA_flag:
    net.cuda()

if random_initialization:
    net.apply(weights_init)
else:
    checkpoint = torch.load(saved_weights)
    net.load_state_dict(checkpoint['model_state_dict'])
      
    
# %% ----- Loss and optimizer ---------------------------------------------------------------------------------------

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
#optimizer = optim.RMSprop(net.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
#optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False)
model_initial = copy.deepcopy(net)


# %% ----- Training the net ---------------------------------------------------------------------------------------------

start = time.time()

running_loss_history = []
running_corrects_history = []
val_running_loss_history = []
val_running_corrects_history = []

StateVecList = []

for e in range(epochs):
    
    print('\nepoch :', (e+1))  
    running_loss = 0.0
    running_corrects = 0.0
    val_running_loss = 0.0
    val_running_corrects = 0.0
    train_loader = batch_generator(BS, train_set)
    val_loader = batch_generator(BS, val_set)
    
    train_amount = 0.0
    val_amount = 0.0
      
    for inputs, labels in train_loader:
        
        labels = string_tensor(labels)
        inputs = torch.stack(inputs)
        inputs = torch.unsqueeze(inputs, 1)
        outputs = net(inputs)
        loss = criterion(outputs, labels.long())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.long().data)
        
        train_amount += BS
    
    if PCA_flag:
        #StateVecList.append(weights_to_list(net))
        print('nvm')

    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            
            val_inputs = torch.stack(val_inputs)
            val_inputs = torch.unsqueeze(val_inputs, 1)
            val_labels = string_tensor(val_labels)
            val_outputs = net(val_inputs)
            val_loss = criterion(val_outputs, val_labels.long())
        
            _, val_preds = torch.max(val_outputs, 1)
            val_running_loss += val_loss.item()
            val_running_corrects += torch.sum(val_preds == val_labels.long().data)
            
            val_amount += BS
    
    epoch_loss = running_loss
    epoch_acc = running_corrects
    running_loss_history.append(epoch_loss)
    running_corrects_history.append(epoch_acc)
    val_epoch_loss = val_running_loss
    val_epoch_acc = val_running_corrects
    val_running_loss_history.append(val_epoch_loss)
    val_running_corrects_history.append(val_epoch_acc)
     
    print('\ntraining loss: {:.4f} '.format(epoch_loss))
    print('validation loss: {:.4f}'.format(val_epoch_loss))

if CUDA_flag:
    torch.cuda.synchronize()
    
end = time.time()
print('\n-----------------------------------------------------------------------')
print('\nTime of training: {:d} s'.format(round((end - start))))


# %% ----- Prediction and training stats and graphs --------------------------------------------------------------------

plt.rcParams.update({'font.size': fig_font_size})

correct = [0,0,0]
total = [0,0,0]
testsize = [0,0,0]
AV_all = [[],[],[],[],[]]

with torch.no_grad():
    for data in test_loader:
        images_0_0, labels_0 = data
        images_0_1 = torch.stack(images_0_0)
        images_0_2 = torch.unsqueeze(images_0_1, 1)
        outputs_0 = net(images_0_2)
        _, predicted_0 = torch.max(outputs_0.data, 1)
        total[0] += string_tensor(labels_0).size(0)
        testsize[0] += 1
        
        correct_index_0 = predicted_0 == string_tensor(labels_0).long()
        correct[0] += correct_index_0.sum().item()
        
        if MAV_flag:
            for x in range(len(outputs_0)):
                if correct_index_0[x]:
                    AV_all[predicted_0[x]].append(outputs_0[x])
        
    train_loader = batch_generator(BS, train_set)
    val_loader = batch_generator(BS, val_set)
    
    for data in train_loader:
        images_1_0, labels_1 = data
        images_1_1 = torch.stack(images_1_0)
        images_1_2 = torch.unsqueeze(images_1_1, 1)
        outputs_1 = net(images_1_2)
        _, predicted_1 = torch.max(outputs_1.data, 1)
        total[1] += string_tensor(labels_1).size(0)
        testsize[1] += 1
        
        correct_index_1 = predicted_1 == string_tensor(labels_1).long()
        correct[1] += correct_index_1.sum().item()
        
        if MAV_flag:
            for x in range(len(outputs_1)):
                if correct_index_1[x]:
                    AV_all[predicted_1[x]].append(outputs_1[x])
        
    for data in val_loader:
        images_2_0, labels_2 = data
        images_2_1 = torch.stack(images_2_0)
        images_2_2 = torch.unsqueeze(images_2_1, 1)
        outputs_2 = net(images_2_2)
        _, predicted_2 = torch.max(outputs_2.data, 1)
        total[2] += string_tensor(labels_2).size(0)
        testsize[2] += 1
        
        correct_index_2 = predicted_2 == string_tensor(labels_2).long()
        correct[2] += correct_index_2.sum().item()
        
        if MAV_flag:
            for x in range(len(outputs_2)):
                if correct_index_2[x]:
                    AV_all[predicted_2[x]].append(outputs_2[x])

if MAV_flag:  
    
    MAV = np.empty((5,5))
    
    for i in range(5):      
        AV_array = np.array([list(x.cpu().numpy()) for x in AV_all[i]])
        MAV[i] = np.mean(AV_array, axis=0)
        
        
        
        
        
"""
================================================
================================================
===== SECONDARY DATASET LOOPS FOR MAV
================================================
================================================
"""


AV_dist_correct = [[],[],[],[],[]]
AV_dist_wrong = []

if MAV_flag:

    train_loader = batch_generator(BS, train_set)
    val_loader = batch_generator(BS, val_set)
    test_loader = batch_generator(BS, test_set)
    
    with torch.no_grad():
        for data in test_loader:
            images_0_0, labels_0 = data
            images_0_1 = torch.stack(images_0_0)
            images_0_2 = torch.unsqueeze(images_0_1, 1)
            outputs_0 = net(images_0_2)
            _, predicted_0 = torch.max(outputs_0.data, 1)
            
            correct_index_0 = predicted_0 == string_tensor(labels_0).long()
            
            
            for x in range(len(outputs_0)):
                avdiff = outputs_0[x].cpu().numpy() - MAV[predicted_0[x]]
                avdist = np.linalg.norm(avdiff)
                
                if correct_index_0[x]:
                    AV_dist_correct[predicted_0[x]].append(avdist)
                else:
                    AV_dist_wrong.append(avdist)
            
            
        
        for data in train_loader:
            images_1_0, labels_1 = data
            images_1_1 = torch.stack(images_1_0)
            images_1_2 = torch.unsqueeze(images_1_1, 1)
            outputs_1 = net(images_1_2)
            _, predicted_1 = torch.max(outputs_1.data, 1)
            
            correct_index_1 = predicted_1 == string_tensor(labels_1).long()
            
            for x in range(len(outputs_1)):
                avdiff = outputs_1[x].cpu().numpy() - MAV[predicted_1[x]]
                avdist = np.linalg.norm(avdiff)
                
                if correct_index_1[x]:
                    AV_dist_correct[predicted_1[x]].append(avdist)
                else:
                    AV_dist_wrong.append(avdist)
            
            
        for data in val_loader:
            images_2_0, labels_2 = data
            images_2_1 = torch.stack(images_2_0)
            images_2_2 = torch.unsqueeze(images_2_1, 1)
            outputs_2 = net(images_2_2)
            _, predicted_2 = torch.max(outputs_2.data, 1)
            
            correct_index_2 = predicted_2 == string_tensor(labels_2).long()
                           
            for x in range(len(outputs_0)):
                avdiff = outputs_2[x].cpu().numpy() - MAV[predicted_2[x]]
                avdist = np.linalg.norm(avdiff)
                
                if correct_index_2[x]:
                    AV_dist_correct[predicted_2[x]].append(avdist)
                else:
                    AV_dist_wrong.append(avdist)
    
    AV_dist_tot = [x for i in AV_dist_correct for x in i]  
                
    AVCA = np.array(AV_dist_tot)
    AVC_max = np.max(AVCA)
    AVC_min = np.min(AVCA)
    AVC_avg = np.mean(AVCA)
    AVC_std = np.std(AVCA)
    
    AVWA = np.array(AV_dist_wrong)
    AVW_max = np.max(AVWA)
    AVW_min = np.min(AVWA)
    AVW_avg = np.mean(AVWA)
    AVW_std = np.std(AVWA)
    
    test_loader = batch_generator(BS, test_set)
    
    with torch.no_grad():
        
        ImagesChecked = 0
        ImagesCheckedCorrect = 0
        
        for data in test_loader:
            images_3_0, labels_3 = data
            images_3_1 = torch.stack(images_3_0)
            images_3_2 = torch.unsqueeze(images_3_1, 1)
            outputs_3 = net(images_3_2)
            _, predicted_3 = torch.max(outputs_3.data, 1)
            
            correct_index_3 = predicted_3 == string_tensor(labels_3).long()
            
            for x in range(len(outputs_3)):
                avdiff = outputs_3[x].cpu().numpy() - MAV[predicted_3[x]]
                avdist = np.linalg.norm(avdiff)
                if avdist < AVW_min:
                #if avdist < 2.5:
                    ImagesChecked += 1
                    if correct_index_3[x]:
                        ImagesCheckedCorrect += 1
                        
    
    df1 = pd.DataFrame({'1':AVCA})
    df2 = pd.DataFrame({'2':AVWA})
    df = pd.concat([df1, df2], ignore_index=True, axis=1)
    df.columns = ['correct','incorrect']        
    
    
    MAV_dist_plot = plt.figure(figsize=(9,7))
    plt.grid(1, which='major')
    plt.grid(1, which='minor', color='k', linestyle='-', alpha=0.08)
    plt.minorticks_on()
    sbplot = sb.stripplot(data=df)
    plt.ylabel('distance to MAV')        
    x = plt.gca().axes.get_xlim()
    plt.plot(x, len(x)*[AVW_min],'r')


# %%
    
    
"""
================================================
================================================
===== FOOLING DATASET
================================================
================================================

"""    

if MAV_flag:
    
    thresh_flag = True
    thresh = 4.7
    
    os.chdir(dataset_fool_dir)

    DirListTop_fool = os.listdir()
    names_fool = DirListTop_fool
    
    N_fool = []
    
    for x in names_fool:
        
        os.chdir(x)
        DirListSub_fool = os.listdir()
        N_fool.append(len(DirListSub_fool))
        os.chdir(dataset_fool_dir)
        
    NC_fool = np.cumsum(N_fool)
        
    listOfFiles_fool = getListOfFiles(dataset_fool_dir)
    listOfImages_fool = files_tensor(listOfFiles_fool)
    
    label_fool = np.zeros(sum(N_fool))
    labels_fool = [x for x in range(len(names_fool))]
    
    for i in range(0,NC_fool[0]):
        label_fool[i] = labels_fool[0]
        
    for k in range(1,len(N_fool)):
        for i in range(NC_fool[k-1],NC_fool[k]):
            label_fool[i] = labels_fool[k]
     
    dataset_fool = [listOfImages_fool, label_fool]
    
    list_1 = AV_dist_tot
    list_2 = AV_dist_wrong
    
    data_loader = batch_generator(BS, dataset_fool)
    
    dist_test = []
    
    with torch.no_grad():    
        for data in data_loader:
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
    
    df0 = pd.DataFrame({'0':list_1 + list_2})            
    df1 = pd.DataFrame({'1':list_1})
    df2 = pd.DataFrame({'2':list_2})
    df3 = pd.DataFrame({'3':dist_test})
    
    df = pd.concat([df0, df3], ignore_index=True, axis=1)
    df.columns = ['original set','fooling set']        
    
    
    MAV_dist_plot = plt.figure(figsize=(9,7))
    plt.grid(1, which='major')
    plt.grid(1, which='minor', color='k', linestyle='-', alpha=0.08)
    plt.minorticks_on()
    sbplot = sb.stripplot(data=df, jitter=0.15)
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
        
        
        
                
        os.chdir(fname)
        
        path = '{}\\MAV_distance_dist.png'.format(fname)
        MAV_dist_plot.savefig(path, dpi=fig_dpi)
        
        f = open('MAV results.txt','w+')
        f.write('--------------------------------------------------------------------------------------')
        f.write('\n\nAccuracy on the original dataset: \t\t{} %'.format(A1))
        f.write('\nAccuracy on thresholded original dataset: \t{} %'.format(A2))
        f.write('\n\nThreshold: {}'.format(thresh))    
        f.write('\n\nPart of the original dataset thrown away: \t{} %'.format(P1))
        f.write('\nPart of the secondary dataset thrown away: \t{} %'.format(P2))
        f.write('\n\n--------------------------------------------------------------------------------------')
        f.close()
        
# %%    
    
    
"""
================================================
================================================
===== OPENMAX
================================================
================================================
"""   

if openmax_flag:
        
    tail = 2
    open_thresh = 0.2
    alpha = 4
    
    maxarray = np.array([1,2,3,4,5])
    
    weibull_models = weibull_tailfitting(AV_dist_correct, MAV, names, tailsize=tail, distance_type='euclidean')
    
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
    
    
    print('\n-------------------------------------------------------------------------------------')
    print('\nOpenMax threw out {} % of the original dataset'.format(open_throw_2))
    print('OpenMax threw out {} % of the fooling dataset'.format(open_throw))
    print('\nDifference: {:.1f} %'.format(open_throw - open_throw_2))
    
                
    os.chdir(fname)
    
    path = '{}\openmax_plot.png'.format(fname)
    openmax_plot.savefig(path, dpi=fig_dpi)
    
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
    f.close()
    
    os.chdir(DT)
    
# %%    
    
    
"""
================================================
================================================
===== SECONDARY DATASET LOOPS FOR BASELINE
================================================
================================================
"""


if baseline_flag:
    
    train_loader = batch_generator(BS, train_set)
    val_loader = batch_generator(BS, val_set)
    test_loader = batch_generator(BS, test_set)
    
    unc_correct = []
    unc_wrong = []
    
    with torch.no_grad():
        
        for data in test_loader:
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
                
                if correct_index[x]:
                    unc_correct.append(uncert)
                else:
                    unc_wrong.append(uncert)
                    
                    
        for data in train_loader:
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
                
                if correct_index[x]:
                    unc_correct.append(uncert)
                else:
                    unc_wrong.append(uncert)
                    
                    
        for data in val_loader:
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
                
                
                if correct_index[x]:
                    unc_correct.append(uncert)
                else:
                    unc_wrong.append(uncert)
            

    df1 = pd.DataFrame({'1':unc_correct})
    df2 = pd.DataFrame({'2':unc_wrong})
    df = pd.concat([df1, df2], ignore_index=True, axis=1)
    df.columns = ['correct','wrong']        
    
    baseline_plot = plt.figure(figsize=(9,7))
    plt.grid(1, which='major')
    plt.grid(1, which='minor', color='k', linestyle='-', alpha=0.08)
    plt.minorticks_on()
    sbplot = sb.stripplot(data=df, jitter=.15)
    plt.ylabel('uncertainty in %') 
    plt.title('baseline uncertainty plot')
    
    

    

    
        
total_tot = sum(total)
correct_tot = sum(correct)
testsize_tot = sum(testsize)

print('\nAccuracy of the network on the test images: {:.1f} %'.format((100 * correct[0] / total[0])))
print('Test result obtained from {} images coming from {} batches'.format(total[0], testsize[0]))
print('\nAccuracy of the network on the total set of all images: {:.1f} %'.format((100 * correct_tot / total_tot)))
print('Test result obtained from {} images coming from {} batches'.format(total_tot, testsize_tot))

if MAV_flag:
    print('\nAmount of images in high confidence set: {}'.format(ImagesChecked))
    print('This is {:.1f}% of the test set'.format(100*ImagesChecked / total[0]))
    
    print('\nAmount of correctly classified images in high confidence set: {}'.format(ImagesCheckedCorrect))
    print('Accuracy on high confidence set: {:.1f} %'.format(100*ImagesCheckedCorrect / ImagesChecked))

#print('\nAmount of correct predictions in test 1: {}'.format(correct[0]))

if MAV_flag:
    print('\n-------------------------------------------------------------------------------------')
    print('\nAccuracy on the original dataset: \t\t{} %'.format(A1))
    print('Accuracy on thresholded original dataset: \t{} %'.format(A2))
    print('\nThreshold: {}'.format(thresh))
    print('\nPart of the original dataset thrown away: \t{} %'.format(P1))
    print('Part of the secondary dataset thrown away: \t{} %'.format(P2))
    if openmax_flag:
        print('\n-------------------------------------------------------------------------------------')
        print('\nOpenMax threw out {} % of the original dataset'.format(open_throw_2))
        print('OpenMax threw out {} % of the fooling dataset'.format(open_throw))
        print('\nDifference: {:.1f} %'.format(open_throw - open_throw_2))

val_plot = plt.figure(figsize=(9,7))
plt.grid(1, which='major')
plt.grid(1, which='minor', color='k', linestyle='-', alpha=0.08)
plt.minorticks_on()
plt.plot(running_loss_history, 'r',label='training loss')
plt.plot(val_running_loss_history,'b', label='validation loss')
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend()

v_len = len(val_set[0])
t_len = len(train_set[0])

acc_plot = plt.figure(figsize=(9,7))
plt.grid(1, which='major')
plt.grid(1, which='minor', color='k', linestyle='-', alpha=0.08)
plt.minorticks_on()
train_corrected = [100*float(x)/(train_amount) for x in running_corrects_history] 
plt.plot(train_corrected,'r', label='training accuracy [%]')
val_corrected = [100*float(x)/(val_amount) for x in val_running_corrects_history]
plt.plot(val_corrected,'b', label='validation accuracy [%]')
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend()


# %% ----- confusion matrix --------------------------------------------------------------------------------------------
# code inspiration from https://deeplizard.com/learn/video/0LhiS6yu2qQ

test_loader = batch_generator (BS, test_set)
all_preds = torch.tensor([])
all_labels = torch.tensor([])

misc_preds_t = []
misc_labels_t = []
misc_img_t = []
misc_outputs_t = []
maxdif_t = []


for inputs, labels in test_loader:
    
        inputs = torch.stack(inputs)
        inputs = torch.unsqueeze(inputs, 1)
        
        labels = string_tensor(labels)
        all_labels = torch.cat((all_labels, labels),dim=0)
        
        outputs = net(inputs)
        all_preds = torch.cat((all_preds, outputs),dim=0)
        
        imlist = list(inputs)
        outputs1 = outputs.argmax(dim=1)
        vec_equal = outputs1 == labels
        
        if misclassified_outputs_flag:
            
            misc_outputs_0 = outputs[~vec_equal]
            misc_outputs_1 = misc_outputs_0.cpu().detach().numpy().tolist()
            misc_outputs = [list(map(lambda x: round(x,1), i)) for i in misc_outputs_1]
            misc_outputs_t += misc_outputs
            
            maxdif = [max(x) - sec_max(x) for x in misc_outputs_1]
            maxdif_t += maxdif
        
        if misclassified_images_flag:
            
            misc_ind = [x for x in range(BS) if vec_equal[x] == False]
        
            misc_img = [imlist[x] for x in misc_ind]
            misc_img_t += misc_img
        
            misc_labels = [labels[x] for x in misc_ind]
            misc_labels_t += misc_labels
        
            misc_preds = [outputs1[x] for x in misc_ind]
            misc_preds_t += misc_preds             

       
AL = all_labels
AL2 = np.array(list(AL), dtype=np.int)
AL3 = torch.tensor(AL2)

TP = all_preds.argmax(dim=1)
TP2 = np.array(list(TP), dtype=np.int)
TP3 = torch.tensor(TP2)

ConfAcc = sum(TP3 == AL3).item()
#print('Amount of correct predictions in test 2: {}'.format(ConfAcc))
print('\n-----------------------------------------------------------------------------------')

stacked = torch.stack((AL3, TP3), dim=1)

cmt = torch.zeros(len(N),len(N), dtype=torch.int64)


for p in stacked:
    tl, pl = p.tolist()
    cmt[tl, pl] = cmt[tl, pl] + 1
    
cmt2 = cmt.cpu()

if len(names) == 3:
    conf_fig_size = (8,8)
elif len(names) == 4:
    conf_fig_size = (9,9)
else:
    conf_fig_size = (10,10)

conf_fig = plt.figure(figsize = conf_fig_size)
plot_confusion_matrix(cmt2, names)

conf_fig_2 = plt.figure(figsize = conf_fig_size)
plot_confusion_matrix(cmt2, names, normalize='row', percentage=True)

conf_fig_3 = plt.figure(figsize = conf_fig_size)
plot_confusion_matrix(cmt2, names, normalize='full', percentage=True)


# %% ----- loss landscape ----------------------------------------------------------------------------------------------
# source: https://github.com/marcellodebernardi/loss-landscapes/blob/master/examples/core-features.ipynb

if loss_landscape_flag:
    start = time.time()

    train_loader = batch_generator (BS, train_set)
    x, y = iter(train_loader).__next__()
    x = torch.stack(x)
    x = torch.unsqueeze(x, 1)
    y = torch.tensor(y).long()
    metric = loss_landscapes.metrics.Loss(criterion, x, y)

    LCP = loss_landscapes.random_plane(net, metric, 100, STEPS, normalization='filter', deepcopy_model=True)
    
    if CUDA_flag:
        torch.cuda.synchronize()
        
    end = time.time()
    print('\nTime of calculating loss landscape: {:d} s'.format(round((end - start))))

    loss_con = plt.figure()
    plt.contour(LCP, levels=50)
    plt.title('Loss Contours around Trained Model')
    plt.show()

    loss_surf_1 = fig = plt.figure(figsize=(9,7))
    ax = plt.axes(projection='3d')

    X = np.array([[j for j in range(STEPS)] for i in range(STEPS)])
    Y = np.array([[i for _ in range(STEPS)] for i in range(STEPS)])
    ax.plot_surface(X, Y, LCP, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
    ax.set_title('Surface Plot of Loss Landscape')
    ax.set_xlabel(r'$\theta$', fontsize=18, labelpad=10)
    ax.set_ylabel(r"$\theta '$", fontsize=18, labelpad=10)
    ax.set_zlabel('Loss', fontsize=18, labelpad=10, rotation=90)

    loss_surf_2 = fig = plt.figure(figsize=(9,7))
    ax = plt.axes(projection='3d')

    ax.plot_surface(X, Y, LCP, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
    #ax.set_title('Surface Plot of Loss Landscape')
    ax.view_init(30, 45)
    ax.set_xlabel(r'$\theta$', fontsize=18, labelpad=10)
    ax.set_ylabel(r"$\theta '$", fontsize=18, labelpad=10)
    ax.set_zlabel('Loss', fontsize=18, labelpad=10, rotation=90)

# %% ----- PCA ----------------------------------------------------------------------------------------------------------
  
if PCA_flag:
    
    PCA_time_start = time.time()  
    StateVecArray = np.array(StateVecList)
    TrainedNetVector = StateVecArray[epochs-1]
    
    pca = PCA(n_components=2)
    PC = pca.fit_transform(StateVecArray)
    PC_norm = []
    
    for i in range(len(PC[0])):
        col = PC[:,i]
        col_fixed = col / np.linalg.norm(col)
        PC_norm.append(col_fixed)
  
    SVA_1 = []
    for i in range(len(StateVecArray[0])):
        weightvec = StateVecArray[:,i]
        SVA_1.append(np.dot(weightvec, PC_norm[0]))
        
    SVA_2 = []
    for i in range(len(StateVecArray[0])):
        weightvec = StateVecArray[:,i]
        SVA_2.append(np.dot(weightvec, PC_norm[1]))
        
    SVA_1 = np.array(SVA_1)
    SVA_2 = np.array(SVA_2)
        
    if filter_normalization_flag:
        
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
            
        Seg0 = TrainedNetVector[0:Sizes_C[0]]
        Seg0_Norm = np.linalg.norm(Seg0)
        
        Seg1 = SVA_1[0:Sizes_C[0]]
        Seg1_Norm = np.linalg.norm(Seg1)
        
        Seg2 = SVA_2[0:Sizes_C[0]]
        Seg2_Norm = np.linalg.norm(Seg2)
        
        SVA_1[0:Sizes_C[0]] = SVA_1[0:Sizes_C[0]] * Seg0_Norm / Seg1_Norm
        SVA_2[0:Sizes_C[0]] = SVA_2[0:Sizes_C[0]] * Seg0_Norm / Seg2_Norm
        
        TestNorm0 = np.linalg.norm(TrainedNetVector[0:Sizes_C[0]])
        TestNorm1 = np.linalg.norm(SVA_1[0:Sizes_C[0]])
        TestNorm2 = np.linalg.norm(SVA_2[0:Sizes_C[0]])
        
        print('\nTestNorm of first filter from core: \t\t{}'.format(TestNorm0))
        print('TestNorm of first filter from PCA vector 1: \t{}'.format(TestNorm1))
        print('TestNorm of first filter from PCA vector 2: \t{}'.format(TestNorm2))
        
        for x in range(len(Sizes_C) - 1):
            
            Seg0 = TrainedNetVector[Sizes_C[x]:Sizes_C[x+1]]
            Seg0_Norm = np.linalg.norm(Seg0)
            
            Seg1 = SVA_1[Sizes_C[x]:Sizes_C[x+1]]
            Seg1_Norm = np.linalg.norm(Seg1)
            
            Seg2 = SVA_2[Sizes_C[x]:Sizes_C[x+1]]
            Seg2_Norm = np.linalg.norm(Seg2)
            
            SVA_1[Sizes_C[x]:Sizes_C[x+1]] = SVA_1[Sizes_C[x]:Sizes_C[x+1]] * Seg0_Norm / Seg1_Norm
            SVA_2[Sizes_C[x]:Sizes_C[x+1]] = SVA_2[Sizes_C[x]:Sizes_C[x+1]] * Seg0_Norm / Seg2_Norm
    
    
    RIstep = int(round(FilterSteps/2))
    
    loss_array = np.zeros([FilterSteps+1,FilterSteps+1])
      
    
    for i in range(-RIstep, RIstep+1):
        for j in range(-RIstep, RIstep+1):
            NetAdd_1 = distance_multiplier*(i/RIstep)*SVA_1
            NetAdd_2 = distance_multiplier*(j/RIstep)*SVA_2
            NetWeights = TrainedNetVector + NetAdd_1 + NetAdd_2
        
            net_updated = list_to_weights(NetWeights, net, True)  
            
            if CUDA_flag:
                net_updated.cuda()
            
            val_running_loss = 0.0
            val_loader = batch_generator (BS, val_set)
            
            with torch.no_grad():
                for val_inputs, val_labels in val_loader:
            
                    val_inputs = torch.stack(val_inputs)
                    val_inputs = torch.unsqueeze(val_inputs, 1)
                    val_labels = string_tensor(val_labels)
                    val_outputs = net_updated(val_inputs)
                    val_loss = criterion(val_outputs, val_labels.long())
                
                    val_running_loss += val_loss.item()
                
            loss_array[i+RIstep, j+RIstep] = val_running_loss
            
            
    x = np.arange(-RIstep+1, RIstep+1)
    y = np.arange(-RIstep, RIstep+1)
    X,Y = np.meshgrid(x,y)
    
    loss_array_cor = loss_array[:,1:]
    
    if bowl:
        
        tv0 = loss_array[round(FilterSteps/2) - 1, FilterSteps - 1]
        tv1 = loss_array[round(FilterSteps/2) - 1, 0]
        tv2 = loss_array[0, round(FilterSteps/2) - 1]
        tv3 = loss_array[FilterSteps - 1, round(FilterSteps/2) - 1]
        
        bowlmax = max([tv0, tv1, tv2, tv3])
        bowlmin = min([tv0, tv1, tv2, tv3])
        
        for vi in range(FilterSteps):
            for vj in range(FilterSteps):
                if loss_array[vi,vj] > bowlmin:
                    loss_array[vi,vj] = bowlmin
                
    
    PCA_fig_1 = plt.figure(figsize=(10,7))
    ax = PCA_fig_1.gca(projection='3d')
    PCA_surf = ax.plot_surface(X,Y, loss_array_cor, cmap='coolwarm')
    ax.set_xlabel(r'X', fontsize=20)
    ax.set_ylabel(r'Y', fontsize=20)
    ax.set_zlabel(r'Z', fontsize=20)
    
    PCA_fig_2 = plt.figure(figsize=(10,7))
    ax = PCA_fig_2.gca(projection='3d')
    PCA_surf_2 = ax.plot_surface(X,Y, loss_array_cor, cmap='coolwarm')
    ax.set_xlabel(r'X', fontsize=20)
    ax.set_ylabel(r'Y', fontsize=20)
    ax.set_zlabel(r'Z', fontsize=20)
    ax.view_init(30, 45)
    
    PCA_fig_3 = plt.figure(figsize=(10,7))
    PCA_cont = plt.contour(X,Y, loss_array_cor, 100)
    ax = PCA_fig_3.gca()
    ax.set_xlabel(r'X', fontsize=20)
    ax.set_ylabel(r'Y', fontsize=20)
    if contour_center:
        hlinex = [-19,20]
        hliney = [0,0]
        plt.plot(hlinex, hliney,'k')
        vlinex = [0,0]
        vliney = [-20,20]
        plt.plot(vlinex, vliney,'k')
    
    
        loss_array_cor[0][0] = loss_array_cor[0][1]
        LMIN = np.min(loss_array_cor)
        LMINLOC = np.argmin(loss_array_cor)
        RowLen = len(loss_array_cor[0])
        min_remain = LMINLOC % RowLen
        min_y = (LMINLOC - min_remain) / RowLen
        min_x = min_remain
        plt.plot(min_x - 20, min_y - 20, 'rx', markersize=15)
    
        
    PCA_time_end = time.time()
    PCA_time = PCA_time_end - PCA_time_start
    print('\nPCA time: {} s'.format(round(PCA_time,1)))
    
        
# %% ----- Saving results -----------------------------------------------------------------------------------------------


    
os.chdir(fname)

if MAV_flag:
    mavpath1 = '{}\\MAV.mat'.format(fname)
    mavpath2 = '{}\\MAV_dist_cor.mat'.format(fname)
    mavpath3 = '{}\\MAV_dist_inc.mat'.format(fname)
    sio.savemat(mavpath1, {'data':MAV})
    sio.savemat(mavpath2, {'data':AV_dist_correct})
    sio.savemat(mavpath3, {'data':AV_dist_wrong})
    
    f = open('AV stats.txt','w+')
    f.write('Correct predictions:')
    f.write('\n\nAV_max = {:.2f}'.format(AVC_max))
    f.write('\nAV_min = {:.2f}'.format(AVC_min))
    f.write('\nAV_avg = {:.2f}'.format(AVC_avg))
    f.write('\nAV_std = {:.2f}'.format(AVC_std))
    
    f.write('\n\nWrong predictions:')
    f.write('\n\nAV_max = {:.2f}'.format(AVW_max))
    f.write('\nAV_min = {:.2f}'.format(AVW_min))
    f.write('\nAV_avg = {:.2f}'.format(AVW_avg))
    f.write('\nAV_std = {:.2f}'.format(AVW_std))
    f.close()

f = open('run info.txt','w+')
A = str(net)
f.write('-----------------------------------------------------------------------------------------\n\n')
f.write(A)
f.write('\n\n-----------------------------------------------------------------------------------------')
f.write('\n\nData used:\n')

for i in range(len(N)):
    f.write('\nClass {}: {} images'.format(names[i], N[i]))

f.write('\n\nTrain set size: {} images'.format(len(train_set[0])))
f.write('\nVal set length: {} images'.format(len(val_set[0])))
f.write('\nTest set size: \t{} images'.format(len(test_set[0])))
f.write('\n\nTotal size of dataset: {} images'.format(sum(N)))

f.write('\n\n-----------------------------------------------------------------------------------------')

f.write('\n\ninput_size = {}'.format(input_size))
f.write('\nBS = {}'.format(BS))
f.write('\nepochs = {}'.format(epochs))
f.write('\nSTEPS = {}'.format(STEPS))
f.write('\nSeedVal = {}'.format(SeedVal))

f.write('\n\nLoss function: {}'.format(criterion))
f.write('\n\nOptimizer: \n{}'.format(optimizer))    

f.write('\n\n-----------------------------------------------------------------------------------------')    
        
f.write('\n\nAccuracy of the network on the test images: {:.1f} %'.format((100 * correct[0] / total[0])))
f.write('\nTest result obtained from {} images coming from {} batches'.format(total[0], testsize[0]))
f.write('\n\nAccuracy of the network on the total set of all images: {:.1f} %'.format((100 * correct_tot / total_tot)))
f.write('\nTest result obtained from {} images coming from {} batches'.format(total_tot, testsize_tot))

if MAV_flag:
    f.write('\n\nAmount of images in high confidence set: {}'.format(ImagesChecked))
    f.write('\nThis is {:.1f}% of the test set'.format(100*ImagesChecked / total[0]))
    
    f.write('\n\nAmount of correctly classified images in high confidence set: {}'.format(ImagesCheckedCorrect))
    f.write('\nAccuracy on high confidence set: {:.1f} %'.format(100*ImagesCheckedCorrect / ImagesChecked))

if PCA_flag:
    f.write('\n\n-----------------------------------------------------------------------------------------')  
    f.write('\n\nPCA - filter normalization: {}'.format(filter_normalization_flag))
    f.write('\nPCA - distance multiplier: {}'.format(distance_multiplier))
    f.write('\nPCA - FilterSteps: {}'.format(FilterSteps))

if misclassified_outputs_flag:
    f.write('\n\n-----------------------------------------------------------------------------------------') 
    f.write('\n\nMisclassified net outputs: ')
    f.write('\n')
    maxdif_t2 = [round(x,1) for x in maxdif_t]
    for x in range(len(misc_outputs_t)):
        f.write('\n{} \t- maxdif = {}'.format(misc_outputs_t[x], maxdif_t2[x]))
f.write('\n\n-----------------------------------------------------------------------------------------')

f.close()

cwd = os.getcwd()

torch.save({'model_state_dict': net.state_dict()
            }, r'{}\\final_weights.pth'.format(cwd))
    
torch.save({'model_state_dict': model_initial.state_dict()
            }, r'{}\\initial_weights.pth'.format(cwd))
    
path  = '{}\\loss_curve.png'.format(cwd)    
val_plot.savefig(path, dpi=fig_dpi)
path  = '{}\\accuracy_curve.png'.format(cwd)    
acc_plot.savefig(path, dpi=fig_dpi)
path  = '{}\\confusion_matrix.png'.format(cwd)    
conf_fig.savefig(path, dpi=fig_dpi)
path  = '{}\\confusion_matrix_normalize_row.png'.format(cwd)    
conf_fig_2.savefig(path, dpi=fig_dpi)
path  = '{}\\confusion_matrix_normalize_full.png'.format(cwd)    
conf_fig_3.savefig(path, dpi=fig_dpi)

if PCA_flag:
    path  = '{}\\PCA_loss_landscape_1.png'.format(cwd)
    PCA_fig_1.savefig(path, dpi=fig_dpi)
    path  = '{}\\PCA_loss_landscape_2.png'.format(cwd)
    PCA_fig_2.savefig(path, dpi=fig_dpi)
    path  = '{}\\PCA_loss_contour.png'.format(cwd)
    PCA_fig_3.savefig(path, dpi=fig_dpi)

if loss_landscape_flag:
    path  = '{}\\loss_surface_contour.png'.format(cwd)    
    loss_con.savefig(path, dpi=fig_dpi)
    path  = '{}\\loss_surface_angle_1.png'.format(cwd)    
    loss_surf_1.savefig(path, dpi=fig_dpi)
    path  = '{}\\loss_surface_angle_2.png'.format(cwd)    
    loss_surf_2.savefig(path, dpi=fig_dpi)
    
if MAV_flag:    
    path = '{}\\MAV_distance_dist.png'.format(cwd)
    MAV_dist_plot.savefig(path, dpi=fig_dpi)

if misclassified_images_flag:
    mcfname = 'misclassified images'
    os.mkdir(mcfname)
    mcfdir = '{}\\{}'.format(fname, mcfname)
    os.chdir(mcfdir)

    for i in range(len(misc_preds_t)):
    
        imagets0 = misc_img_t[i][0]
        imagets = imagets0.cpu()
        predts = int(misc_preds_t[i])
        predname = names[predts]
        labts = int(misc_labels_t[i])
        labname = names[labts]
        fstag = '{} - label {}, classified as {}'.format(i, labname, predname)
    
        sf = plt.figure(figsize=(6,6))
        plt.imshow(imagets, cmap='jet')
        path = '{}\\{}.png'.format(mcfdir, fstag)
        sf.savefig(path)
        
if baseline_flag:
    path = '{}\\baseline_uncertainty_plot.png'.format(cwd)
    baseline_plot.savefig(path, dpi=fig_dpi)
    
    sio.savemat('unc_correct.mat', {'data':unc_correct})
    sio.savemat('unc_wrong.mat', {'data':unc_wrong})
        
    
os.chdir(DT)
#os.startfile(fname)
        




