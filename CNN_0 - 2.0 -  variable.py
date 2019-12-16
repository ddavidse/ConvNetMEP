# %% Importing libraries

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io

import time
import os
import random
import itertools
import copy

import loss_landscapes
import loss_landscapes.metrics

from mpl_toolkits.mplot3d import axes3d, Axes3D # ignore warning, we need this 
from scipy.special import softmax

# %%  ----- CUDA initialization -----

torch.set_default_tensor_type('torch.cuda.FloatTensor')


# %%  ----- Set Parameters and Folders-----

# Desktop (place where result folders are created):
DT = r'D:\ddavidse\Desktop'

# Location of training data:
#dirName = r'D:\ddavidse\Desktop\150x450 -- 150x150 - 4 class + background'
dirName = r'D:\ddavidse\Desktop\150x450 -- 150x150 - 4 class (reduced) + background'
#dirName = r'D:\ddavidse\Desktop\3 class cuts\particle cuts 300x900 new collected-- square cuts'
#dirName = r'D:\ddavidse\Desktop\3 class cuts\300x900 square cuts 150x150'
#dirName = r'D:\ddavidse\Desktop\300x900 2nd acq collected -- 300x300'

BS = 20             # batch size
epochs = 15         # number of epochs for training
STEPS = 40          # loss landscape steps

random_initialization = True
saved_weights = r'D:\ddavidse\Desktop\Network_runs\Network_run_6\initial_weights.pth'

set_random_seed = False
seed_value = 294634

# %% ----- Select features -----

loss_landscape_flag = False             # make True to create a loss landscape
misclassified_images_flag = False       # make True to output misclassified images
misclassified_outputs_flag = True       # make True to output misclassified net output values to run info.txt
                

# %%----- detecting number of images per class -----

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
    

# %% detecting file size

os.chdir(names[0])
filelist = os.listdir()
testfile = filelist[0]

testimage = scipy.io.loadmat(testfile)
testimage2 = list(testimage.values())
testimage3 = testimage2[3]

input_size = len(testimage3)

# %%----- Obtaining and storing data -----------------------------------------------------------------------------------------------

# Getting the list of Data

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
    
    # create a list of files and sub directories names in the given directory 
    listOfFiles = os.listdir(dirName)
    allFiles = list()
    
    # Iterate over all the entries
    # Create full path
    # If entry is a directory then get the list of files in this directory 
    
    for entry in listOfFiles:
        
        fullPath = os.path.join(dirName, entry)        
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)             
    return allFiles


# Training data
listOfFiles = getListOfFiles(dirName)
listOfImages = files_tensor(listOfFiles)

# Labels 
# to use the criterion CrossEntropyLoss the classes must be from 0 to n
label = np.zeros(sum(N))
labels = [x for x in range(len(names))]

for i in range(0,NC[0]):
    label[i] = labels[0]
    
for k in range(1,len(N)):
    for i in range(NC[k-1],NC[k]):
        label[i] = labels[k]
    
shuffle_list = list(zip(listOfImages, label))

# %% code to keep distribution fixed

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

for k in range(len(N)):
    zip_train[k] = zipZ[k][0:int(3*N[k]/5)]
    zip_val[k] = zipZ[k][int(3*N[k]/5):int(4*N[k]/5)]
    zip_test[k] = zipZ[k][int(4*N[k]/5):N[k]]
    
train = [x for s in zip_train for x in s]
val = [x for s in zip_val for x in s]
test = [x for s in zip_test for x in s]

files1, labels1 = zip(*test)
test_set = [files1, labels1]

files2, labels2 = zip(*train)
train_set = [files2, labels2]

files3, labels3 = zip(*val)
val_set = [files3, labels3]

# %% adjusting sets to fit in batches

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
    

# %% batch generator and loaders

def batch_generator(batch_size, dataset): # return batches of batch_size from the listOfFiles samples list
        
    batch = [[],[]]
    counter = 0
    
    shuffle_list = list(zip(dataset[0], dataset[1]))
    random.shuffle(shuffle_list) # shuffle the list of samples each time the function batch_generator function is called
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
        
train_loader = batch_generator (BS, train_set)
test_loader = batch_generator (BS, test_set)
val_loader = batch_generator (BS, val_set)


# %%----- Utility functions ---------------------------------------------------------------------------------------------------------

# Function converting list of strings into tensor
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

# %%----- Soft coding input size ----------------------------------------------------------------------------------------------
    
if input_size % 2 == 0:
    Size_1 = input_size / 2
else:
    Size_1 = (input_size - 1) / 2
    
if Size_1 % 2 == 0:
    Size_2 = Size_1 / 2
else:
    Size_2 = (Size_1 - 1) / 2
    
Size_2 = int(Size_2)


# %%----- Defining the neural net ---------------------------------------------------------------------------------------------------

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__() 
        #note: nn.Conv2D(in, out, kernel, stride=1, padding)
        #note: nn.MaxPool2d(kernel, stride, padding)

        self.features = nn.Sequential(
                                      nn.Conv2d(1,5,5,1,2),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2,2),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(5,8,5,1,2),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2,2),
                                      nn.ReLU(inplace=True),                                                                        
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
    

# %% ----- Initialize, save and load weights ------------------------------------------------------------------------------

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.uniform_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.uniform_(m.bias)


net = Net()
net.cuda()

if random_initialization:
    net.apply(weights_init)
else:
    checkpoint = torch.load(saved_weights)
    net.load_state_dict(checkpoint['model_state_dict'])


sname = '{}\\model_parameters'.format(DT)
if not os.path.exists(sname):
        os.mkdir(sname)
nsname = '{}\\current_net.pth'.format(sname)

torch.save({            
            'model_state_dict': net.state_dict(),
            #'optimizer_state_dict': optimizer.state_dict()
            }, nsname)

      
# %% Loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
#optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
#optimizer = optim.Adadelta(net.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=1e-5)
#optimizer = optim.RMSprop(net.parameters(), lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False)

model_initial = copy.deepcopy(net)

# %%----- Training the net ---------------------------------------------------------------------------------------------------------

#net = net.double()
start = time.time()

running_loss_history = []
running_corrects_history = []
val_running_loss_history = []
val_running_corrects_history = []

for e in range(epochs):
    
    print('epoch :', (e+1))  
    running_loss = 0.0
    running_corrects = 0.0
    val_running_loss = 0.0
    val_running_corrects = 0.0
    #scheduler.step()
    train_loader = batch_generator(BS, train_set)
    val_loader = batch_generator(BS, val_set)
    
    train_amount = 0.0
    val_amount = 0.0
      
    for inputs, labels in train_loader:
        
        labels = string_tensor(labels)
        inputs = torch.stack(inputs)
        inputs = torch.unsqueeze(inputs, 1)
        #inputs = inputs.float()
        outputs = net(inputs)
        loss = criterion(outputs, labels.long())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.long().data)
        
        train_amount += BS

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
     
    print('training loss: {:.4f} '.format(epoch_loss))
    print('validation loss: {:.4f}'.format(val_epoch_loss))
    print(' ')


torch.cuda.synchronize()
end = time.time()
print('Time of training: {:d} s'.format(round((end - start))))

#model_final = copy.deepcopy(net)


# %%----- Prediction and training stats and graphs -------------------------------------------------------

correct = 0
total = 0
testsize = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = torch.stack(images)
        images = torch.unsqueeze(images, 1)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += string_tensor(labels).size(0)
        testsize += 1
        correct += (predicted == string_tensor(labels).long()).sum().item()

print(' ')
print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
print('Test result obtained from {} images coming from {} batches'.format(total, testsize))

#get_ipython().run_line_magic('matplotlib', 'inline')

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

# %% ----------- confusion matrix -------------------------------------------------------------
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

stacked = torch.stack((AL3, TP3), dim=1)

cmt = torch.zeros(len(N),len(N), dtype=torch.int64)


for p in stacked:
    tl, pl = p.tolist()
    cmt[tl, pl] = cmt[tl, pl] + 1
    

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
                
    print('\n',cm)
      
    plt.imshow(cm, interpolation='none', cmap=cmap)
    
    plt.title(title)
    plt.colorbar()
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

cmt2 = cmt.cpu()
conf_fig_size = (10,10)

conf_fig = plt.figure(figsize = conf_fig_size)
plot_confusion_matrix(cmt2, names)

conf_fig_2 = plt.figure(figsize = conf_fig_size)
plot_confusion_matrix(cmt2, names, normalize='row', percentage=True)

conf_fig_3 = plt.figure(figsize = conf_fig_size)
plot_confusion_matrix(cmt2, names, normalize='full', percentage=True)



# %% loss landscape
# source: https://github.com/marcellodebernardi/loss-landscapes/blob/master/examples/core-features.ipynb

if loss_landscape_flag:
    start = time.time()

    train_loader = batch_generator (BS, train_set)
    x, y = iter(train_loader).__next__()
    x = torch.stack(x)
    x = torch.unsqueeze(x, 1)
    y = torch.tensor(y).long()
    metric = loss_landscapes.metrics.Loss(criterion, x, y)

    #LCP = loss_landscapes.random_plane(model_final, metric, 10, STEPS, normalization='filter', deepcopy_model=True)
    LCP = loss_landscapes.random_plane(net, metric, 100, STEPS, normalization='filter', deepcopy_model=True)
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
    ax.plot_surface(X, Y, LCP, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('Surface Plot of Loss Landscape')
    ax.set_xlabel(r'[$\theta$]', fontsize=20)
    ax.set_ylabel(r"[$\theta '$]", fontsize=20)
    ax.set_zlabel('Loss', fontsize=20)

    loss_surf_2 = fig = plt.figure(figsize=(9,7))
    ax = plt.axes(projection='3d')

    ax.plot_surface(X, Y, LCP, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('Surface Plot of Loss Landscape')
    ax.view_init(30, 45)
    ax.set_xlabel(r'[$\theta$]', fontsize=20)
    ax.set_ylabel(r"[$\theta '$]", fontsize=20)
    ax.set_zlabel('Loss', fontsize=20)
    

# %% Saving results 

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
    
os.chdir(fname)

f = open('run info.txt','w+')
A = str(net)
f.write(A)
f.write('\n\n-----------------------------------------------------------------------------------------')
f.write('\n\nData used:\n')

for i in range(len(N)):
    f.write('\nClass {}: {} images'.format(names[i], N[i]))

f.write('\n\nTrain set size: {} images'.format(len(train_set[0])))
f.write('\nVal set length: {} images'.format(len(val_set[0])))
f.write('\nTest set size: {} images'.format(len(test_set[0])))

f.write('\n\n-----------------------------------------------------------------------------------------')

f.write('\n\ninput_size = {}'.format(input_size))
f.write('\nBS = {}'.format(BS))
f.write('\nepochs = {}'.format(epochs))
f.write('\nSTEPS = {}'.format(STEPS))
f.write('\nSeedVal = {}'.format(SeedVal))

f.write('\n\nLoss function: {}'.format(criterion))
f.write('\n\nOptimizer: \n{}'.format(optimizer))    

f.write('\n\n-----------------------------------------------------------------------------------------')    
        
f.write('\n\nAccuracy of the network on the test images: %d %%' % (100 * correct / total))
f.write('\nTest result obtained from {} images coming from {} batches'.format(total, testsize))

if misclassified_outputs_flag:
    f.write('\n\nMisclassified net outputs: ')
    f.write('\n')
    maxdif_t2 = [round(x,1) for x in maxdif_t]
    for x in range(len(misc_outputs_t)):
        f.write('\n{} \t- maxdif = {}'.format(misc_outputs_t[x], maxdif_t2[x]))

f.close()

cwd = os.getcwd()

torch.save({'model_state_dict': net.state_dict()
            }, r'{}\\final_weights.pth'.format(cwd))
    
torch.save({'model_state_dict': model_initial.state_dict()
            }, r'{}\\initial_weights.pth'.format(cwd))
    
path  = '{}\\loss_curve.png'.format(cwd)    
val_plot.savefig(path, dpi=300)
path  = '{}\\accuracy_curve.png'.format(cwd)    
acc_plot.savefig(path, dpi=300)
path  = '{}\\confusion_matrix.png'.format(cwd)    
conf_fig.savefig(path, dpi=300)
path  = '{}\\confusion_matrix_normalize_row.png'.format(cwd)    
conf_fig_2.savefig(path, dpi=300)
path  = '{}\\confusion_matrix_normalize_full.png'.format(cwd)    
conf_fig_3.savefig(path, dpi=300)

if loss_landscape_flag:
    path  = '{}\\loss_surface_contour.png'.format(cwd)    
    loss_con.savefig(path, dpi=300)
    path  = '{}\\loss_surface_angle_1.png'.format(cwd)    
    loss_surf_1.savefig(path, dpi=300)
    path  = '{}\\loss_surface_angle_2.png'.format(cwd)    
    loss_surf_2.savefig(path, dpi=300)

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
        
    
os.chdir(DT)
os.startfile(fname)
        

# %% deleting the net

del net



