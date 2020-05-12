# ConvNet
This repository contains Convolutional Neural Network code used for the master thesis project "Application of Deep Learning to Coherent Fourier Scatterometry data"

----------------------------------------------------------------------------------------------------------------------------------

To use the code:

1. download CNN_functions.py and CNN_masterscript.py and put them in the same folder
2. On lines 46 and 49 (and optionally also 52) enter the correct paths
3. Use lines 56-98 to set the desired parameters and features
4. In case of no CUDA: use ctrl F to go through the script and comment out all cuda lines
5. Run the script

For a basic test, make confusion_flag = True and batchnorm_flag = True while leaving the rest as False

----------------------------------------------------------------------------------------------------------------------------------

PCA - trajectory.py is a seperate version of CNN_masterscript that was used for plotting trajectories on loss contour plots that were obtained with PCA

----------------------------------------------------------------------------------------------------------------------------------

The directory OpenMax_code contains the code used on a separate PC (and thus, a separate python installation) for OpenMax calculations
