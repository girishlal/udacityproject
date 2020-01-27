#!/usr/bin/env python
# coding: utf-8
# */Aaipnd-project-master/train_commonfns.py
#                                                                             
# PROGRAMMER: Girish Lal Pudieduth  
# DATE CREATED:    01/01/2020                               
# REVISED DATE: 
# PURPOSE: common support needed for train program
#     AND
#    Common functions. The functions are described later in this file
##

# Imports python modules to setup command line parameters
import argparse
import os.path

# for the plotting

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# This is to manage the actual lable names of the flowers
import json
import urllib


# Define get_train_input_args function to return with parser.parse_args() parsed argument 
#       collection that you created with this function
# 
# 

def get_train_input_args():
    """
    Retrieves and parses the 7 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to create and define these 7 command line arguments. If 
    the user fails to provide some or all of the needed arguments, then the default 
    values are used for the missing arguments. 
    
    Command Line Arguments:
      1. Training Image Folder as --data_dir with default value 'flowers'
      2. CNN Model Architecture as --arch with default value 'vgg16'
      3. Check point Save Directory as  --save_dir  with default value null and means current folder
      4. Learning Rate as  --learning_rate  with default value 0.001
      5. epoch as  --epoch  with default value 1
      6. If to use GPU as --gpu. If proided, True. Default is False ( means cpu)
      7. Hidden Units as -hidden_unit if nulll use [1000, 500]. 
      
      
      
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object
     
     Train a new network on a data set with train.py

Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains
Options:
Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
Choose architecture: python train.py data_dir --arch "vgg13"
Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
Use GPU for training: python train.py data_dir --gpu

Example 
train.py --data_dir flowers --arch vgg16 --learning_rate 0.001 --gpu cuda

    """
    # Create Parse using ArgumentParser
    chImagesParser = argparse.ArgumentParser()

    chImagesParser.add_argument('data_dir', type = str, default = 'flowers', help = 'Path to the folder of flower images') 
    chImagesParser.add_argument('--arch', type = str, default = 'vgg16', help = 'CNN Model Architecture') 
    chImagesParser.add_argument('--save_dir', type = str, default = '.', help = 'The Checkpoint file folder to save the model') 
    chImagesParser.add_argument('--learning_rate', type = float, default = 0.001, help = 'The learning rate to be used for training the model') 
    chImagesParser.add_argument('--epoch', type = int, default = 5, help = 'The number of epocs to use for training') 
    chImagesParser.add_argument('--gpu', type = str, default = 'cpu', help = 'If to use CUDA or not. If not provided then use cpu. Even if GPU is given, if the system does not have a GPU, then cpu is used ') 
    chImagesParser.add_argument('--hidden_units', type = str, default = '1000,250', help = 'Provide hidden units for each hidden layer') 
    
    

    return chImagesParser.parse_args()


def get_predict_input_args():
    """
Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint
Options:
Return top KK most likely classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference: python predict.py input checkpoint --gpu

    """
    # Create Parse using ArgumentParser
    chImagesParser = argparse.ArgumentParser()

    chImagesParser.add_argument('flowerfile', type = str, default = 'flowerssmall/valid/1/image_06764.jpg', help = '/path/to/image to predict its name') 
    chImagesParser.add_argument('checkpoint', type = str, default = 'project2_gpud_vgg16.pth', help = 'The filename of the checkpoint - saved model') 
    chImagesParser.add_argument('--top_k', type = int, default = 3, help = 'the number of top KK most likely classes for the image') 
    chImagesParser.add_argument('----category_names', type = str, default = 'cat_to_name.json', help = 'the real name file for the classes of flowers') 
    chImagesParser.add_argument('--gpu', type = str, default = 'cpu', help = 'If to use CUDA or not. If not provided then use cpu. Even if GPU is given, if the system does not have a GPU, then cpu is used ') 
    
    

    return chImagesParser.parse_args()

def check_predict_command_line_arguments(in_arg) :
    """
    Check if proper command lines are provided. If not, proper defaults are set.
    See documentation on function "get_predict_input_args" for the details of expected command lines
    """
    
    if in_arg is None:
        print("* Doesn't Check the Command Line Arguments because 'get_predict_input_args' hasn't been defined.")
        return False
    else:
        
        if(os.path.exists(in_arg.flowerfile)==False):
            print("Command Line Arguments:\n     Error!!! - The flower file to predict  ", in_arg.flowername, 
              "does not exsists")
            return False
        if(os.path.exists( in_arg.checkpoint)==False):
            print("Command Line Arguments:\n     Error!!! - The model checkpoint file does not exists  ", in_arg.checkpoint, 
              "does not exsists")
            return False
        
        if(os.path.exists( in_arg.category_names)==False):
            print("Command Line Arguments:\n     Error!!! - The category realname file does not exists  ", in_arg.category_names, 
              "does not exsists")
            return False
        # prints command line agrs
        print("\n Command Line Arguments: Flower to predict file =", in_arg.flowerfile, 
              "\n    Checkpoint file =", in_arg.checkpoint, "\n top KK =", in_arg.top_k,
              "\n    category real name file = ", in_arg.category_names,
             "\n    gpu = ", in_arg.gpu)
        return True

def check_command_line_arguments(in_arg, archsupported) :
    """
    Check if proper command lines are provided for predict process. If not, proper defaults are set.
    See documentation on function "get_train_input_args" for the details of expected command lines
    """
    print(type(archsupported))
    
    if in_arg is None:
        print("* Doesn't Check the Command Line Arguments because 'get_input_args' hasn't been defined.")
        return False
    else:
        # prints command line agrs
        print("\n Command Line Arguments: data dir =", in_arg.data_dir, 
              "\n    arch =", in_arg.arch, "\n save dir =", in_arg.save_dir,
              "\n    learning rate = ", in_arg.learning_rate,
             "\n    epoc = ", in_arg.epoch,
             "\n    gpu = ", in_arg.gpu)
        if(os.path.exists(in_arg.data_dir)==False):
            print("Command Line Arguments:\n     Error!!! - the training, test, and validation folders ", in_arg.data_dir, 
              "does not exsists")
            return False
        if((in_arg.arch in archsupported) == False):
            print("Command Line Arguments:\n     Error!!! : Network Architecture  ", in_arg.arch, 
              "Not supported")
            return False  
        if (in_arg.hidden_units == None):
            print("Command Line Arguments:\n     Error!!! : Network Hidden Layer  ", in_arg.hiddenlayers, 
              "Not supported. It should be like 1000,250")
            return False    
        try:
            print ("Hidden Units :", in_arg.hidden_units)
            
            values = [int(i) for i in in_arg.hidden_units.split(',')]
            print ("Hidden Units Array:", values)
            
        except  Exception as defect:
            print("Command Line Arguments:\n     Error!!! : Network Hidden Layer Not supported. It should be like 1000,250")
            return False    
            
        return True


    
def imshow(image, ax=None, title=None, normalize=True):
    matplotlib.use('agg')
    """The image viewer given a Image Tensor"""
    
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension

    image = image.numpy().transpose((1, 2, 0))
    
    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    
        image = np.clip(image, 0, 1)
    
    ax.set_title(title)
    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax

def TestDataloaders(imgdataloader, title=None):
    matplotlib.use('agg')
    #check some examples  of test, train and validation data 
    
    data_iter = iter(imgdataloader)

    images, labels = next(data_iter)
    print (images.shape, labels.shape)
    fig, axes = plt.subplots(figsize=(10,4), ncols=4)
    for ii in range(4):
        ax = axes[ii]
        imshow(images[ii], ax=ax, normalize=True)
        
def plotModelMetric(objNModule):
    # to print and show the training and test losses and accuracy
    
    print ("Accuracy : ", objNModule.accuracy)
    plt.plot(objNModule.train_losses, label='Training loss')
    plt.plot(objNModule.test_losses, label='Validation loss')
    plt.legend(frameon=False)

