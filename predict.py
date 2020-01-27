#!/usr/bin/env python
# coding: utf-8
# PROGRAMMER: Girish Lal Pudieduth
# DATE CREATED:    01/01/2020
# REVISED DATE:
# PURPOSE: This is developed as a part of Create Your Own Image Classifier of AI Programming with Python Nanodegree Program
# This module is used to predict names of the flowers given an image with an Image Classifier developed Deep Learning with PyTorch and trainined using train program.
#   Example call:
#    python predict.py "imagename with path" --arch vgg16 --topk 5
# Other options are alos available. use -h to get other parameters


"""
Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint
Options:
Return top KK most likely classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference: python predict.py input checkpoint --gpu

example
python predict.py flowerssmall/valid/1/image_06764.jpg savedir/project2_gpud_vgg16.pth

"""

from train_commonfns import *  # The common functions
import NeuralNetwork  # the class that defines the CNN model and train and validate functions

def main():
    arch_supported ={"vgg16":1, "resnet18":2, "googlenet":3, "vgg19":4, "alexnet":5}  # three architectures are supported via torchvision.

    #get all the necessary inputs to start the program

    in_arg = get_predict_input_args()

    # Function that checks command line arguments using in_arg. if not valid call, the prgram exits
    if (check_predict_command_line_arguments(in_arg)== False):
        return False


    objNModule = NeuralNetwork.NModule(gpu= in_arg.gpu, mode="predict", categoryfile=in_arg.category_names, checkpoint=in_arg.checkpoint, topk=in_arg.top_k, flowerfile=in_arg.flowerfile)

    print ("Created Neural network Module for prediction: ", objNModule)

    if objNModule == None:
        print("No Neural Object was created. Exiting  the code. The previous print statement provides details of failures")
        return False

    if objNModule.loadTheModelfromChkPt() == False:
        print("loadTheModelfromChkPt() failed. Exiting  the code. The previous print statement provides details of failures")
        return False

    print("The Model is ready to be used for prediction " ,objNModule)

    '''
    #with active_session():  # inside Udacity infra we can use this run a long running process

    '''
    probs, classes, classnames,image = objNModule.predict()
    print ("\nProbabilities : ", probs)
    print ("\classes :  ",classes)
    print ("\Class Names :", classnames)



    '''
    if objNModule.predict() == False:
        return False
        print("predict() had some Issues. The program failed. The previous print statement provides details of failures")
    '''

    return True


# To start the program
if __name__ == "__main__":
    main()
