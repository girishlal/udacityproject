#!/usr/bin/env python
# coding: utf-8
# PROGRAMMER: Girish Lal Pudieduth  
# DATE CREATED:    01/01/2020                               
# REVISED DATE: 01/18/2020
# PURPOSE: This is developed as a part of Create Your Own Image Classifier of AI Programming with Python Nanodegree Program
# This module is used to train the flowers database to develop an Image Classifier with Deep Learning with PyTorch.
#   Example call:
#    python train.py data_dir --arch vgg16
# Other options are also available. use -h to get other parameters
'''
usage: train.py [-h] [--arch ARCH] [--save_dir SAVE_DIR]
                [--learning_rate LEARNING_RATE] [--epoch EPOCH] [--gpu GPU]
                data_dir

positional arguments:
  data_dir              Path to the folder of flower images

optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH           CNN Model Architecture
  --save_dir SAVE_DIR   The Checkpoint file folder to save the model
  --learning_rate LEARNING_RATE
                        The learning rate to be used for training the model
  --epoch EPOCH         The number of epocs to use for training
  --gpu GPU             If to use CUDA or not. If not provided then use cpu.
                        Even if GPU is given, if the system does not have a
                        GPU, then cpu is used
'''
            

from train_commonfns import *  # The common functions 
import NeuralNetwork  # the class that defines the CNN model and train, validate and (predict) functions

#from workspace_utils import active_session


def main():
    arch_supported ={"vgg16":1, "resnet18":2, "googlenet":3, "vgg19":4, "alexnet":5}  # three architectures are supported via torchvision.
    #get all the necessary inputs to start the program
    
    in_arg = get_train_input_args()

    # Function that checks command line arguments using in_arg. if not valid call, the prgram exits
    if (check_command_line_arguments(in_arg, arch_supported)== False):
        return False


    # Initialize the CNN model class
    objNModule = NeuralNetwork.NModule(in_arg.data_dir, in_arg.save_dir, in_arg.epoch, in_arg.arch, in_arg.learning_rate,in_arg.gpu, hiddenlayer=in_arg.hidden_units )
    #print ("Created Neural network Module: ", objNModule)
    
    

    if objNModule == None:
        print("No Neural Object was created. Exiting  the code. The previous print statement provides details of failures")
        return False
        
    if objNModule.setupdata() == False:
        print("Setupdata() failed. Exiting  the code. The previous print statement provides details of failures")
        return False
    
   
    '''
    # Series of tests
    print("\nTraining  Examples")
    TestDataloaders(objNModule.trainloader)
    print("\nValidation Examples")
    TestDataloaders(objNModule.validationloader)
    print("\nTest Examples")
    TestDataloaders(objNModule.testloader)
    '''
    # below the model, optimizer, criteria etc are setup. Hyper parameters are setup
    
    if objNModule.initializeModel() == False:
        print("initializeModel() failed. Exiting  the code. The previous print statement provides details of failures")
        return False
    
    print("The Model is ready to be trained " ,objNModule)

   # with active_session():  # inside Udacity infra we can use this run a long running process
    print("The Model Training Starting " ,objNModule)
    
    
    if objNModule.train(print_every=5) == False:
        return False
        print("Train() had some Issues. The program failed. The previous print statement provides details of failures")

    # The train provides validation results as well. We are are plotting it here
    print("The Model Picturing Starting ")
    
    plotModelMetric(objNModule)
    
    # Run validation and print results below
    test_loss, accuracy = objNModule.validation(objNModule.testloader)
    
    print("Test Loss: {:.3f}.. ".format(test_loss/len(objNModule.testloader)),
       "Test Accuracy: {:.3f}".format(accuracy/len(objNModule.testloader)))
    
    # if all are ok, then save the model for later use. 
    objNModule.saveTheModel()
    
    return True

# To start the program
if __name__ == "__main__":
    main()
