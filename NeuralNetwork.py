
# PROGRAMMER: Girish Lal Pudieduth  
# DATE CREATED:    01/01/2020                               
# REVISED DATE: 
# PURPOSE: This is developed as a part of Create Your Own Image Classifier of AI Programming with Python Nanodegree Program
# This module is defines Image Classifier with Deep Learning with PyTorch. We use transfer learning and uses trained modules from #Torchvision.

# The class create, initialize a Image classifier. train() and validate() functions are used for training.


import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms, models

import matplotlib 
matplotlib.use('agg')

import matplotlib.pyplot as plt
from PIL import Image

import numpy as np

import json
import urllib
import os

import collections

from torch.autograd import Variable


class NModule:

    def __init__(self, data_dir="", save_dir="",epoch="",architecture="", learningrate="", gpu="", categoryfile="", checkpoint="", flowerfile="", topk="", hiddenlayer="1000,250", mode="train"):
        torch.nn.Module.dump_patches = True
        if mode=="train":
            self.mode = "train"
            self.data_dir = data_dir
            self.save_dir = save_dir

            #hper parameters
            self.epoch = epoch
            self.architecture = architecture
            self.learningrate = learningrate
            self.gpu = gpu
            self.dropout = 0.5

            # Use GPU if it's availablesma
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("device from Torch is ", str(self.device))

            if  (str(self.device).find("cuda")) and self.gpu.lower() == "cpu":
                self.device="cpu"

            print("device to be used is ", self.device)
            # setup data folders
            self.train_dir = data_dir + '/train'
            self.valid_dir = data_dir + '/valid'
            self.test_dir = data_dir + '/test'


            self.catfilename = "cat_to_name.json"
            self.label = None

            #initialize the metric data structures

            self.train_losses = []
            self.test_losses = []
            self.accuracy = 0

            # to load the trained network

            self.model = None
            self.criterion = None
            self.optimizer = None

            # training , validation and testing data initialized. 

            self.trainloader = None
            self.validationloader = None
            self.testloader = None 
            self.hiddenlayers = [int(i) for i in hiddenlayer.split(',')]
            
        if mode=="predict":
            self.mode = "predict"
            self.gpu = gpu
            self.checkpoint = checkpoint
            self.flowerfile  =    flowerfile
            if categoryfile is None:
                self.catfilename = cat_to_name.json
            else:
                self.catfilename = categoryfile
            self.label = None
            self.topk= topk
  
            #hyper parameters
            self.architecture = architecture
            
            
            # Use GPU if it's availablesma
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            print("device from Torch is ", str(self.device))

            if  (str(self.device).find("cuda")) and self.gpu.lower() == "cpu":
                self.device="cpu"

            print("device to be used is ", self.device)
            
            # to load the trained network

            self.model = None
            self.criterion = None
            self.optimizer = None

            
    def initializeModel(self):
        # use t he model
        num_in_features = 25088
        num_out_features = 102
        modeltype = None
        
        if self.mode=="train":
            print ("Init Model Started for Training")
            try:
                if (self.architecture == "vgg16"):
                    self.model = models.vgg16(pretrained=True)
                    num_in_features =25088
                    modeltype = "class"
                elif (self.architecture == "resnet18"):
                    self.model = models.resnet18(pretrained=True)
                    num_in_features =512
                    modeltype = "fc"
                    print("restnet model: ", self.model) 
                elif (self.architecture == "googlenet"):
                    self.model = models.googlenet(pretrained=True) 
                    num_in_features =1024
                    modeltype = "fc"
                    print("googlenet model: ", self.model)
                elif (self.architecture == "alexnet"):
                    self.model = models.alexnet(pretrained=True) 
                    num_in_features =9216
                    modeltype = "class"
                    print("alextnet model: ", self.model)
                elif (self.architecture == "vgg19"):
                    self.model = models.vgg19(pretrained=True)
                    num_in_features =25088
                    modeltype = "class"
                    print("vgg19 model: ", self.model)
                else:
                    print ("Not supported architecture. Can not proceed : ", self.architecture) 
                    return False    
                
                    
            except Exception as defect:
                print ("An error occured in Initializing the Network model. Can not proceed : ", defect) 
                return False
            self.modeltype = modeltype
            
            try:
                # Freeze parameters so we don't backprop through them
                for param in self.model.parameters():
                    param.requires_grad = False
                    
                layermodule = nn.ModuleList([nn.Linear(in_features=num_in_features, out_features=self.hiddenlayers[0], bias=True), nn.ReLU(inplace=True), nn.Dropout(p=self.dropout,  inplace=False)])
                
                for i in range(0, len(self.hiddenlayers) -1):
                    layermodule = layermodule.extend([nn.Linear(in_features=self.hiddenlayers[i], out_features=self.hiddenlayers[i+1], bias=True), nn.ReLU(inplace=True), nn.Dropout(p=self.dropout,  inplace=False)])
                
                layermodule = layermodule.extend([nn.Linear(in_features=self.hiddenlayers[-1], out_features=num_out_features, bias=True),nn.LogSoftmax(dim=1)])
               
                ''' 
                self.model.classifier = nn.Sequential(nn.Linear(in_features=num_in_features, out_features=self.hiddenlayers[0], bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p=self.dropout,  inplace=False),
                                     nn.Linear(in_features=self.hiddenlayers[0], out_features=self.hiddenlayers[-1], bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p=self.dropout, inplace=False),
                                     nn.Linear(in_features=self.hiddenlayers[-1], out_features=num_out_features, bias=True),
                                     nn.LogSoftmax(dim=1))

                '''
                
                if (self.architecture == "vgg16"):
                    self.model.classifier = nn.Sequential(*layermodule)
                elif (self.architecture == "resnet18"):
                    self.model.fc = nn.Sequential(*layermodule)
                elif (self.architecture == "googlenet"):
                    self.model.fc = nn.Sequential(*layermodule)
                elif (self.architecture == "vgg19"):
                    self.model.classifier = nn.Sequential(*layermodule)
                elif (self.architecture == "alexnet"):
                    self.model.classifier = nn.Sequential(*layermodule)
                else:
                    print ("Architecture can not assigned. Can not proceed : ", self.architecture) 
                    return False    
            except Exception as defect:
                    print ("An error occured in setting archtitecture to the assigned  Network model. Can not proceed : ", defect) 
                    return False
            try:
                    # Only train the classifier parameters, feature parameters are frozen
             
                if modeltype == "class":
                    print("Optimizer set class")
                    self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.learningrate)
                else:
                    print("Optimizer set fc")
                    self.optimizer = optim.Adam(self.model.fc.parameters(), lr=self.learningrate)
            except Exception as defect:
                print ("An error occured in setting Optimizer. Can not proceed : ", defect) 
                return False

            self.criterion = nn.NLLLoss()
            self.model = self.model.to(self.device)
            
            print ("Init Model Training Ended ***********")
        
        return True
    
    def train(self,print_every=40):
    
        steps = 0
        running_loss = 0
        train_losses, test_losses  = [], []
        accuracy = 0

        for e in range(self.epoch):
            # Model in training mode, dropout is on
            self.model.train()
            for images, labels in self.trainloader:
                steps += 1
                images, labels = images.to(self.device), labels.to(self.device)
                #print(images.shape, labels.shape)
                self.optimizer.zero_grad()

                output = self.model.forward(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if steps % print_every == 0:
                    # Model in inference mode, dropout is off
                    self.model.eval()

                    # Turn off gradients for validation, will speed up inference
                    with torch.no_grad():
                        test_loss, accuracy = self.validation(self.testloader)
                        train_losses.append(running_loss/len(self.trainloader))
                        test_losses.append(test_loss/len(self.testloader))
                        self.accuracy  =  accuracy /len(self.testloader)
                    print("Epoch: {}/{}.. ".format(e+1, self.epoch),
                          "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Test Loss: {:.3f}.. ".format(test_loss/len(self.testloader)),
                          "Test Accuracy: {:.3f}".format(self.accuracy))

                    running_loss = 0

                    # Make sure dropout and grads are on for training
                    self.model.train()

        self.train_losses = train_losses
        self.test_losses = test_losses
       # self.accuracy = accuracy

        return True
    
    def validation(self, testloader):
        accuracy = 0
        test_loss = 0
        for images, labels in testloader:
            #images.resize_(images.size()[0], 50176)
            images, labels = images.to(self.device), labels.to(self.device)
            output = self.model.forward(images)
            test_loss += self.criterion(output, labels).item()

            ## Calculating the accuracy 
            # Model's output is log-softmax, take exponential to get the probabilities
            ps = torch.exp(output)
            # Class with highest probability is our predicted class, compare with true label
            equality = (labels.data == ps.max(1)[1])
            #print("eq ",equality) 
            # Accuracy is number of correct predictions divided by all predictions, just take the mean
            accuracy += equality.type_as(torch.FloatTensor()).mean()

        return test_loss, accuracy


    def setupdata(self):
        print ("Setup Data Started")
        try:
            train_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

            validation_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])


            test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
            self.train_data = datasets.ImageFolder(self.train_dir, transform=train_transforms)
            validation_data = datasets.ImageFolder(self.valid_dir, transform=validation_transforms)
            test_data = datasets.ImageFolder(self.test_dir, transform=test_transforms)
        except Exception as defect :
               print ("An error occured in SetupData. Can not proceed : ", defect) 
               return False
# TODO: Using the image datasets and the trainforms, define the dataloaders
        self.trainloader = torch.utils.data.DataLoader(self.train_data, batch_size=64, shuffle=True)
        self.validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64)
        self.testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

        print ("Setup Data Ended")
        if(os.path.exists(self.catfilename)==False):
            print("Category label files does not exists:\n     Error!!! -  ", self.catfilename, "does not exsists")
            return False
        try:
            with open(self.catfilename, 'r') as f:
                self.label = json.load(f)
        except  Exception as defect:
            print ("An error occured in loading label data. Can not proceed : ", defect) 
            return False
    
        return True

    def saveTheModel(self):
        print("SAVE " ,self.model)
        objclassifier = None
        
        if (self.architecture == "vgg16"):
            objclassifier = self.model.classifier
        elif (self.architecture == "resnet18"):
            objclassifier = self.model.fc
        elif (self.architecture == "googlenet"):
            objclassifier = self.model.fc
        elif (self.architecture == "vgg19"):
            objclassifier = self.model.classifier
        elif (self.architecture == "alexnet"):
            objclassifier = self.model.classifier
        else:
            print ("Not supported architecture. Can not proceed : ", self.architecture) 
            return False    
        print ("calssifier ", objclassifier)
       
                
        checkpoint = {
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'trainingloss': self.train_losses,
                'validationloss' : self.test_losses,
                'class_to_idx' : self.train_data.class_to_idx,
                'modelname' : self.architecture,
                'hiddenlayer' :self.hiddenlayers,
                'learingRate': self.learningrate,
                'classifier' : objclassifier,
                'modeltype' : self.modeltype,
                'dropout': self.dropout
                }
                
        modelchkpointfile = self.save_dir + '/project2_gpud_' + self.architecture + ".pth"
        torch.save(checkpoint, modelchkpointfile)   
    
    def loadTheModelfromChkPt(self):
        
        num_in_features = 25088
        num_out_features = 102
        modeltype = None
        
        try:
            checkpoint = torch.load(self.checkpoint)
            self.modelname = checkpoint['modelname']
            self.architecture = self.modelname 
            self.learningrate =checkpoint['learingRate']
            self.hiddenlayers = checkpoint['hiddenlayer']
            self.modeltype = checkpoint['modeltype']
            self.dropout = checkpoint['dropout']
            
            if (self.architecture == "vgg16"):
                self.model = models.vgg16(pretrained=True)
                num_in_features =25088
                modeltype = "class"
            elif (self.architecture == "resnet18"):
                self.model = models.resnet18(pretrained=True)
                num_in_features =512
                modeltype = "fc"
                print("restnet model: ", self.model) 
            elif (self.architecture == "googlenet"):
                self.model = models.googlenet(pretrained=True) 
                num_in_features =1024
                modeltype = "fc"
                print("googlenet model: ", self.model)
            elif (self.architecture == "alexnet"):
                self.model = models.alexnet(pretrained=True) 
                num_in_features =9216
                modeltype = "class"
                print("alextnet model: ", self.model)
            elif (self.architecture == "vgg19"):
                self.model = models.vgg19(pretrained=True)
                num_in_features =25088
                modeltype = "class"
                print("vgg19 model: ", self.model)
            else:
                print ("Not supported architecture. Can not proceed : ", self.architecture) 
                return False    
            self.modeltype = modeltype
            print("hidden layers", self.hiddenlayers)
            
            layermodule = nn.ModuleList([nn.Linear(in_features=num_in_features, out_features=self.hiddenlayers[0], bias=True), nn.ReLU(inplace=True), nn.Dropout(p=self.dropout,  inplace=False)])
                
            for i in range(0, len(self.hiddenlayers) -1):
                layermodule = layermodule.extend([nn.Linear(in_features=self.hiddenlayers[i], out_features=self.hiddenlayers[i+1], bias=True), nn.ReLU(inplace=True), nn.Dropout(p=self.dropout,  inplace=False)])
                
                layermodule = layermodule.extend([nn.Linear(in_features=self.hiddenlayers[-1], out_features=num_out_features, bias=True),nn.LogSoftmax(dim=1)])
            
            try:     
                if (self.architecture == "vgg16"):
                    self.model.classifier = nn.Sequential(*layermodule)
                elif (self.architecture == "resnet18"):
                    self.model.fc = nn.Sequential(*layermodule)
                elif (self.architecture == "googlenet"):
                    self.model.fc = nn.Sequential(*layermodule)
                elif (self.architecture == "vgg19"):
                    self.model.classifier = nn.Sequential(*layermodule)
                elif (self.architecture == "alexnet"):
                    self.model.classifier = nn.Sequential(*layermodule)
                else:
                    print ("Architecture can not assigned. Can not proceed : ", self.architecture) 
                    return False    
            except Exception as defect:
                    print ("An error occured in setting archtitecture to the assigned  Network model. Can not proceed : ", defect) 
                    return False
                
            for param in self.model.parameters():
                param.requires_grad = False

        except  Exception as defect:
            print ("An error occured in loading the model from checkpoint.", self.checkpoint , "Can not proceed : ", defect) 
            return False
        
        
        print ("modelname %s, Model: %s" %(self.modelname, self.model) )
        
        try:
            
            with open(self.catfilename, 'r') as f:
                self.classes = json.load(f)
        except Exception as defect:
            print ("An error occured in loading the class names from Class Meta file. Can not proceed : ", defect) 
            return False
        
        try:
            if self.modeltype == "class":
                print("Optimizer set class")
                self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.learningrate)
            else:
                print("Optimizer set fc")
                self.optimizer = optim.Adam(self.model.fc.parameters(), lr=self.learningrate)
        except  Exception as defect:
            print ("An error occured in loading the classifier from checkpoint. Can not proceed : ", defect) 
            return False
        try:    
            self.model.load_state_dict(checkpoint['model_state_dict'])
            #self.optimizer = optim.Adam(self.model.parameters(), lr=self.learningrate)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        except  Exception as defect:
            print ("An error occured in loading the  state from checkpoint. Can not proceed : ", defect) 
            return False
        
        self.criterion = nn.NLLLoss()

        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['trainingloss']

        self.model.to(self.device)

        return True
    def predict(self):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        self.model=self.model.to(self.device)
        self.model.eval()

        image = self.process_image(self.flowerfile)
        
        print(image.shape)
        flat_size = 3*224*224
        X = torch.randn(64, flat_size)
        imageobj = X.view(-1, *image.shape)  
        imageobj = imageobj.to(self.device)

        with torch.no_grad():
            output = self.model.forward(imageobj)
            ps = torch.exp(output)
            topItems = ps[0].topk(self.topk,sorted=True)
            probs = topItems[0]
            classes =topItems[1]
            classnames = self.getclassnames(classes.cpu())
        return probs, classes, classnames,image
    def getclassnames(self,classes):
        classnames ={}
        numclasses = classes.numpy()
        for i in range(numclasses.size):
            classnames[numclasses[i]] = self.classes.get(str(numclasses[i]))
        return classnames
    
    def process_image(self,image_path ):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        size = 256, 256
        print(image_path)
        #if url==True:
        #    imageobj = Image.open(urllib.request.urlopen(image_path))
        #else:
        
        imageobj = Image.open(image_path)

        Image_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])



        image =  Image_transforms(imageobj) 
        return image


    def __str__(self):
        if self.mode=="train":
            objectprint ="The NmModule is: \n data_dir: %s \n save_dir : %s \n epoch :  %s \n architecture : %s \n learning rate : %s \n device: %s"
            objectprint = objectprint + "\n training folder : %s, \n Validation folder : %s, \n Testing folder : %s"

            objectprint = objectprint + "\n Label File name : %s, \n Training Loss : %s, \n Test Losses : %s , \n Accuracy %s"

            objectprint = objectprint + "\n Model : %s, \n Criterion : %s, \n Optimizer : %s \n Mode : %s , \n Hidden layers : %s"

            return   objectprint % (self.data_dir, self.save_dir, self.epoch, self.architecture, self.learningrate, self.device, self.train_dir, self.valid_dir, self.test_dir, self.catfilename, self.train_losses, self.test_losses, self.accuracy, self.model, self.criterion,self.optimizer, self.mode, self.hiddenlayers)
        if self.mode=="predict":
            objectprint ="The NmModule is: \n architecture : %s \n device : %s \n category file : %s"
            objectprint = objectprint + "\n model : %s, \n criterion : %s, \n optimizer : %s"

            objectprint = objectprint + "\n GPU : %s, \n Check point file : %s, \n Category File : %s , \n Top KK %s"

            objectprint = objectprint + "\n Flower Name file : %s"

            return   objectprint % (self.architecture, self.device, self.catfilename,self.model, self.criterion,self.optimizer, self.gpu,self.checkpoint,self.catfilename, self.topk, self.flowerfile)