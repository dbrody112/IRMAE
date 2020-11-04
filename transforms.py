
from torchvision import transforms
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import cv2

class transformSet:
    #creating synthetic dataset
    #compress maybe?
    def train_set_twoShapes(size = args['train_len']):
        dataset = []
        for i in range(size):
            if(i%500==0):
                print(f' this is the {i}th iteration')
            #if(np.random.randint(0,2) == 0):
            #    dataset.append(createCircle())
            #else:
            dataset.append(createSquareNewestTry()+createCircleNewestTry())
        
        return dataset

    def eval_set_twoShapes(size = args['eval_len']):
        dataset = []
        for i in range(size):
            if(i%500==0):
                print(f' this is the {i}th iteration')
            #if(np.random.randint(0,2) == 0):
            #    dataset.append(createCircle())
            #else:
            dataset.append(createSquareNewestTry()+createCircleNewestTry())
        return dataset
    
    def gaussianNoise(train_tensor = train_tensor, mean = 0,var = 0.000009 ,size = args['train_len']):
        train_tensor_gaussian_noise = []
        #best so far has been var = 0.000009
        for i in range(size):
            if(i%500==0):
                print(f'this is the {i}th iteration')
            image = train_tensor[i]
            row,col,ch= image.shape
            gaussian_mean = float(mean)
            gaussian_var = float(var)
            sigma = float(gaussian_var**0.5)
            gauss = torch.normal(gaussian_mean,sigma,(row,col,ch))
            gauss = torch.reshape(gauss,(row,col,ch))
            noisy = image + gauss
            train_tensor_gaussian_noise.append(noisy)
        return train_tensor_gaussian_noise

    def normalize(size = args['train_len'],train_tensor = train_tensor):
        normalized_tensor = []
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        for i in range(size):
            if(i%500==0):
                print(f'this is the {i}th iteration')
            normalized_tensor.append(normalize(train_tensor[i]))
        return normalized_tensor
        


    #probability must be 1/something
    def randomizedGaussianNoise(train_tensor = train_tensor, mean = 0,var = 0.000009 ,size = args['train_len'], probability = 0.5):
        train_tensor_gaussian_noise = []
        #best so far has been var = 0.000009
        for i in range(size):
            if(i%500==0):
                print(f'this is the {i}th iteration')
            image = train_tensor[i]
            if(np.random.randint(0,probability**-1) == 0): #high is exclusive
                row,col,ch= image.shape
                gaussian_mean = float(mean)
                gaussian_var = float(var)
                sigma = float(gaussian_var**0.5)
                gauss = torch.normal(gaussian_mean,sigma,(row,col,ch))
                gauss = torch.reshape(gauss,(row,col,ch))
                noisy = image + gauss
                train_tensor_gaussian_noise.append(noisy)
            else:
                train_tensor_gaussian_noise.append(image)
            
        return train_tensor_gaussian_noise


# In[ ]:


    def randomizedNormalization(train_tensor = train_tensor,size = args['train_len'], probability = 0.5):
        normalized_tensor = []
        #best so far has been var = 0.000009
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        for i in range(size):
            if(i%500==0):
                print(f'this is the {i}th iteration')
            image = train_tensor[i]
            if(np.random.randint(0,probability**-1) == 0):
                normalized_tensor.append(normalize(image))
            else:
                normalized_tensor.append(image)
               
        return normalized_tensor
