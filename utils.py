
from torchvision import transforms
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

class syntheticShapeDataset:
    def rectangleOrCircle():
    if(np.random.uniform(0,1)) >= 0.5:
        return 'rectangle'
    else:
        return 'circle'


def randParamsNumpy():
    x = np.random.randint(low = 8,high = 25)
    y = np.random.randint(low = 8,high = 25)
    size = np.random.randint(low = 3,high = 9)
    return x,y,size

def pickColor(only_rand_blue = False):
    if(only_rand_blue == True):
        red = 0
        green = 0
        blue = np.random.uniform(0,1)
    else:
        red = np.random.uniform(0,1)
        green = np.random.uniform(0,1)
        blue = np.random.uniform(0,1)
    return red,green,blue

def randParamsPytorch():
    x = torch.randint(low = 8,high = 25,size = (1,1))[0][0]
    y = torch.randint(low = 8,high = 25,size = (1,1))[0][0]
    size = torch.randint(low = 3,high = 9,size = (1,1))[0][0]
    return x,y,size

 
def createSquare(only_rand_blue = False):
    x,y,size = randParamsPytorch()
    red,green,blue = pickColor(only_rand_blue)
    z = torch.zeros(3,32,32) #or torch.ones for white background
    z[0][:,x-size:x+size][y-size:y+size] = red
    z[1][:,x-size:x+size][y-size:y+size] = green
    z[2][:,x-size:x+size][y-size:y+size] = blue
    return z
#,[x,y,size]

def createCircle(only_rand_blue = False):
    
    x,y,size = randParamsPytorch()
    red,green,blue = pickColor(only_rand_blue)
    z = torch.zeros(3,32,32) #or torch.ones for white background
    
    X = np.random.multivariate_normal([0, 0], [[1, 0], [0,1]], 10000)
    Z = X / 10 + X / np.sqrt(np.square(X).sum(axis=1, keepdims=True))
    #plt.plot(np.floor(size*np.array(Z[:,0])),np.floor(size*np.array(Z[:,1])),'x')
    new_dict = {}
    grouped_x = pd.DataFrame({'x':np.floor(size*np.array(Z[:,0])),'y':np.floor(size*np.array(Z[:,1]))}).groupby(by='x')
    for i in grouped_x:
        val_x1 = min([j if j>0 else 20 for j in i[1]['y']] )
        val_x2 = max([j if j<0 else -20 for j in i[1]['y']])
        val_x1 = val_x1+size+(x-size)
        val_x2= val_x2+size+(x-size)
        
        new_dict[(int(i[0]+size+(y-size)))] = [(int(val_x2)),(int(val_x1))]
#ask if size is radius or total size  
    #print(f'new_dct is {new_dict}')
    for i in range(2*size):
        y_axis = int(i+(y-size))
        bounds = new_dict[y_axis]
        
        #print(f'the bounds are{bounds}, size is {size} and x and y are {x},{y}')
        z[0][:,bounds[0]:bounds[1]][32-y_axis-1] = red
        z[1][:,bounds[0]:bounds[1]][32-y_axis-1] = green
        z[2][:,bounds[0]:bounds[1]][32-y_axis-1] = blue
        
    return z


# In[4]:


import math
from skimage.draw import circle
to_pil_image= transforms.ToPILImage()

def createCircleNewTry(only_rand_blue=False):
    x,y,size = randParamsPytorch()
    red,green,blue = pickColor(only_rand_blue)
    img = torch.zeros((3,32,32))
    rr,cc = circle(r = x,c = y,radius = size)
    img[0][rr, cc] = red
    img[1][rr, cc] = green
    img[2][rr, cc] = blue
    return img


# In[5]:


def createCircleNewestTry(only_rand_blue = False):
    to_pil_image= transforms.ToPILImage()
    x,y,size = randParamsNumpy()
    red,green,blue = pickColor(only_rand_blue)
    z = np.zeros((32,32,3))
    cv2.circle(z,(x,y),size,(red,green,blue),-1)
    z = np.transpose(z,[2,0,1])
    z = torch.FloatTensor(z)
    return z
#,[x,y,size]

def createSquareNewestTry(only_rand_blue = False):
    to_pil_image= transforms.ToPILImage()
    x,y,size = randParamsNumpy()
    red,green,blue = pickColor(only_rand_blue)
    z = np.zeros((32,32,3))
    cv2.rectangle(z,(x,y),(x+size,y+size),(red,green,blue),-1)
    z = np.transpose(z,[2,0,1])
    z = torch.FloatTensor(z)

    return z
#,[x,y,size]


