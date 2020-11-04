#!/usr/bin/env python
# coding: utf-8

# In[2]:


#x^2+y^2 = r^2

import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import cv2
import utils.syntheticShapeDataset
import model.IMRAE

args = {
    'latent_dim':32,
    'lr':0.0001,
    'epochs':100,
    'batch_size' : 32,
    'train_len' : 50000,
    'eval_len' : 10000
}


# In[21]:
train_tensor_data,eval_tensor_data = syntheticShapeDataset('small')



test_dataloader = DataLoader(eval_tensor_data, batch_size = args['batch_size'], shuffle = True)
train_dataloader = DataLoader(train_tensor_data, batch_size = args['batch_size'], shuffle = True)


#maybe decrease the size to ensure square?
def train(lr,train_dataloader,num_epochs,regularization = None, l = 0,lmbda = 1e-10):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    imrae2 = IMRAE(l)
    imrae2.to(device)
    to_pil_image= transforms.ToPILImage()
    
    optimizer = torch.optim.Adam(params=imrae2.parameters(), lr=lr)
    num_epochs = num_epochs
    x=[]
    
    
    for epoch in range(num_epochs):
        train_loss_avg = 0
        num_batches = 0
        
        
        for batch in train_dataloader:
            optimizer.zero_grad()
            l1_regularization = 0
            l2_regularization = 0
            batch = batch.to(device)
            reconstructed = imrae2(batch)
            loss = F.mse_loss(reconstructed, batch)
            
            if(regularization=='l1'):
                l1_regularization = torch.norm(ae_real_trained_l1.encoder(batch),1)
            loss+= lmbda*l1_regularization  
            if(regularization == 'l2'):
                l2_regularization = torch.norm(ae_real_trained_l1.encoder(batch),2)**2
            loss+= lmbda*l2_regularization
            
            loss.backward()
            optimizer.step()
            train_loss_avg+=(loss.item())
            num_batches += 1
            x.append(to_pil_image(reconstructed[0].detach().cpu().clone()))
    
        train_loss_avg /= num_batches
        print(f'Epoch [{epoch+1} / {num_epochs}] average reconstruction error: {train_loss_avg}')
        
    return imrae2,x,train_loss_avg
   


# In[26]:


def train_trained_model(imrae2,lr,train_dataloader,num_epochs,regularization = None,lmbda = 1e-10):
    
    to_pil_image= transforms.ToPILImage()
    
    optimizer = torch.optim.Adam(params=imrae2.parameters(), lr=lr)
    num_epochs = num_epochs
    x=[]
    
    
    for epoch in range(num_epochs):
        train_loss_avg = 0
        num_batches = 0
        
        for batch in train_dataloader:
            l1_regularization = torch.FloatTensor(0)
            l2_regularization = torch.FloatTensor(0)
            optimizer.zero_grad()
            batch = batch.to(device)
            reconstructed = imrae2(batch)
            loss = F.mse_loss(reconstructed, batch)
            if(regularization=='l1'):
                l1_regularization = torch.norm(ae_real_trained_l1.encoder(batch),1)
            loss+= lmbda*l1_regularization  
            if(regularization == 'l2'):
                l2_regularization = torch.norm(ae_real_trained_l1.encoder(batch),2)**2
            loss+= lmbda*l2_regularization     
            
            print(loss.item())
            loss.backward()
            optimizer.step()
            train_loss_avg+=(loss.item())
            num_batches += 1
            x.append(to_pil_image(reconstructed[0].detach().cpu().clone()))
    
        train_loss_avg /= num_batches
        print(f'Epoch [{epoch+1} / {num_epochs}] average reconstruction error: {train_loss_avg}')
        
    return imrae2,x,train_loss_avg



#alternative if there is malfunction in train 

#regularization = None
#lmbda = 1e-10
optimizer = torch.optim.Adam(params=imrae_4.parameters(), lr=0.0001)
num_epochs = 100
x=[]
    
for epoch in range(num_epochs):
    train_loss_avg = 0
    num_batches = 0
        
        
    for batch in train_dataloader:
        optimizer.zero_grad()
        l1_regularization = 0
        l2_regularization = 0
        batch = batch.to(device)
        reconstructed = imrae_4(batch)
        loss = F.mse_loss(reconstructed, batch)
        
        #if(regularization=='l1'):
        #    l1_regularization = torch.norm(imrae_4.encoder(batch),1)
        #loss+= lmbda*l1_regularization  
        #if(regularization == 'l2'):
        #    l2_regularization = torch.norm(imrae_4.encoder(batch),2)**2
        #loss+= lmbda*l2_regularization
            
        loss.backward()
        optimizer.step()
        train_loss_avg+=(loss.item())
        num_batches += 1
        x.append(to_pil_image(reconstructed[0].detach().cpu().clone()))
    
    train_loss_avg /= num_batches
    print(f'Epoch [{epoch+1} / {num_epochs}] average reconstruction error: {train_loss_avg}')
        


ae_real_trained_l2, image_array_ae_real_l2, train_loss_avg_ae_real_l2 = train(lr=0.0001 ,train_dataloader = train_twoShape_dataloader ,num_epochs = 5,regularization = "l2")

ae_real_trained_l1, image_array_ae_real_l2, train_loss_avg_ae_real_l2 = train(lr=0.0001 ,train_dataloader = train_twoShape_dataloader ,num_epochs = 5,regularization = "l1")

ae_real_trained, image_array_ae_real, train_loss_avg_ae_real = train(lr=0.0001 ,train_dataloader = train_dataloader ,num_epochs = 100)

imrae_2_trained_real, image_array_imrae_2_real, train_loss_avg_imrae_2_trained_real = train(lr = 0.0001, train_dataloader = train_dataloader, num_epochs = 100, l = 2)

imrae_4_trained, image_array_imrae_4, train_loss_avg_imrae_4_trained = train(lr = 0.0001, train_dataloader = train_dataloader, num_epochs = 10, l = 4)


torch.save([MODEL_NAME].state_dict(),"[FILENAME].pt")





def singular_values(irmae, test_dataloader,layers = 0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    irmae.eval()
    z = []
    for batch in test_dataloader:
    
        num_matrices = 0
        
        with torch.no_grad():
            batch = batch.to(device)
            latent_vec = irmae.encoder(batch)
            if(layers > 0):
                z_portion = irmae.linear_between(latent_vec)
            else:
                z_portion = latent_vec
            z.append(z_portion)
    
    z = torch.cat(z,axis = 0).cpu().numpy()
    latent_covariance = np.cov(z,rowvar = False)
    
    
           
    _,diag,_ = np.linalg.svd(latent_covariance)
    return (diag/max(diag)),latent_covariance



# In[76]:
diag_ae_real_trained, latent_cov_ae_real_train = singular_values(ae_real_trained, test_dataloader)
diag_2_real_trained, latent_cov_2_real_trained = singular_values(imrae_2_trained_real, test_dataloader)
#ae_real_trained_l1_diag, cov_l1 = singular_values(model5, test_dataloader)
ae_real_trained_l2_diag,cov_l2 = singular_values(ae_real_trained_l2, test_dataloader)
#test_diag,_ = singular_values(ae_real_trained_l2,test_dataloader)
ae_real_trained_l1_diag, cov_l1 = singular_values(ae_real_trained_l2,test_dataloader)
imrae_4_diag, cov_4 = singular_values(imrae_4_trained, test_dataloader)


# In[77]:

plt.plot(diag_ae_real_trained,label = f'ae with matrix rank: {torch.matrix_rank(torch.tensor(latent_cov_ae_real_train))}')
plt.plot(diag_2_real_trained, label = f"imrae (l=2) (cv2 rects: {torch.matrix_rank(torch.tensor(latent_cov_2_real_trained))}")
plt.plot(ae_real_trained_l2_diag, label = f"ae (l2 reg)  cv2 rects: {torch.matrix_rank(torch.tensor(cov_l2))}")
plt.plot(ae_real_trained_l1_diag, label = f"ae (l1 reg) cv2 rects: {torch.matrix_rank(torch.tensor(cov_l1))}")
plt.plot(imrae_4_diag, label = f"imrae (l=4) cv2 rects: {torch.matrix_rank(torch.tensor(cov_4))}")


plt.vlines(7,-1,1,linestyles = "dashed")
plt.ylim(0,0.01)
plt.ylabel('singular values')
plt.xlabel('singular value rank')
plt.title("cv2 squares, cv2 rects")
plt.legend()


# In[ ]:



import torchvision.utils


def plt_images(image):
    to_pil_image= transforms.ToPILImage()
    plt.imshow(to_pil_image(image))

images,labels = iter(test_dataloader).next()

plt_images(torchvision.utils.make_grid(images[1:31],10,3))
plt.show()


# In[71]:


import cv2 as cv
def interpolate(models,x):
    to_pil_image= transforms.ToPILImage()
    fig,axs = plt.subplots(len(models),len(x), figsize= (20,12))
    index = np.random.randint(30)
    images = iter(test_dataloader).next()
    images = images.to(device)
    row = 0
    for (model,name) in models:
        model.eval()
        z = model.linear_between(model.encoder(images))
        z1 = z[index]
        z2 = z[index+1]
        for b,i in enumerate(x):
            interpolated_image = i*z1 + (1-i) * z2
            ans = torch.reshape(model.decoder(interpolated_image),(3,32,32))
            ans_np = np.transpose(ans.cpu().detach().numpy(), [2,1,0])
            axs[row, b].imshow(ans_np)
            #axs[row, b].imshow(to_pil_image(torch.reshape(model.decoder(interpolated_image),(3,32,32)).cpu()))
            axs[row, b].set_title(f'{name}, x:{np.round(i,decimals = 1)}',fontdict = {'fontsize':8})
        row+=1
            
            



interpolate([(ae_real_trained,"ae"),(ae_real_trained_l1,"ae(reg = l1)"),(ae_real_trained_l2,"ae(reg = l2)"),(imrae_2_trained_real,"irmae (l=2)"),(imrae_4_trained,"irmae (l=4)")],np.round(np.linspace(0,1,10),decimals = 1))

