



class encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 4, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64,kernel_size = 4, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2, padding = 1)
        self.conv5 = nn.Conv2d(in_channels = 256, out_channels = 32, kernel_size = 4, stride = 2, padding = 1)
    def forward(self,x):
        #print(x.shape)
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = F.relu(self.conv3(x))
        #print(x.shape)
        x = F.relu(self.conv4(x))
        #print(x.shape)
        x = F.relu(self.conv5(x))
        #print(x.shape)
        x = x.view(x.size(0),-1)
        
        return x


# In[22]:


class linear_between(nn.Module):
    def __init__ (self, linear_layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(32,32) for i in range(linear_layers)])
    def forward(self,x):
         #maybe replace with (x.view(x.size(0),-1))
        for layer in self.layers:
            x = layer(x)
        #print(x.shape)
        #print(x.shape)
        return x


# In[23]:


class decoder(nn.Module):
    def __init__ (self):
        super().__init__()
        
        self.convt1 = nn.ConvTranspose2d(in_channels = 32, out_channels = 256,kernel_size = 4, stride = 2, padding = 1)
        self.convt2 = nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)
        self.convt3 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 4, stride = 2, padding =1)
        self.convt4 = nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 4, stride = 2, padding = 1)
        self.convt5 = nn.ConvTranspose2d(in_channels = 32, out_channels = 3, kernel_size = 4, stride = 2, padding = 1)
    def forward(self,x):
        #print(x.shape)
        x = x.view(-1,32,1,1)
        x = F.relu(self.convt1(x))
        #print(x.shape)
        x = F.relu(self.convt2(x))
        #print(x.shape)
        x = F.relu(self.convt3(x))
        #print(x.shape)
        x = F.relu(self.convt4(x))
        #print(x.shape)
        x = torch.tanh(self.convt5(x))
        #print(x.shape)
        
        return x

class IMRAE(nn.Module):
    def __init__(self,linear_layers):
        super().__init__()
        self.linear_layers = linear_layers
        self.encoder = encoder()
        self.linear_between = linear_between(linear_layers)
        self.decoder = decoder()
        
    def forward(self,x):
        x = self.encoder(x)
        x = self.linear_between(x)
        x = self.decoder(x)
        return x
