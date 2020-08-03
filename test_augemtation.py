#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import Augmentor
from PIL import Image, ImageFilter
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from unet import UNET, train
from torch import nn
from train import train_net
from torchvision import transforms, utils, datasets
import torchvision
import matplotlib.image as mpimg

# In[4]:
print('sef')

## In[3]:


path = "./Image_Sequences/PhC-C2DL-PSC/Sequence 1/t098.tif"
mask_path = "./Image_Sequences/PhC-C2DL-PSC/Sequence 1 Masks/t098mask.tif"


# In[4]:


img=mpimg.imread(mask_path)
imgplot = plt.imshow(img)
plt.show()


# image = np.array(Image.open(path))
# mask = np.array(Image.open(mask_path))

## In[5]:


#image = np.array(Image.open(path))
#mask = np.array(Image.open(mask_path))

image = mpimg.imread(path)
#mask = Image.open(mask_path)
mask = mpimg.imread(mask_path).astype(np.uint8)
#mask_cv = cv2.imread(mask_path)


## In[6]:



rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)

transform_train = transforms.Compose([
    transforms.RandomCrop((576, 720), padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),

    transforms.ToTensor(),
    transforms.Normalize(rgb_mean, rgb_std),
])

#trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
#trainset = datasets.ImageFolder(root='./Image_Sequences/PhC-C2DL-PSC/Sequence 1/t000.tif',
#                                           transform=transform_train)

#dire = "./Image_Sequences/PhC-C2DL-PSC/Sequence 1/"
#trainset = datasets.ImageFolder(root=dire, transform=transform_train)
#trainset = datasets.ImageFolder(root='./Image_Sequences/PhC-C2DL-PSC/Sequence_1', transform=transform_train)
trainset = datasets.ImageFolder(root='./Image_Sequences/PhC-C2DL-PSC/train_seq1', transform=transform_train)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=10, shuffle=True, num_workers=2)


## In[101]:


trainset


## In[97]:


for x, y in trainloader:
    #print(x)
    plt.imshow(x[0][0])
    plt.show()
    print(y)


## In[80]:


np.shape(trainset[0][0])


## In[85]:


fig, axs = fig.subplots(4, 3)
axs[0, 0].plot(image)
axs[0, 1].plot(mask)

plt.show()


## In[103]:


#a = [x for x in trainset]
fig=plt.figure(figsize=(10, 10))
fig.add_subplot(4, 3, 1)
plt.imshow(image)
fig.add_subplot(4, 3, 2)
plt.imshow(mask)

i = 0
fig.add_subplot(4, 3, 4)
plt.imshow(trainset[i][0][0])
fig.add_subplot(4, 3, 5)
plt.imshow(trainset[i][0][1])
fig.add_subplot(4, 3, 6)
plt.imshow(trainset[i][0][2])

i = 1
fig.add_subplot(4, 3, 7)
plt.imshow(trainset[i][0][0])
fig.add_subplot(4, 3, 8)
plt.imshow(trainset[i][0][1])
fig.add_subplot(4, 3, 9)
plt.imshow(trainset[i][0][2])

i = 2
fig.add_subplot(4, 3, 10)
plt.imshow(trainset[i][0][0])
fig.add_subplot(4, 3, 11)
plt.imshow(trainset[i][0][1])
fig.add_subplot(4, 3, 12)
plt.imshow(trainset[i][0][2])

plt.show()

## In[7]:


fig=plt.figure()
fig.add_subplot(1, 2, 1)
plt.imshow(image)
fig.add_subplot(1, 2, 2)
plt.imshow(mask)


## In[4]:


rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(rgb_mean, rgb_std),
])

kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(root=data_root, train=True, transform=transform_train,
    transforms.Normalize(rgb_mean, rgb_std),
, download=False), batch_size=128, **kwargs)

print('len(train_loader)=',len(train_loader))


## In[89]:


# Define the demo dataset
class DogDataset3(Dataset):
    '''
    Sample dataset for Augmentor demonstration.
    The dataset will consist of just one sample image.
    '''

    def __init__(self, image, mask):
        self.image = image
        self.mask = mask

    def __len__(self):
        return 1 # return 1 as we have only one image

    def __getitem__(self, idx):
        # Returns the augmented image
        
        # Initialize the pipeline
        #p = Augmentor.DataPipeline([[np.array(self.image)]], labels=[[np.array(self.mask)]])
        #p = Augmentor.DataPipeline([np.array(self.image), np.array(self.mask)])
        p = Augmentor.DataPipeline([[np.array(self.image), np.array(self.mask)]])

        #p = Augmentor.DataPipeline([[[np.array(self.image)], [np.array(self.mask)]]])

        #p = Augmentor.DataPipeline([[np.array(self.image)]])
        #p = Augmentor.DataPipeline([[self.image]])

        
        # Apply augmentations
        p.rotate(0.5, max_left_rotation=10, max_right_rotation=10) # rotate the image with 50% probability
        p.shear(0.5, max_shear_left = 10, max_shear_right = 10) # shear the image with 50% probability
        p.zoom_random(0.5, percentage_area=0.7) # zoom randomly with 50% probability

        # Sample from augmentation pipeline
        images_aug = p.sample(idx)
        
        # Get augmented image
        augmented_image = images_aug
        
        # convert to tensor and return the result
        #return TF.to_tensor(augmented_image)
        return torch.tensor(augmented_image)

## Initialize the dataset, pass the augmentation pipeline as an argument to init function
train_ds = DogDataset3(image, mask)

## Initialize the dataloader
trainloader = DataLoader(torch.transpose(train_ds[2], 0, 1), batch_size=10, shuffle=True, num_workers=0)

#valid_dl = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)


## In[27]:


a = train_ds[2]


# torch.transpose(a,0,1)

# In[88]:


for y in trainloader:
    print(x)
    print(y)


# In[9]:


train_ds[0]


# for x in train_ds[10]:
#     print(x)
#           

# In[91]:


fig=plt.figure()
fig.add_subplot(2, 2, 1)
plt.imshow(image)
fig.add_subplot(2, 2, 2)
plt.imshow(mask)
fig.add_subplot(2, 2, 3)
plt.imshow(train_ds[1][0][0])
fig.add_subplot(2, 2, 4)
plt.imshow(train_ds[1][0][1])


# In[36]:


type(train_ds[2])
#train_ds[0].shape()
#np.shape(train_ds[0])


# In[9]:


model = UNET(1, 3).to('cpu')


# In[13]:


def acc_fn(y, y_true):
    return 0


# In[41]:


for x, y in trainloader:
    print(y)
 


# In[79]:


optimizer = torch.optim.Adam(model.parameters(),eps=0.000001,lr=0.0001,
                                 betas=(0.9,0.999),weight_decay=0.0001)
loss = nn.CrossEntropyLoss()
#train(model, trainloader, [0], loss, optimizer, acc_fn, 1)
train_net(model, trainloader, loss, optimizer, 1)


# In[ ]:


len(trainloader.data_queue)


# In[ ]:




# Initialize pipeline
p = Augmentor.DataPipeline([[np.array(image), np.array(mask)]])

# Apply augmentations
p.rotate(1, max_left_rotation=3, max_right_rotation=3)
p.shear(1, max_shear_left = 3, max_shear_right = 3)
p.zoom_random(1, percentage_area=0.9)

# Sample from augmentation pipeline
images_aug = p.sample(1)

# Access augmented image and mask
augmented_image = images_aug[0][0]
augmented_mask = images_aug[0][1]

