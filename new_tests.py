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

##
path = "./Image_Sequences/PhC-C2DL-PSC/Sequence 1/t098.tif"
mask_path = "./Image_Sequences/PhC-C2DL-PSC/Sequence 1 Masks/t098mask.tif"

image = mpimg.imread(path)
#mask = Image.open(mask_path)
mask = mpimg.imread(mask_path).astype(np.uint8)

##

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
        p.zoom_random(0.5, percentage_area=0.7) # zoom randomly with 50% probability
        p.rotate(0.5, max_left_rotation=10, max_right_rotation=10) # rotate the image with 50% probability
        p.shear(0.5, max_shear_left = 10, max_shear_right = 10) # shear the image with 50% probability
        p.flip_random(0.5)
        #p.crop_random(0.9, 0.7, randomise_percentage_area=False)

        # Sample from augmentation pipeline
        images_aug = p.sample(idx)
        
        # Get augmented image
        augmented_image = images_aug
        
        # convert to tensor and return the result
        #return TF.to_tensor(augmented_image)
        return torch.tensor(augmented_image)


## Initialize the dataset, pass the augmentation pipeline as an argument to init function
train_ds = DogDataset3(image, mask)

# Initialize the dataloader
trainloader = DataLoader(train_ds[10], batch_size=2, shuffle=False, num_workers=0)
#trainloader = DataLoader(torch.transpose(train_ds[10], 0, 1), batch_size=10, shuffle=False)
#trainloader = DataLoader(torch.transpose(train_ds[10], 0, 1), batch_size=2, shuffle=False, num_workers=0)

##
#fig=plt.figure(figsize=(10, 10))
#for x, y in trainloader:
#for x in trainloader:
for data in trainloader:
    print(data.shape)
    for x, y in data:
        print(len(x))
        print(type(x))
        print(x.shape)
        fig=plt.figure(figsize=(10, 10))
        fig.add_subplot(3, 2, 1)
        plt.imshow(image)
        fig.add_subplot(3, 2, 2)
        plt.imshow(mask)
        fig.add_subplot(3, 2, 3)
        plt.imshow(x)
        fig.add_subplot(3, 2, 4)
        plt.imshow(y)
        #fig.add_subplot(3, 2, 5)
        #plt.imshow(y[1])
        #fig.add_subplot(3, 2, 6)
        #plt.imshow(x[1])

        plt.show()
##
#for x in trainloader:
for x, y in trainloader:
    print(len(x))
    print(type(x))
    print(x.shape)
    print(y.shape)
    #print(np.asarray([x]))
    #print(torch.tensor(np.array([x])).shape)
    #print(torch.tensor([y]).shape)
#x, y = trainloader
##

def acc_fn(y, y_true):
    return 0

model = UNET(1, 1).to('cpu')

## In[79]:
testloader = DataLoader(x, batch_size=10, shuffle=False, num_workers=0)
#print(model([x]))

## In[79]:
for data in testloader:
    print(data.shape)
    model.eval()
    with torch.no_grad():
        #for data, target in test_loader:
        #    data, target = data.to(device), target.to(device)
        #    output = model(data)
            # sum up batch loss
        for x in data:
            x = x.to('cpu')
            print(model(x))
## In[79]:

optimizer = torch.optim.Adam(model.parameters(),eps=0.000001,lr=0.0001,
                                 betas=(0.9,0.999),weight_decay=0.0001)
loss = nn.CrossEntropyLoss()
#train(model, trainloader, [0], loss, optimizer, acc_fn, 1)
train_net(model, trainloader, loss, optimizer, 1)


## In[ ]:

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

