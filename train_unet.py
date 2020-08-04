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
import torch.nn.functional as F
from pdb import set_trace as bp

path = "./Image_Sequences/PhC-C2DL-PSC/Sequence 1/t098.tif"
img_path = './Image_Sequences/DIC-C2DH-HeLa/Sequence 1/'
mask_path = './Image_Sequences/DIC-C2DH-HeLa/Sequence 1 Masks/'

def find_traning_data(mask_path, img_path):
    imgs = []
    masks = []
    for i in range(100):
        try:
            masks.append(mpimg.imread(mask_path +'t0'+ str(i) + 'mask.tif').astype(np.float64))
            imgs.append(mpimg.imread(img_path +'t0'+ str(i) + '.tif').astype(np.float64))
            #fig=plt.figure(figsize=(10, 10))
            #fig.add_subplot(2, 1, 1)
            #plt.imshow(imgs[-1])
            #fig.add_subplot(2, 1, 2)
            #plt.imshow(masks[-1])
            #plt.show()
        except:
            pass
    return imgs, masks

##
imgs, masks = find_traning_data(mask_path, img_path)

##
test_imgs = []
for i in range(84):
    if i < 10 :
        test_imgs.append(mpimg.imread(img_path +'t00'+ str(i) + '.tif').astype(np.float64))
    else:
        test_imgs.append(mpimg.imread(img_path +'t0'+ str(i) + '.tif').astype(np.float64))
##

#image = mpimg.imread(path).astype(np.float64)
##mask = Image.open(mask_path)
#mask = mpimg.imread(mask_path).astype(np.float64)

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
        p = Augmentor.DataPipeline([[np.array(self.image), np.array(self.mask)]])

        # Apply augmentations
        p.zoom_random(0.5, percentage_area=0.7) # zoom randomly with 50% probability
        p.rotate(0.5, max_left_rotation=10, max_right_rotation=10) # rotate the image with 50% probability
        p.shear(0.5, max_shear_left = 10, max_shear_right = 10) # shear the image with 50% probability
        p.flip_random(0.5)

        # Sample from augmentation pipeline
        images_aug = p.sample(idx)
        
        # Get augmented image
        augmented_image = torch.tensor(images_aug)
        
        # convert to tensor and return the result
        #bp()
        return augmented_image.unsqueeze(2)

def train(model, imgs, masks, optimizer): 
    for img, mask in zip(imgs, masks):
        print('change img')
        train_ds = DogDataset3(img, mask)
        data_set = train_ds[50]
        train_dataset = torch.utils.data.TensorDataset(data_set[:, 0], data_set[:, 1])
        trainloader = DataLoader(train_dataset, batch_size=5, shuffle=False)

        for data, target in trainloader:
            optimizer.zero_grad()    # zero the gradients
            output = model(data)       # apply network
            #bp()
            #loss = nn.CrossEntropyLoss(output, target)
            loss = F.mse_loss(output, target)
            #loss = F.nll_loss(output, target)
            print('training loss: ', loss)
            loss.backward()          # compute gradients
            optimizer.step()         # update weights

##
def test(model, data):
    model.eval()
    with torch.no_grad():
        for x in data:
            output = model(x)
            fig=plt.figure(figsize=(10, 10))
            fig.add_subplot(2, 1, 1)
            plt.imshow(x)
            fig.add_subplot(2, 1, 2)
            plt.imshow(output)
            plt.show()

##
model = UNET(1, 1).to('cpu')
optimizer = torch.optim.Adam(model.parameters(),eps=0.000001,lr=0.0001,
                                 betas=(0.9,0.999),weight_decay=0.0001)

##
train(model, imgs, masks, optimizer)
##
test(model, torch.tensor(test_imgs[:3]))
##


train_ds = DogDataset3(image, mask)

# Initialize the dataloader
data_set = train_ds[13]
train_dataset = torch.utils.data.TensorDataset(data_set[:, 0],data_set[:, 1])
trainloader = DataLoader(train_dataset, batch_size=3, shuffle=False)
#bp()
