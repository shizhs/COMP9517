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
from pdb import set_trace as bp


path = "./Image_Sequences/PhC-C2DL-PSC/Sequence 1/t098.tif"
mask_path = "./Image_Sequences/PhC-C2DL-PSC/Sequence 1 Masks/t098mask.tif"

image = mpimg.imread(path)
#mask = Image.open(mask_path)
mask = mpimg.imread(mask_path).astype(np.uint8)

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


train_ds = DogDataset3(image, mask)

# Initialize the dataloader
trainloader = DataLoader(train_ds[10], batch_size=2, shuffle=False, num_workers=0)
for x, y in trainloader:
    print(len(x))
    print(type(x))
    print(x.shape)
    print(y.shape)
    #print(np.asarray([x]))
    #print(torch.tensor(np.array([x])).shape)
    #print(torch.tensor([y]).shape)
#x, y = trainloader

def acc_fn(y, y_true):
    return 0

model = UNET(1, 1).to('cpu')

testloader = DataLoader(x, batch_size=10, shuffle=False, num_workers=0)
#print(model([x]))


optimizer = torch.optim.Adam(model.parameters(),eps=0.000001,lr=0.0001,
                                 betas=(0.9,0.999),weight_decay=0.0001)
loss = nn.CrossEntropyLoss()
#train(model, trainloader, [0], loss, optimizer, acc_fn, 1)

for data in testloader:
    print(data.shape)
    model.eval()
    bp()
    train_net(model, data, loss, optimizer, 1)
    with torch.no_grad():
        #for data, target in test_loader:
        #    data, target = data.to(device), target.to(device)
        #    output = model(data)
            # sum up batch loss
        for x in data:
            x = x.to('cpu')
            bp()
            print(model(x))



