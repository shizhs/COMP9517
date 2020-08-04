import torch
import Augmentor
from PIL import Image, ImageFilter
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from unet import UNET
from torch import nn
from train import train_net
from torchvision import transforms, utils, datasets
import torchvision
import matplotlib.image as mpimg
import torch.nn.functional as F
from pdb import set_trace as bp

path = "./Image_Sequences/PhC-C2DL-PSC/Sequence 1/t098.tif"
mask_path = "./Image_Sequences/PhC-C2DL-PSC/Sequence 1 Masks/t098mask.tif"

image = mpimg.imread(path).astype(np.float64)
#mask = Image.open(mask_path)
mask = mpimg.imread(mask_path).astype(np.float64)
##

fig=plt.figure(figsize=(10, 10))
fig.add_subplot(2, 1, 1)
plt.imshow(image)
fig.add_subplot(2, 1, 2)
plt.imshow(mask)
plt.show()
