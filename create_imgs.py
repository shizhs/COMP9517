import numpy as np
import pandas as pd
from pdb import set_trace as bp
from PIL import Image, ImageFilter
import cv2
from skimage import color
from skimage import io
import matplotlib.pyplot as plt
import math
from scipy import ndimage as ndi
from skimage.feature import peak_local_max


##
def con_stretch(img):
    c = 255
    d = 0
    b = 255
    for row in img:
        for pix in row:
            c = pix if pix < c else c
            d = pix if pix > d else d

    for row in range(len(img)):
        for x in range(len(img[row])):
            img[row][x] = (img[row][x] - c)*(b/(d-c))

def background_subtract(img, N, M):
    if (M == 0):
        A = fast_min_max_filter(img, N, True)
        B = fast_min_max_filter(img, N, False)
    elif (M == 1):
        A = fast_min_max_filter(img, N, False)
        B = fast_min_max_filter(img, N, True)
    #B = B-B.min()
    O = img - B
    con_stretch(O)
    return O

def fast_min_max_filter(img, N, max_neighbour):
    #out = np.zeros_like(img)
    out =  np.zeros(np.shape(img))
    col_values = np.zeros(np.shape(img))
    for y in range(len(img)):
        for i in range(N):
            low = y-N if y-N > 0 else 0
            high = y+N if y+N < len(img) else len(img)-1
            #bp()
            col_values[y][i] = img.T[i][low:high+1].max() if max_neighbour == True else  img.T[i][low:high+1].min()

        for x in range(len(img[y])):
            low = y-N if y-N > 0 else 0
            high = y+N if y+N < len(img) else len(img)-1
            if x+N < len(img[y]):
                col_values[y][x+N] = img.T[x+N][low:high+1].max() if max_neighbour == True else img.T[x+N][low:high+1].min()
            low_col = x-N if x-N > 0 else 0
            high_col = x+N if x+N < len(img[y]) else len(img[y])-1
            out[y][x] = col_values[y][low_col:high_col+1].max() if max_neighbour == True else col_values[y][low_col:high_col+1].min()
    return out
##
#imgp1 = cv2.imread("./Image_Sequences/PhC-C2DL-PSC/Sequence 1/t001.tif")
path = "./Image_Sequences/PhC-C2DL-PSC/Sequence 1/t001.tif"
image = Image.open(path).convert('L')
image = color.rgb2gray(np.asarray(image))

##
#iteration = 3
#for i in range(iteration):
#    img = ndi.minimum_filter(image.copy(), size=3)
#for i in range(iteration):
#    img = ndi.maximum_filter(img, size=3)


#filter_size = 3
#img = image.copy()
#img = ndi.minimum_filter(img, size=filter_size)
#img = ndi.maximum_filter(img, size=filter_size)

#image = image - img
img = background_subtract(image.copy(), 3, 0)
fig=plt.figure()
fig.add_subplot(2, 2, 1)
plt.imshow(image)
fig.add_subplot(2, 2, 2)
plt.imshow(img)
plt.show()

##
#img = Image.open(path).convert('L')
img = Image.open(path)
im2 = img.filter(ImageFilter.MaxFilter(size = 3)) 
#im2 = im2.filter(ImageFilter.MaxFilter(size = 3)) 
#im2 = im2.filter(ImageFilter.MinFilter(size = 3)) 
im2 = im2.filter(ImageFilter.MinFilter(size = 3)) 
im2.show()
img.show()
##


imgplot = plt.imshow(img)
plt.show()
#img = Image.open(path).convert("LA")
#img = np.array(img)
#image = cv2.findContours(np.array(imgp1).copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
##
fig=plt.figure(figsize=(20, 20))
fig.add_subplot(4, 4, 1)
plt.imshow(img)
#plt.show()
#cv2.imshow('Original Image', img) 
#cv2.waitKey(0)
##
#image = cv2.findContours(np.array(imgp1).copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
#image, contours = cv2.findContours(img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
#image, contours = cv2.findContours(img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
#image = cv2.findContours(img.copy(), 1, 1)
##
#fig=plt.figure(figsize=(2, 2))
#fig=plt.figure()
#fig.add_subplot(1, 2, 1)
#plt.imshow(img)

#image = cv2.fastNlMeansDenoising(img,50,50,7,21)
#fig.add_subplot(1, 2, 2)
#imgplot = plt.imshow(image)
#
#image = cv2.fastNlMeansDenoising(img,50,50,7,21)
#fig.add_subplot(1, 2, 2)
##
###
#ret, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
#imgplot = plt.imshow(otsu)
#plt.show()
##

img = ndi.maximum_filter(image, size=10, mode='constant')




##
#image = image[:,:,0]
i =1
image = img
image = cv2.fastNlMeansDenoising(image,50,50,7,21)
sigma=15/(2*math.sqrt(2*np.log(2)))
image=cv2.GaussianBlur(image,(3, 3),sigma, sigma)
image = ndi.maximum_filter(image, size=10, mode='constant')
local_max = peak_local_max(image, indices=False, min_distance=15)
fig=plt.figure()
fig.add_subplot(1, 4, 1)
plt.imshow(img, cmap="gray")
local_max=local_max.astype(np.uint8)
fig.add_subplot(1, 4, 2)
plt.title("local maxima")
plt.imshow(local_max, cmap="gray")

fig.add_subplot(1, 4, 3)
ret,otsu = cv2.threshold(image,0,255, cv2.THRESH_OTSU)
plt.title("Otsu on max filtered image")
plt.imshow(otsu, cmap="gray")

fig.add_subplot(1, 4, 4)
m = ndi.maximum_filter(otsu, size=5, mode='constant')
#     m = ndi.minimum_filter(m, size=10, mode='constant')
plt.imshow(m, cmap="gray")
image = cv2.bitwise_and(otsu,otsu,mask = local_max)
#     fig.add_subplot(3, 4, i*4+4)
plt.title("Otsu and local maxima")
plt.imshow(image, cmap="gray")
plt.show()
#return m
##
fig=plt.figure()
fig.add_subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
#plt.show()
fig.add_subplot(1, 2, 2)
plt.imshow(image, cmap="gray")
plt.show()
##

#contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours, hierarchy = cv2.findContours(img.copy().astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
print(contours)
print(hierarchy)
##

