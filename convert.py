import numpy as np
import pandas as pd
from pdb import set_trace as bp

# Imports PIL module  
from PIL import Image 
  
# open method used to open different extension image file 
more = True
imgs=[]
a, b, c = 0, 0, 0
#while(more):
path =  './Image_Sequences/PhC-C2DL-PSC/'
counter = 0
for a in range(10):
    for b in range(10):
        for c in range(10):
            try:
                img_name = path + 'Sequence 1/t' + str(a) + str(b) + str(c)+'.tif'
                #imgs.append(np.array([counter, np.array(Image.open(img_name).convert("L"))] ))
                imgs.append([counter, Image.open(img_name).convert("L")] )
                #imgs.append(np.array([counter, Image.open(img_name).convert("L")))
                counter +=1
            except:
                break

#imgs_pd = pd.DataFrame(imgs)
#imgs_df = pd.DataFrame(np.array(imgs))
##
#np.save(path + 'seq_1.npy', imgs)
#imgs_df.to_csv(path + 'seq_1.csv')
#imgs_df.to_csv(path + 'seq_1.csv', header=False, index=False)

#bp()
##
imgs = np.array(imgs)
print(type(imgs))
print(type(imgs[0]))
print(type(imgs[0][0]))
##

