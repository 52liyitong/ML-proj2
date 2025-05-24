#in this file we will use FCN model to segment the road
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from PIL import Image
import Data_processing
import Classification
import torch
root_dir = "training1/"
image_dir = root_dir + "images/"
files = os.listdir(image_dir)
#randomly split the data into training set and test set
n = len(files)
n=10
train_index = np.random.choice(n, size=int(n*0.8), replace=False)
test_index = np.setdiff1d(np.arange(n), train_index)


imgs = [Data_processing.load_image(image_dir + files[i]) for i in train_index]
imgs_fortrain = [Data_processing.load_image(image_dir + files[i]) for i in test_index]

gt_dir = root_dir + "groundtruth/"

gt_imgs = [Data_processing.load_image(gt_dir + files[i]) for i in train_index]
gt_imgs_fortrain = [Data_processing.load_image(gt_dir + files[i]) for i in test_index]


foreground_threshold = 0.25 
#create training set
#case2: CNN
#we need to load the data again

batch_size=4
X=imgs
Y=gt_imgs
#convert Y to 0,1 matrix
foreground_threshold = 0.25 

#batch the data
X_batch=[]
Y_batch=[]
for i in range(0,len(X),batch_size):
  X_batch.append(X[i:i+batch_size])
  Y_batch.append(Y[i:i+batch_size])
print(len(X_batch))
print(len(Y_batch))

X_batch=np.asarray(X_batch)
Y_batch=np.asarray(Y_batch)
print(X_batch.shape)
#convert to torch tensor
X_batch=torch.from_numpy(X_batch).float()
Y_batch=torch.from_numpy(Y_batch).long()
model=Classification.train_FCN8s(X_batch,Y_batch)
print("training done")
file_name=files[0]
img=Data_processing.load_image(image_dir + file_name)
img=torch.from_numpy(img).float()
#add batch dimension
img=img.unsqueeze(0)
print(img.shape)
out_image=model(img)
out_image=torch.argmax(out_image,dim=1)
out_image=out_image.squeeze(0)
print(out_image.shape)
out_image=out_image.cpu().numpy()
#plot the output image
plt.imshow(out_image,cmap='gray')
plt.show()

