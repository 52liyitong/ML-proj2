import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from PIL import Image
import Data_processing
import Classification
import torch
root_dir = "training/"
image_dir = root_dir + "images/"
files = os.listdir(image_dir)
#CNN_with_data_augmentation
n = len(files)
train_index = np.random.choice(n, size=int(n*0.8), replace=False)
test_index = np.setdiff1d(np.arange(n), train_index)

imgs = [Data_processing.load_image(image_dir + files[i]) for i in train_index]
imgs_fortrain = [Data_processing.load_image(image_dir + files[i]) for i in test_index]

gt_dir = root_dir + "groundtruth/"

gt_imgs = [Data_processing.load_image(gt_dir + files[i]) for i in train_index]
gt_imgs_fortrain = [Data_processing.load_image(gt_dir + files[i]) for i in test_index]

patch_size = 16  # each patch is 16*16 pixels
img_patches = [Data_processing.img_crop(imgs[i], patch_size, patch_size) for i in range(len(train_index))]
gt_patches = [Data_processing.img_crop(gt_imgs[i], patch_size, patch_size) for i in range(len(train_index))]

img_patches1=[Data_processing.img_crop(imgs_fortrain[i], patch_size, patch_size) for i in range(len(test_index))]
gt_patches1=[Data_processing.img_crop(gt_imgs_fortrain[i], patch_size, patch_size) for i in range(len(test_index))]
# Linearize list of patches
img_patches = np.asarray(
    [
        img_patches[i][j]
        for i in range(len(img_patches))
        for j in range(len(img_patches[i]))
    ]
)
gt_patches = np.asarray(
    [
        gt_patches[i][j]
        for i in range(len(gt_patches))
        for j in range(len(gt_patches[i]))
    ]
)
img_patches1=np.asarray(
    [
        img_patches1[i][j]
        for i in range(len(img_patches1))
        for j in range(len(img_patches1[i]))
    ]
)
gt_patches1=np.asarray(
    [
        gt_patches1[i][j]
        for i in range(len(gt_patches1))
        for j in range(len(gt_patches1[i]))
    ]
)

foreground_threshold = 0.25 
#create training set
X = []
Y = []

#case2: CNN
#we need to load the data again

batch_size=50
X=img_patches
Y=gt_patches
#here we randomly choose 100 patches to do data_augmentation
#index list
index_list=np.random.choice(len(X),size=2000,replace=False)
X_list = X.tolist()  # 转换为列表
gt_patches_list = Y.tolist()  # 转换为列表
for i in index_list:
    X_list.append(Data_processing.image_crop_with_data_augmentation(X[i]))
    gt_patches_list.append(Y[i])
X = np.array(X_list)  # 转回 NumPy 数组
Y = np.array(gt_patches_list)  # 转回 NumPy 数组
#convert Y to 0,1 matrix
foreground_threshold = 0.25 
Y=np.asarray([Data_processing.value_to_class(np.mean(gt_patches_list[i]),foreground_threshold) for i in range(len(gt_patches_list))])
#batch the data
X_batch=[]
Y_batch=[]
for i in range(0,len(X),batch_size):
  X_batch.append(X[i:i+batch_size])
  Y_batch.append(Y[i:i+batch_size])
X_batch=np.asarray(X_batch)
Y_batch=np.asarray(Y_batch)
#convert to torch tensor
X_batch=torch.from_numpy(X_batch).float()
Y_batch=torch.from_numpy(Y_batch).long()
model=Classification.train_CNN(X_batch,Y_batch)
#accuracy
avg_acc=0

X=img_patches1
Y=gt_patches1
Y=np.asarray([Data_processing.value_to_class(np.mean(gt_patches1[i]),foreground_threshold) for i in range(len(gt_patches1))])
num_positive=np.sum(Y==1)
num_negative=np.sum(Y==0)
total_num=len(Y)
print(np.sum(Y==1)/len(Y))
print(np.sum(Y==0)/len(Y))
X_batch=[]
Y_batch=[]
for i in range(0,len(X),batch_size):
  X_batch.append(X[i:i+batch_size])
  Y_batch.append(Y[i:i+batch_size])
X_batch=np.asarray(X_batch)
Y_batch=np.asarray(Y_batch)
X_batch=torch.from_numpy(X_batch).float()
Y_batch=torch.from_numpy(Y_batch).long()
 #batch the patchset
total=0
count=0
with torch.no_grad():
  for i in range(len(X_batch)):
    output=model(X_batch[i])
    pred=torch.argmax(output,axis=1)
    acc=torch.sum(pred==Y_batch[i]).item()/len(Y_batch[i])
    total+=acc
    count+=1
total/=count
print("Accuracy: ",total)
test_set_images = os.listdir("test_set_images/")
#create a submission.csv file
with open("submission3.csv", "w") as f:
  f.writelines("id,prediction\n")
  for i in range(len(test_set_images)):
    file_name="test_"+str(i+1)
    file_name+="/"+file_name+".png"
    X=np.asarray(Data_processing.img_crop(Data_processing.load_image("test_set_images/"+file_name),patch_size,patch_size))
    X=torch.from_numpy(X).float()
    Z=model(X)
    Z=torch.argmax(Z,axis=1)
    Z=Z.numpy()
    img=Data_processing.load_image("test_set_images/"+file_name)
    #we need to assign the same value to the each patch
    Z=Data_processing.label_to_img(38*16,38*16,16,16,Z)
    new_img=Data_processing.make_img_overlay(img,Z)
    plt.imsave("test_set_images/" + file_name + "_result3.png", Z,cmap="gray")
    plt.imsave("test_set_images/" + file_name + "combained_result3.png", new_img, cmap="gray")
    f.writelines("{}\n".format(s) for s in Data_processing.mask_to_submission_strings("test_set_images/" + file_name + "_result3.png"))












