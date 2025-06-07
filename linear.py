import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from PIL import Image
import Data_processing
import Classification
import torch
#linear_regression and CNN
root_dir = "training/"
image_dir = root_dir + "images/"
files = os.listdir(image_dir)
#randomly split the data into training set and test set
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
for i in train_index: 
  X.append(Data_processing.extract_img_features_2d(image_dir + files[i], patch_size))
  Y.append(Data_processing.extract_values(gt_dir + files[i], patch_size,foreground_threshold))

X_batch = np.vstack(X)  
Y_batch = np.concatenate(Y) 
#case1: logistic regression
logreg = Classification.logistic_regression(X_batch,Y_batch)
avg_acc=0
for i in test_index:
 Xi = Data_processing.extract_img_features_2d(image_dir + files[i], patch_size)
 Zi = logreg.predict(Xi)
 Yi=  Data_processing.extract_values(gt_dir + files[i], patch_size,foreground_threshold)
 acc=np.sum(Zi==Yi)/len(Zi)
 avg_acc+=acc
avg_acc/=len(test_index)
print("Accuracy: ",avg_acc)

#use the model to predict the test set
#test_set_images folder has structue like this:
#test_set_images/test1/test1.png
#read the test set images
test_set_images = os.listdir("test_set_images/")
#create a submission.csv file
with open("submission1.csv", "w") as f:
  f.writelines("id,prediction\n")
  for i in range(len(test_set_images)):
    file_name="test_"+str(i+1)
    file_name+="/"+file_name+".png"
    X=Data_processing.extract_img_features_2d("test_set_images/"+file_name, patch_size)
    Z=logreg.predict(X)
    img=Data_processing.load_image("test_set_images/"+file_name)
    #we need to assign the same value to the each patch
    Z=Data_processing.label_to_img(38*16,38*16,16,16,Z)
    new_img=Data_processing.make_img_overlay(img,Z)
    plt.imsave("test_set_images/" + file_name + "_result1.png", Z,cmap="gray")
    plt.imsave("test_set_images/" + file_name + "_combined_result1.png", new_img, cmap="gray")
    f.writelines("{}\n".format(s) for s in Data_processing.mask_to_submission_strings("test_set_images/" + file_name + "_result1.png"))
  
  