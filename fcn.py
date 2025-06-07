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
n =100
train_index = np.random.choice(n, size=int(n*0.8), replace=False)
test_index = np.setdiff1d(np.arange(n), train_index)

#check if train_index and test_index have same elements,if yes,out put the index
#randomly choose 10% in train_index to do data augmentation
index_list=np.random.choice(len(train_index),size=int(len(train_index)*0.1),replace=False)

  #with 50% probability to rotate the image
  

imgs = [Data_processing.load_image(image_dir + files[i]) for i in train_index]
imgs_fortrain = [Data_processing.load_image(image_dir + files[i]) for i in test_index]
imgs=Data_processing.normalize_images(imgs)
imgs_fortrain=Data_processing.normalize_images(imgs_fortrain)
gt_dir = root_dir + "groundtruth/"

gt_imgs = [Data_processing.load_image(gt_dir + files[i]) for i in train_index]
gt_imgs_fortrain = [Data_processing.load_image(gt_dir + files[i]) for i in test_index]
gt_imgs=Data_processing.normalize_images(gt_imgs)
gt_imgs_fortrain=Data_processing.normalize_images(gt_imgs_fortrain)

#for all imgs,normalize them

new_gt_imgs=[]
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

X_batch=np.asarray(X_batch)
Y_batch=np.asarray(Y_batch)
#convert to torch tensor
X_batch=torch.from_numpy(X_batch).float()
Y_batch=torch.from_numpy(Y_batch).long()
model=Classification.train_FCN8s(X_batch,Y_batch)
print("training done")

#test the model 
X=imgs_fortrain
Y=gt_imgs_fortrain
X_batch=[]
Y_batch=[]
for i in range(0,len(X),batch_size):
  X_batch.append(X[i:i+batch_size])
  Y_batch.append(Y[i:i+batch_size])
X_batch=np.asarray(X_batch)
Y_batch=np.asarray(Y_batch)
X_batch=torch.from_numpy(X_batch).float()
Y_batch=torch.from_numpy(Y_batch).long()
total=0
count=0
with torch.no_grad():
   for i in range(len(X_batch)):
    out_image_batches=model(X_batch[i])
    for j in range(len(out_image_batches)):
      out_image=out_image_batches[j]
      out_image=torch.argmax(out_image,dim=0)
      out_image=out_image.numpy()
      patch_size=16
      h,w=out_image.shape
      #输出out_imge中的最大值
      test_image=Y_batch[i][j].numpy()

     
      for k in range(0,h,patch_size):
        for l in range(0,w,patch_size):
          patch=test_image[k:k+patch_size,l:l+patch_size]
          if np.mean(patch)>0.5:
           test_image[k:k+patch_size,l:l+patch_size]=1
          else:
           test_image[k:k+patch_size,l:l+patch_size]=0
      gt = test_image
      acc = np.mean(out_image == gt)
      total += acc
      count += 1
total/=count
print("Accuracy: ",total)

img1=imgs[1]
img1=torch.from_numpy(img1).float()
img1=img1.unsqueeze(0)
out_image=model(img1)
out_image=torch.argmax(out_image,dim=1)
out_image=out_image.squeeze(0)
out_image=out_image.numpy()
# 确保图像是2D数组
out_image = out_image.astype(np.uint8)  # 转换为uint8类型
plt.imsave("training/groundtruth/satImage_001_01.png", out_image, cmap="gray")
            
            


image_dir="test_set_images/"
files = os.listdir(image_dir)
with torch.no_grad():
 with open("submission5.csv", "w") as f:
  f.writelines("id,prediction\n")
  num_files=len(files)
  for i in range(num_files):
    file_name="test_"+str(i+1)+"/"
    file_name2="test_"+str(i+1)+".png"
    img=Data_processing.load_image(image_dir + file_name+file_name2)
    img=torch.from_numpy(img).float()

    #add batch dimension
    img=img.unsqueeze(0)
    img=img/255
    out_image=model(img)
    out_image=torch.argmax(out_image,dim=1)
    out_image=out_image.squeeze(0)
    out_image=out_image.numpy()
    # we need to assign same values for each 16*16 patches
    #foreground threshold
    #conver to uint8 and normalize to 0-1
    w,h=out_image.shape
    patch_means=[]
    for k in range(0,w,16):
      for l in range(0,h,16):
        patch=out_image[k:k+16,l:l+16]
        patch_means.append(np.mean(patch))
    patch_means.sort()
    #set 70% to be 0 and 30% to be 1 in the sorted patch_means
    threshold=patch_means[int(len(patch_means)*0.3)]

    for k in range(0,w,16):
      for l in range(0,h,16):
        patch=out_image[k:k+16,l:l+16]
        if np.mean(patch)>threshold:
          out_image[k:k+16,l:l+16]=1
        else:
          out_image[k:k+16,l:l+16]=0
    
    plt.imsave("test_set_images/" + file_name + file_name2 + "_result5.png", out_image,cmap="gray")
    f.writelines("{}\n".format(s) for s in Data_processing.mask_to_submission_strings("test_set_images/" + file_name + file_name2 + "_result5.png"))

