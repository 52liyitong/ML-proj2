#in this file we will use FCN model to segment the road
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from PIL import Image
import Data_processing
import Classification3
import torch
import torch.nn.functional as F

root_dir = "training1/"
image_dir = root_dir + "images/"
files = os.listdir(image_dir)
#randomly split the data into training set and test set
n =100
train_index = np.random.choice(n, size=int(n*0.8), replace=False)
test_index = np.setdiff1d(np.arange(n), train_index)


imgs = [Data_processing.load_image(image_dir + files[i]) for i in train_index]
imgs_fortrain = [Data_processing.load_image(image_dir + files[i]) for i in test_index]
gt_dir = root_dir + "groundtruth/"

gt_imgs = [Data_processing.load_image(gt_dir + files[i]) for i in train_index]
gt_imgs_fortrain = [Data_processing.load_image(gt_dir + files[i]) for i in test_index]

##########################
##data augmentation here##
##########################
#if you want to do data augmentation, you can change the size of index_list,in our experiment, we set it to 0.1*len(train_index)
index_list=np.random.choice(len(train_index),size=int(len(train_index)*0),replace=False)
for i in index_list:
  img=Data_processing.load_image(image_dir + files[train_index[i]])
  z=np.random.rand()
  if z<0.5:
    img=Data_processing.mirror_image(img)
    imgs.append(img)
    gt_img=Data_processing.load_image(gt_dir + files[train_index[i]])
    gt_img=Data_processing.mirror_image(gt_img)
    gt_imgs.append(gt_img)
  else:
    img=Data_processing.rotate_image(img)
    imgs.append(img)
    gt_img=Data_processing.load_image(gt_dir + files[train_index[i]])
    gt_img=Data_processing.rotate_image(gt_img)
    gt_imgs.append(gt_img)
print(len(imgs))


new_gt_imgs=Data_processing.croped_img(gt_imgs)
print(np.max(new_gt_imgs[0]))
print("ok")
new_gt_imgs_fortrain=Data_processing.croped_img(gt_imgs_fortrain)




foreground_threshold = 0.25 

#create training set
#case2: CNN
#we need to load the data again

batch_size=4
X=imgs
Y=new_gt_imgs
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
model=Classification3.train_segformer_patch(X_batch,Y_batch)
print("training done")

#test the model 
X=imgs_fortrain
Y=new_gt_imgs_fortrain
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
total = 0.0
count = 0

with torch.no_grad():
  model.eval()
  iou_total = 0.0
  for i in range(len(X_batch)):
    imgs = X_batch[i].permute(0, 3, 1, 2)  # [B, C, H, W]
    out = model(pixel_values=imgs)
    logits = out.logits  # [B, C, H, W]

    for j in range(logits.shape[0]):
      out_image = logits[j]  # [C, H, W]
      out_image = torch.argmax(out_image, dim=0)  # [H, W]

      out_image_np = out_image.cpu().numpy()
      gt = Y_batch[i][j].squeeze(0).cpu().numpy()  # [H, W]
      acc = np.mean(out_image_np == gt)
      total += acc
      count += 1

      # MIoU calculation (for binary segmentation: 0-background, 1-foreground)
      intersection = np.logical_and(out_image_np == 1, gt == 1).sum()
      union = np.logical_or(out_image_np == 1, gt == 1).sum()
      iou = intersection / union if union != 0 else 1.0
      iou_total += iou

total /= count
miou = iou_total / count
print("Accuracy:", total)
print("MIoU:", miou)


            
        

image_dir="test_set_images/"
files = os.listdir(image_dir)
with torch.no_grad():
 with open("submission7.csv", "w") as f:
  f.writelines("id,prediction\n")
  num_files=len(files)
  for i in range(num_files):
    file_name="test_"+str(i+1)+"/"
    file_name2="test_"+str(i+1)+".png"
    img=Data_processing.load_image(image_dir + file_name+file_name2)
    img=torch.from_numpy(img).float()
    #add batch dimension
    img=img.unsqueeze(0)
    img=img.permute(0,3,1,2)
    out_image = model(img)
    out_image=torch.argmax(out_image.logits,dim=1)
    out_image=out_image.squeeze(0)
    out_image=out_image.numpy()
    #convet back to 608*608
    new_out_image=np.zeros((608,608))
    for i in range (38):
      for j in range (38):
        new_out_image[i*16:(i+1)*16,j*16:(j+1)*16]=out_image[i,j]
    out_image=new_out_image
    
    plt.imsave("test_set_images/" + file_name + file_name2 + "_result7.png", out_image,cmap="gray")
    f.writelines("{}\n".format(s) for s in Data_processing.mask_to_submission_strings("test_set_images/" + file_name + file_name2 + "_result7.png"))

