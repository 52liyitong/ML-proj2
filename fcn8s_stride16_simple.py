#in this file we will use FCN model to segment the road
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from PIL import Image
import Data_processing
import Classification2
import torch
root_dir = "training1/"
image_dir = root_dir + "images/"
files = os.listdir(image_dir)
#randomly split the data into training set and test set
n =100
train_index = np.random.choice(n, size=int(n*0.8), replace=False)
test_index = np.setdiff1d(np.arange(n), train_index)


print(train_index)

imgs = [Data_processing.load_image(image_dir + files[i]) for i in train_index]
imgs_fortrain = [Data_processing.load_image(image_dir + files[i]) for i in test_index]
gt_dir = root_dir + "groundtruth/"

gt_imgs = [Data_processing.load_image(gt_dir + files[i]) for i in train_index]
gt_imgs_fortrain = [Data_processing.load_image(gt_dir + files[i]) for i in test_index]

#data augmentation
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


#output the maxvalue of imgs[0]

new_gt_imgs=Data_processing.croped_img(gt_imgs)
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
model=Classification2.train_fcn8s_patch_stride16_simple(X_batch,Y_batch)
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
with torch.no_grad():
  mlou_total = 0
  for i in range(len(X_batch)):
    out_image_batches = model(X_batch[i])
    for j in range(len(out_image_batches)):
      out_image = out_image_batches[j]
      out_image = torch.argmax(out_image, dim=0)
      out_image = out_image.numpy()
      test_image = Y_batch[i][j].numpy()
      gt = test_image
      acc = np.mean(out_image == gt)
      total += acc
      # mIoU calculation
      intersection = np.logical_and(out_image == 1, gt == 1).sum()
      union = np.logical_or(out_image == 1, gt == 1).sum()
      if union == 0:
        iou = 1.0 if intersection == 0 else 0.0
      else:
        iou = intersection / union
      mlou_total += iou
      count += 1
total /= count
mlou = mlou_total / count
print("Accuracy: ", total)
print("mIoU: ", mlou)

            
            


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
    img=img/255
    #add batch dimension
    img=img.unsqueeze(0)
    out_image=model(img)
    out_image=torch.argmax(out_image,dim=1)
    out_image=out_image.squeeze(0)
    out_image=out_image.numpy()
    #convet back to 608*608
    new_out_image=np.zeros((608,608))
    for i in range (38):
      for j in range (38):
        new_out_image[i*16:(i+1)*16,j*16:(j+1)*16]=out_image[i,j]
    out_image=new_out_image
    
    plt.imsave("test_set_images/" + file_name + file_name2 + "_result5.png", out_image,cmap="gray")
    f.writelines("{}\n".format(s) for s in Data_processing.mask_to_submission_strings("test_set_images/" + file_name + file_name2 + "_result5.png"))

