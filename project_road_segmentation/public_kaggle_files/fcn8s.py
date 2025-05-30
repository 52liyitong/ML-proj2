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
n =60
train_index = np.random.choice(n, size=int(n*0.8), replace=False)
test_index = np.setdiff1d(np.arange(n), train_index)

#check if train_index and test_index have same elements,if yes,out put the index



imgs = [Data_processing.load_image(image_dir + files[i]) for i in train_index]
imgs_fortrain = [Data_processing.load_image(image_dir + files[i]) for i in test_index]
gt_dir = root_dir + "groundtruth/"

gt_imgs = [Data_processing.load_image(gt_dir + files[i]) for i in train_index]
gt_imgs_fortrain = [Data_processing.load_image(gt_dir + files[i]) for i in test_index]


#for all imgs,normalize them

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
model=Classification.train_fcn8s_patch(X_batch,Y_batch)
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
   for i in range(len(X_batch)):
    out_image_batches=model(X_batch[i])
    for j in range(len(out_image_batches)):
      out_image=out_image_batches[j]
      out_image=torch.argmax(out_image,dim=0)
      out_image=out_image.numpy()

      #输出out_imge中的最大值
      test_image=Y_batch[i][j].numpy()
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
 with open("submission6.csv", "w") as f:
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
    #convet back to 608*608
    new_out_image=np.zeros((608,608))
    for i in range (38):
      for j in range (38):
        new_out_image[i*16:(i+1)*16,j*16:(j+1)*16]=out_image[i,j]
    out_image=new_out_image
    
    plt.imsave("test_set_images/" + file_name + file_name2 + "_result6.png", out_image,cmap="gray")
    f.writelines("{}\n".format(s) for s in Data_processing.mask_to_submission_strings("test_set_images/" + file_name + file_name2 + "_result6.png"))

