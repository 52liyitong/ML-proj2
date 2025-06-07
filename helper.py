import os
import numpy as np
from PIL import Image
root_dir = "training/"
image_dir = root_dir + "images/"
files = os.listdir(image_dir)
#randomly split the data into training set and test set
n = len(files)
train_index = np.random.choice(n, size=int(n*0.8), replace=False)
test_index = np.setdiff1d(np.arange(n), train_index)
#for each image we want to change the size from 25*25 crops to 38*38 crops
# 定义新的图像尺寸
gt_dir = root_dir + "groundtruth/"
new_size = (608, 608) # 38*16 = 608

# 调整训练集图像尺寸
for i in range(len(files)):
    # 读取原始图像
    img_path = image_dir + files[i]
    gt_path = gt_dir + files[i]
    
    # 使用PIL打开图像并调整大小
    img = Image.open(img_path)
    gt = Image.open(gt_path)
    
        # 使用BILINEAR插值调整训练图像
    img = img.resize(new_size, Image.BILINEAR)
    # 使用NEAREST插值调整标签图像以保持二值性质
    gt = gt.resize(new_size, Image.NEAREST)
    img_path="training1/images/"+files[i]
    gt_path="training1/groundtruth/"+files[i]
    img.save(img_path)
    gt.save(gt_path)