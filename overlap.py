import Data_processing
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

####################################################################################
### Function to overlay mask on image，help to visualize the segmentation results###
####################################################################################

def overlap_images(img1, img2, alpha=0.5):
    """
    Overlay img2 mask as red color on img1.
    
    Args:
        img1: RGB image, numpy array, shape (H, W, 3), uint8 [0-255]
        img2: mask image, numpy array, shape (H, W) or (H, W, 1), binary or grayscale
        alpha: transparency for the red overlay (0~1)
        
    Returns:
        result: RGB image with red mask overlay
    """

    if len(img2.shape) == 3:
        img2 = img2[:, :, 0]

    mask = img2.astype(bool)  
   
    # 复制img1，防止修改原图
    result = img1*255

    result[mask, 0] = (1 - alpha) * result[mask, 0] + alpha * 255  # 红色通道
    result[mask, 1] = (1 - alpha) * result[mask, 1] + alpha * 0    # 绿色通道
    result[mask, 2] = (1 - alpha) * result[mask, 2] + alpha * 0    # 蓝色通道
    
    return result.astype(np.uint8)

# Example usage:change the file_name and img2 path to your own,replace 18 with the image number you want to test
#input your image file name here
number=input("input the image number you want to see in the test set:")
number1=input("input the model you want to use:(1 is logistic regression, 2 is CNN, 3 is FCN_simple, 4 is FCN, 5 is FCN_stride16_simple, 6 is FCN_stride16, 7 is Segformer):")
number=int(number)
number1=int(number1)
file_name="test_"+str(number)
file_name+="/"+file_name+".png"
img=Data_processing.load_image("test_set_images/"+file_name)
img2=Data_processing.load_image("test_set_images/test_"+str(number)+"/test_"+str(number)+".png_result"+str(number1)+".png")


overlap_images=overlap_images(img,img2)
plt.imsave("visualization.png", overlap_images)


