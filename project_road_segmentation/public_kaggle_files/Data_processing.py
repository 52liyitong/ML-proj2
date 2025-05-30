#in this file we will process the data for the classification
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from PIL import Image
import cv2
import re
import torch

def accuracy(img1,img2):
   acc=np.sum(img1==img2)/len(img1)
   return acc

def load_image(infilename):
    #load the image
    data = mpimg.imread(infilename)
    return data

def normalize_images(imgs):
    #normalize the images
    return [img for img in imgs]

def img_float_to_uint8(img):
    #convert the image to uint8
    if hasattr(img, 'numpy'):  # 如果是PyTorch Tensor
        rimg = img - img.min()
        rimg = (rimg / rimg.max()).round().to(torch.uint8)
    else:  # 如果是NumPy数组
        rimg = img - np.min(img)
        rimg = (rimg / np.max(rimg)).round().astype(np.uint8)
    return rimg

def img_crop(im, w, h):
    #crop the image into patches,in our lab we set w,h to be 16,16
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j : j + w, i : i + h]
            else:
                im_patch = im[j : j + w, i : i + h, :]
            list_patches.append(im_patch)
    return list_patches

def extract_features(img):
    feat_m = np.mean(img, axis=(0, 1))
    feat_v = np.var(img, axis=(0, 1))
    feat = np.append(feat_m, feat_v)
    return feat

def extract_features_2d(img):
    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    return feat

def extract_img_features(filename, patch_size):
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, patch_size)
    X = np.asarray(
        [extract_features(img_patches[i]) for i in range(len(img_patches))]
    )
    return X


def extract_img_features_2d(filename, patch_size):
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, patch_size)
    X = np.asarray(
        [extract_features_2d(img_patches[i]) for i in range(len(img_patches))]
    )
    return X

def value_to_class(v,foreground_threshold):
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0
    
def extract_values(filename, patch_size,foreground_threshold):
    img = load_image(filename)
    gt_patches = img_crop(img, patch_size, patch_size)
    Y = np.asarray(
        [value_to_class(np.mean(gt_patches[i]),foreground_threshold) for i in range(len(gt_patches))]
    )
    return Y
    


def image_crop_with_data_augmentation(im):
    a=np.random.rand()
    if a < 0.33:
        im_patch = np.rot90(im)
    elif a < 0.66:
        im_patch = np.flip(im)
    else:
        #we first choose the 8*8 patch,then we do oversampling
        #generate random center position
        w=im.shape[0]
        h=im.shape[1]
        center_x = np.random.randint(0,w-8)
        center_y = np.random.randint(0,h-8)
        im_patch = im[center_x:center_x+8,center_y:center_y+8]
        im_patch = cv2.resize(im_patch, (16, 16), interpolation=cv2.INTER_LINEAR)
        #oversampling,change the 8*8*3 patch to 16*16*3 patch

        
                

    return im_patch
    
def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            im[j : j + w, i : i + h] = labels[idx]
            idx = idx + 1
    return im
        
def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = predicted_img * 255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, "RGB").convert("RGBA")
    overlay = Image.fromarray(color_mask, "RGB").convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img
    
def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i : i + patch_size, j : j + patch_size]
            label = patch_to_label(patch)
            yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))

def mask_to_submission_strings1(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    im=im/255
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i : i + patch_size, j : j + patch_size]
            label = patch_to_label(patch)
            yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))

def masks_to_submission(submission_filename, *image_filenames,csv_filename):
    """Converts images into a submission file"""
    #store the result in the csv_filename
    with open(csv_filename, "w") as f:
        for fn in image_filenames[0:]:
            f.writelines("{}\n".format(s) for s in mask_to_submission_strings(fn))

foreground_threshold = 0.25
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0

def img_transform(img):
    # after FCN, the image has too high frequency, we need to smooth it,the size is 608*608 and is Gray
    img=cv2.GaussianBlur(img,(5,5),0)
    return img

def croped_img(gt_imgs):
    #convert the 608*608 image to 16*16 image
    new_gt_imgs=[]
    for img in gt_imgs:
        w,h=img.shape
        new_img=np.zeros((w//16,h//16))
        for i in range(0,w,16):
            for j in range(0,h,16):
                patch=img[i:i+16,j:j+16]
                new_img[i//16,j//16]=patch_to_label(patch)
        new_gt_imgs.append(new_img)
    return new_gt_imgs

def mirror_image(img):
    #mirror the image
    img=np.flip(img)
    return img

def rotate_image(img):
    #rotate the image 90 degrees
    img=np.rot90(img)
    return img


