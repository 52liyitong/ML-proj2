import matplotlib.image as mpimg
import numpy as np
import torch
import matplotlib.pyplot as plt
import os, sys
from PIL import Image
import Data_processing
import Classification
from sklearn import linear_model
def logistic_regression(X_batch,Y_batch):
 logreg = linear_model.LogisticRegression(C=1e5, class_weight="balanced")
 logreg.fit(X_batch, Y_batch)
 return logreg
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第一个卷积层：使用3x3的kernel，因为16x16的patch较小
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)  # 输出: 16x16x16
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.pooling1 = torch.nn.MaxPool2d(2, 2)  # 输出: 8x8x16
 
        
        # 第二个卷积层：使用3x3的kernel
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 输出: 8x8x32
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.pooling2 = torch.nn.MaxPool2d(2, 2)  # 输出: 4x4x32

        
        # 第三个卷积层：使用3x3的kernel
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 输出: 4x4x64
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.pooling3 = torch.nn.MaxPool2d(2, 2)  # 输出: 2x2x64
  
        
        # 全连接层
        self.fc1 = torch.nn.Linear(64 * 2 * 2, 128)  # 2x2x64 = 256
        self.fc2 = torch.nn.Linear(128, 2)
        
    def forward(self, x):
        # 调整输入维度顺序
        x = x.permute(0, 3, 1, 2)  # [batch, channels, height, width]
        
        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pooling1(x)
      
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pooling2(x)
        
        
        # 第三个卷积块
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pooling3(x)
       
        
        # 展平
        x = x.reshape(-1, 64 * 2 * 2)
        
        # 全连接层
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
def train_CNN(X_batch,Y_batch):
    model=CNN()
    criterion=torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([0.7, 1.8]))
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    for epoch in range(4):
        for i in range(len(X_batch)):
            optimizer.zero_grad()
            output=model(X_batch[i])
            loss=criterion(output,Y_batch[i])
            loss.backward()
            optimizer.step()
    return model

class Transformer(torch.nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        #X_batch have batch_size,16,16,3
        #we want to use vision transformer to classify the image,for each 16,16,3 image, we want to divide it into 16 patches,and then use the transformer to classify the image

        

        
class FCN8s(torch.nn.Module):
    def __init__(self):
        super(FCN8s, self).__init__()
        #the input is a image with 16*38,16*38,3 img
        #we first croped the image into 16,16,3 patches , and we want to assing 0,1 to the each patch
        #we want to use FCN to classify the image
        #we want to use the FCN8s model to classify the image
        self.block1=torch.nn.Sequential(
            torch.nn.Conv2d(3,64,kernel_size=3,padding=1),#the shape is 608,608,64
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,64,kernel_size=3,padding=1),#the shape is 608,608,64
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),#the shape is 304,304,64
            
        )
        self.block2=torch.nn.Sequential(
            torch.nn.Conv2d(64,128,kernel_size=3,padding=1),#the shape is 304,304,128
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128,128,kernel_size=3,padding=1),#the shape is 304,304,128
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),#the shape is 152,152,128
        )
        self.block3=torch.nn.Sequential(
            torch.nn.Conv2d(128,256,kernel_size=3,padding=1),#the shape is 152,152,256
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256,256,kernel_size=3,padding=1),#the shape is 152,152,256
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256,256,kernel_size=3,padding=1),#the shape is 152,152,256
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),#the shape is 76,76,256
        )
        self.block4=torch.nn.Sequential(
            torch.nn.Conv2d(256,512,kernel_size=3,padding=1),#the shape is 76,76,512
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512,512,kernel_size=3,padding=1),#the shape is 76,76,512
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512,512,kernel_size=3,padding=1),#the shape is 76,76,512
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),#the shape is 38,38,512
        )
        self.block5=torch.nn.Sequential(
            torch.nn.Conv2d(512,512,kernel_size=3,padding=1),#the shape is 38,38,512
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512,512,kernel_size=3,padding=1),#the shape is 38,38,512
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),    
            torch.nn.Conv2d(512,512,kernel_size=3,padding=1),#the shape is 38,38,512
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),#the shape is 19,19,512
        )
        self.block6=torch.nn.Sequential(
            torch.nn.Conv2d(512,4096,kernel_size=3,padding=1),#the shape is 19,19,4096
            torch.nn.BatchNorm2d(4096),
            torch.nn.ReLU(),
        )
        self.block7=torch.nn.Sequential(
            torch.nn.Conv2d(4096,4096,kernel_size=1),#the shape is 19,19,4096
            torch.nn.BatchNorm2d(4096),
            torch.nn.ReLU(),
        )

        self.upscore_pool5 = torch.nn.Sequential(
            torch.nn.Conv2d(4096, 2, 1), #the shape is 19,19,2
            torch.nn.ConvTranspose2d(2, 2, 2, 2, bias=False)
        )
        self.score_pool4 = torch.nn.Conv2d(512, 2, 1, bias=False)
        self.score_pool3 = torch.nn.Conv2d(256, 2, 1, bias=False)
        self.upscore_pool4 = torch.nn.ConvTranspose2d(2, 2, 2, 2, bias=False)
        self.upscore_pool = torch.nn.ConvTranspose2d(2, 2, 8, 8, bias=False)
                        
    def forward(self,x):
        x=x.permute(0, 3, 1, 2)
        x1=self.block1(x)
        x2=self.block2(x1)
        x3=self.block3(x2)
        x4=self.block4(x3)
        x5=self.block5(x4)
        x6=self.block6(x5)
        x7=self.block7(x6)
        pool5=self.upscore_pool5(x7)
        pool4=self.score_pool4(x4)
        pool3=self.score_pool3(x3)
        pool4=pool4+pool5
        pool4=self.upscore_pool4(pool4)
        pool3=pool3+pool4
        pool3=self.upscore_pool(pool3)
        return pool3

def train_FCN8s(X_batch,Y_batch):
    model=FCN8s()
    criterion=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    for epoch in range(1):
        for i in range(len(X_batch)):
            optimizer.zero_grad()
            output=model(X_batch[i])
            loss=criterion(output,Y_batch[i])
            loss.backward()
            optimizer.step()
    return model

        

        