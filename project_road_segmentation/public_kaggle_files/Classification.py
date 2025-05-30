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
    for epoch in range(4):
        for i in range(len(X_batch)):
            optimizer.zero_grad()
            output=model(X_batch[i])
            loss=criterion(output,Y_batch[i])
            loss.backward()
            optimizer.step()
    return model

import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNPatchClassifier(nn.Module):
    def __init__(self):
        super(FCNPatchClassifier, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 304x304
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 152x152
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 76x76
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 38x38
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 2, kernel_size=1)  # 输出每个 patch 的 2 类得分（不上采样）
        )

    def forward(self, x):
        # x: (B, 608, 608, 3) → permute to (B, 3, 608, 608)
        x = x.permute(0, 3, 1, 2)

        x = self.block1(x)  # → 304x304
        x = self.block2(x)  # → 152x152
        x = self.block3(x)  # → 76x76
        x = self.block4(x)  # → 38x38
        x = self.classifier(x)  # → (B, 2, 38, 38)

        return x

# 训练函数
def train_FCN_PatchClassifier(X_batch, Y_batch, num_epochs=4):
    model = FCNPatchClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(num_epochs):
        for i in range(len(X_batch)):
            optimizer.zero_grad()
            output = model(X_batch[i])  # (1, 3, 608, 608)
            loss = criterion(output, Y_batch[i])  # Y: (1, 38, 38)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Sample {i}, Loss: {loss.item():.4f}")

    return model


class FCN8sPatchClassifier_complex(nn.Module):
    def __init__(self):
        super(FCN8sPatchClassifier_complex, self).__init__()

        # Feature Extractor (类似 VGG16)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 608 -> 304
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 304 -> 152
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 152 -> 76
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 76 -> 38
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 38 -> 19
        )

        # Classifier layers (FC → conv)
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),  # 19x19
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(4096, 2, kernel_size=1)  # 输出通道 = 类别数
        )

        # Skip layers (score)
        self.score_pool4 = nn.Conv2d(512, 2, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, 2, kernel_size=1)

        # Upsample layers
        self.upscore2 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)  # 19 → 38
        self.upscore4 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)  # 38 → 76
        self.upscore8 = nn.ConvTranspose2d(2, 2, kernel_size=8, stride=8, padding=0, bias=False)  # 76 → 608

    def forward(self, x):
        # Input: (B, 608, 608, 3)
        x = x.permute(0, 3, 1, 2)  # → (B, 3, 608, 608)

        x1 = self.block1(x)  # 304x304
        x2 = self.block2(x1)  # 152x152
        x3 = self.block3(x2)  # 76x76 → for skip3
        x4 = self.block4(x3)  # 38x38 → for skip4
        x5 = self.block5(x4)  # 19x19

        score = self.classifier(x5)  # (B, 2, 19, 19)
        score = self.upscore2(score)  # (B, 2, 38, 38)

        score4 = self.score_pool4(x4)  # (B, 2, 38, 38)
        score = score + score4

        score = self.upscore4(score)  # (B, 2, 76, 76)
        score3 = self.score_pool3(x3)  # (B, 2, 76, 76)
        score = score + score3

        score = self.upscore8(score)  # (B, 2, 608, 608)

        # 最后 reshape 到每个 16x16 patch 输出一个值
        # (B, 2, 608, 608) → (B, 2, 38, 38) 通过 pooling
        output = F.adaptive_avg_pool2d(score, (38, 38))  # 平均池化代替明确切块分类

        return output

def train_fcn8s_patch(X_batch, Y_batch, num_epochs=4):
    model = FCN8sPatchClassifier_complex()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(len(X_batch)):
            x = X_batch[i]  # (1, 608, 608, 3)
            y = Y_batch[i]  # (1, 38, 38)

            optimizer.zero_grad()
            output = model(x)  # (1, 2, 38, 38)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

    return model