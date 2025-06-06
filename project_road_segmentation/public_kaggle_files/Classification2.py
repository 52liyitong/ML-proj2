import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN8sPatchClassifier_stride16(nn.Module):
    def __init__(self):
        super(FCN8sPatchClassifier_stride16, self).__init__()

        # 使用 stride=16 的第一层卷积
        self.init_conv = nn.Conv2d(3, 64, kernel_size=3, stride=16, padding=1)  # (608,608) → (38,38)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        # 再堆几层浅层卷积处理每个 patch 的特征
        self.conv_block = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 2, kernel_size=1)  # 输出通道数 = 类别数（二分类）
        )

    def forward(self, x):
        # 输入 x: (B, 608, 608, 3)
        x = x.permute(0, 3, 1, 2)  # → (B, 3, 608, 608)
        x = self.relu1(self.bn1(self.init_conv(x)))  # → (B, 64, 38, 38)
        out = self.conv_block(x)  # → (B, 2, 38, 38)
        return out

def train_fcn8s_patch(X_batch, Y_batch, num_epochs=4):
    model = FCN8sPatchClassifier_stride16()
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

import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN8sPatchClassifier_deep(nn.Module):
    def __init__(self):
        super(FCN8sPatchClassifier_deep, self).__init__()

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)  # 608 → 304
        )
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)  # 304 → 152
        )
        # Block 3
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
            nn.MaxPool2d(2, stride=2)  # 152 → 76
        )
        # Block 4
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
            nn.MaxPool2d(2, stride=2)  # 76 → 38
        )
        # Block 5
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
            nn.MaxPool2d(2, stride=2)  # 38 → 19
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7, padding=3),  # 19×19
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(4096, 2, 1)  # 输出通道 = 类别数
        )

        # Skip score layers
        self.score_pool4 = nn.Conv2d(512, 2, 1)
        self.score_pool3 = nn.Conv2d(256, 2, 1)

        # Upsample layers
        self.upscore2 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)  # 19 → 38
        self.upscore4 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)  # 38 → 76
        self.upscore8 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)  # 76 → 152

        # Final upsampling to 38×38
        self.up_to_38 = nn.ConvTranspose2d(2, 2, kernel_size=16, stride=4, padding=6, bias=False)  # 152 → 608 → pool

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # (B, 3, 608, 608)
        x1 = self.block1(x)  # 304
        x2 = self.block2(x1)  # 152
        x3 = self.block3(x2)  # 76
        x4 = self.block4(x3)  # 38
        x5 = self.block5(x4)  # 19

        score = self.classifier(x5)  # (B, 2, 19, 19)
        score = self.upscore2(score)  # (B, 2, 38, 38)

        score4 = self.score_pool4(x4)  # (B, 2, 38, 38)
        score = score + score4

        score = self.upscore4(score)  # (B, 2, 76, 76)
        score3 = self.score_pool3(x3)  # (B, 2, 76, 76)
        score = score + score3

        score = self.upscore8(score)  # (B, 2, 152, 152)
        score = self.up_to_38(score)  # (B, 2, 608, 608) → 再平均池化
        out = F.adaptive_avg_pool2d(score, (38, 38))  # → (B, 2, 38, 38)

        return out

def train_fcn8s_patch_deep(X_batch, Y_batch, num_epochs=6):
    model = FCN8sPatchClassifier_deep()
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

