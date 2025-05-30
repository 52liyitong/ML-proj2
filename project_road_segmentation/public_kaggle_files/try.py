import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN8sPatchClassifier(nn.Module):
    def __init__(self):
        super(FCN8sPatchClassifier, self).__init__()

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

img=torch.randn(1,608,608,3)
model=FCN8sPatchClassifier()
output=model(img)
print(output.shape)
