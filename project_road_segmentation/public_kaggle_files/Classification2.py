import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN8sPatchClassifier_stride16_simple(nn.Module):
    def __init__(self):
        super(FCN8sPatchClassifier_stride16_simple, self).__init__()

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

def train_fcn8s_patch_stride16_simple(X_batch, Y_batch, num_epochs=4):
    model = FCN8sPatchClassifier_stride16_simple()
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



class FCN8sPatchClassifier_stride16(nn.Module):
    def __init__(self):
        super(FCN8sPatchClassifier_stride16, self).__init__()

        # Step-down convolution: directly downsample 608→38
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=16, padding=1),  # 608→38
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Deep feature extraction (remain at 38×38)
        self.backbone = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(256, 2, kernel_size=1)
        )

        # Skip layers for downsampled inputs
        self.skip76 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=1)
        )
        self.skip152 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=1)
        )

        # Upsampling and fusion
        self.upscore_from_38 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)   # 38→76
        self.upscore_from_76 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)   # 76→152
        self.upscore_from_152 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)  # 152→608


    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # (B, 3, 608, 608)

        # Downsample input manually for skip connections
        x_76 = F.interpolate(x, size=(76, 76), mode='bilinear', align_corners=False)
        x_152 = F.interpolate(x, size=(152, 152), mode='bilinear', align_corners=False)

        # Stride-16 encoding
        x_init = self.initial(x)       # (B, 64, 38, 38)
        x_feat = self.backbone(x_init) # (B, 512, 38, 38)
        out = self.classifier(x_feat)  # (B, 2, 38, 38)

        # Skip connection fusion
        up_38 = self.upscore_from_38(out)                     # → 76×76
        skip_76 = self.skip76(x_76)                           # (B, 2, 76, 76)
        fuse_76 = up_38 + skip_76

        up_76 = self.upscore_from_76(fuse_76)                # → 152×152
        skip_152 = self.skip152(x_152)                       # (B, 2, 152, 152)
        fuse_152 = up_76 + skip_152

        up_final = self.upscore_from_152(fuse_152)           # → 608×608

        final_out = F.adaptive_avg_pool2d(up_final, (38, 38))  # (B, 2, 38, 38)
        return final_out


def train_fcn8s_patch_stride16_complex(X_batch, Y_batch, num_epochs=6):
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

