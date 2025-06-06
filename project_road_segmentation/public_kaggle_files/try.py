import torch
import torch.nn as nn
from transformers import SegformerModel, SegformerConfig
from transformers import SegformerImageProcessor

class SegFormerPatchClassifier(nn.Module):
    def __init__(self, num_classes=2, backbone_name="nvidia/segformer-b0-finetuned-ade-512-512"):
        super().__init__()
        # 1. Load pre-trained SegFormer encoder with output_hidden_states=True
        config = SegformerConfig.from_pretrained(backbone_name, output_hidden_states=True)
        self.encoder = SegformerModel.from_pretrained(backbone_name, config=config)
        
        # 2. Final layer projects encoder output to num_classes
        hidden_size = self.encoder.config.hidden_sizes[-1]  # e.g., 256
        self.linear = nn.Conv2d(hidden_size, num_classes, kernel_size=1)
        self.upsample = nn.Upsample(size=(38, 38), mode='bilinear', align_corners=False)

    def forward(self, x):  # x: [B, H, W, C]
        # 确保输入维度正确 [B, C, H, W]
        if x.shape[1] != 3:  # 如果通道数不在第二维
            x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
            
        outputs = self.encoder(x)
        hidden = outputs.last_hidden_state  # [B, C, H, W]
        
        # 打印形状以进行调试
        print("Input shape after permute:", x.shape)
        print("Hidden state shape:", hidden.shape)
        
        # 直接使用卷积层进行分类
        logits = self.linear(hidden)  # [B, num_classes, H, W]
        logits = self.upsample(logits)  # [B, num_classes, 38, 38]
        
        return logits

# 测试代码
if __name__ == "__main__":
    # 创建测试输入 [batch_size, height, width, channels]
    img = torch.randn(4, 608, 608, 3)  # 修改为4个样本的批次
    model = SegFormerPatchClassifier()
    
    # 前向传播
    output = model(img)
    print(f"Input shape: {img.shape}")
    print(f"Output shape: {output.shape}")

