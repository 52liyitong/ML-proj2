import torch 
from torch import nn 

# 定义卷积层
 # 输入通道4096，输出通道2，kernel_size=1
tconv = nn.ConvTranspose2d(2, 2, 8, 8,bias=False)

# 创建输入张量 [batch_size, channels, height, width]
input = torch.randn(1, 2, 76, 76)  # 添加batch维度，调整通道顺序

# 前向传播


# 转置卷积
tinput = tconv(input)
print('经过逆卷积后的维度:', tinput.shape)  # (1, 2, 38, 38)