import torch
import torch.nn as nn
from transformers import SegformerModel, SegformerConfig
from transformers import SegformerForSemanticSegmentation

class CustomSegformer(SegformerForSemanticSegmentation):
    def __init__(self, config):
        super().__init__(config)
        self.upsample = nn.Upsample(size=(38, 38), mode='bilinear', align_corners=False)
    
    def forward(self, pixel_values, labels=None):
        outputs = super().forward(pixel_values=pixel_values, labels=labels)
        if hasattr(outputs, 'logits'):
            outputs.logits = self.upsample(outputs.logits)
        return outputs

def train_segformer_patch(X_batch, Y_batch, num_epochs=10):
    # 加载预训练模型
    config = SegformerConfig.from_pretrained(
        '/home/w/Downloads/config.json',
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    model = CustomSegformer.from_pretrained(
        '/home/w/Downloads/pytorch_model.bin',
        config=config,
        ignore_mismatched_sizes=True
    )
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(len(X_batch)):
            optimizer.zero_grad()
            imgs = X_batch[i]
            imgs = imgs.permute(0, 3, 1, 2)  # [B, C, H, W]
            labels = Y_batch[i].squeeze(1)   # [B, H, W]
            outputs = model(pixel_values=imgs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(X_batch)}")
    
    return model

            
            
