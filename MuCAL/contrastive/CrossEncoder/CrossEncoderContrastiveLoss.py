import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEncoderContrastiveLoss(nn.Module):
    def __init__(self):
        super(CrossEncoderContrastiveLoss, self).__init__()
        self.temperature = 0.2
    
    def forward(self, logits, batch_size, num_lang, device='cuda'):
        # # 将关键计算转换为fp32
        logits = logits.float()  # 转换为fp32
        
        # Reshape logits to (batch_size * num_lang, batch_size)
        logits = logits.view(batch_size * num_lang, batch_size)
        
        # Sum similarities over all languages for the same instance
        similarities = logits.view(batch_size, num_lang, batch_size).mean(dim=1)  # batch_size x batch_size
        
        # 更稳健的数值稳定化
        similarities = similarities - similarities.max(dim=1, keepdim=True)[0]  # 先减去最大值
        
        # 应用温度系数
        similarities = similarities / self.temperature
        
        # Apply softmax on the graph axis
        log_softmax_similarities = F.log_softmax(similarities, dim=1)  # batch_size x batch_size
        
        # Create labels (0, 1, 2, ..., batch_size-1)
        labels = torch.arange(batch_size).to(similarities.device)

        # Compute the loss
        loss = -log_softmax_similarities[range(batch_size), labels].mean()
        
        return loss

