import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer, AutoModel

class BiEncoderRegression(nn.Module):
    """
    一个简洁的 Bi-Encoder + Regression 结构：
    - 用同一个 SentenceTransformer 分别编码 text 和 graph
    - 拼接它们的向量后，通过 Linear+Sigmoid 得到 [0,1] 区间的匹配分数
    - 保留类似 CrossEncoder 的写法，以便保存 / 加载
    """

    def __init__(self, encoder_model=None):
        super(BiEncoderRegression, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if encoder_model is not None:
            # Initialize Model
            word_embedding_model = models.Transformer(encoder_model)
            
            # Add pooling layer
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')

            self.encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])
            self.encoder.to(self.device)
            
            # 回归头：将 text_embed 与 graph_embed 拼接后 -> [4 * embedding_dim] -> 1
            self.dense = nn.Linear(pooling_model.get_sentence_embedding_dimension() * 4, 1)
            # 移除 sigmoid 激活函数，因为它会被包含在 BCEWithLogitsLoss 中
            
            self.to(self.device)
        else:
            self.encoder = None
            self.dense = None

    def forward(self, texts, graphs):
        """
        texts: List[str]，图的文本描述
        graphs: List[str]，线性化后的 graph 形式
        return: shape = (batch_size,) 的张量, 表示匹配分数(logits)
        """
        # 直接使用底层tokenize和模型调用，保留梯度
        text_tokens = self.encoder.tokenize(texts)
        text_tokens = {k: v.to(self.device) for k, v in text_tokens.items()}
        text_embeds = self.encoder(text_tokens)["sentence_embedding"]
        
        graph_tokens = self.encoder.tokenize(graphs)
        graph_tokens = {k: v.to(self.device) for k, v in graph_tokens.items()}
        graph_embeds = self.encoder(graph_tokens)["sentence_embedding"]

        # 拼接后输入Dense
        cat_embeds = torch.cat([text_embeds, graph_embeds, torch.abs(text_embeds - graph_embeds), text_embeds * graph_embeds], dim=1)  # (batch_size, 4*dim)
        logits = self.dense(cat_embeds).squeeze(-1)                  # (batch_size,)
        logits = 5.0 * torch.tanh(logits / 5.0)                      # 将 logits 映射到 [-5, 5] 区间
        return logits 

    def predict(self, texts, graphs):
        """推理模式，无梯度计算。"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(texts, graphs)
            return torch.sigmoid(logits)  # 在推理时添加sigmoid

    def save_pretrained(self, path):
        """
        保存模型：
        1) self.encoder -> sentence-transformers 的目录结构 (config.json, etc.)
        2) self.dense -> 线性层的 PyTorch state_dict
        """
        if self.encoder is not None:
            self.encoder.save(path)  # 正确的SentenceTransformer保存方法
        
        if self.dense is not None:
            torch.save(self.dense.state_dict(), f"{path}/dense.pt")

    @staticmethod
    def load(path) -> 'BiEncoderRegression':
        """
        从指定路径加载：
        1) SentenceTransformer
        2) Dense层的权重
        """
        instance = BiEncoderRegression(encoder_model=None)
        instance.encoder = SentenceTransformer(path)  # 从path中加载预训练transformer
        instance.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        instance.encoder.to(instance.device)

        embedding_dim = instance.encoder.get_sentence_embedding_dimension()
        instance.dense = nn.Linear(embedding_dim * 2, 1)
        
        dense_sd = torch.load(f"{path}/dense.pt", map_location=instance.device)
        instance.dense.load_state_dict(dense_sd)
        instance.dense.to(instance.device)

        return instance

    def __str__(self):
        return f"BiEncoderRegression using device: {self.device}"

    def check_encoder_training(self):
        """检查编码器是否在训练"""
        # 获取编码器参数的梯度统计
        grad_stats = {}
        for name, param in self.encoder.named_parameters():
            if param.grad is not None:
                grad_stats[name] = {
                    'mean': param.grad.abs().mean().item(),
                    'max': param.grad.abs().max().item()
                }
            else:
                grad_stats[name] = {'mean': 0, 'max': 0}
        
        # 打印一些关键层的梯度统计
        for name, stats in grad_stats.items():
            if 'encoder' in name or 'pooling' in name:
                print(f"{name}: mean={stats['mean']:.6f}, max={stats['max']:.6f}")
        
        return grad_stats