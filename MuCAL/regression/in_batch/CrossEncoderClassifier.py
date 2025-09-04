import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, models

class CrossEncoderClassifier(nn.Module):
    def __init__(self, encoder_model=None):
        super(CrossEncoderClassifier, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if encoder_model is not None:
            # Initialize Model
            word_embedding_model = models.Transformer(encoder_model)
            
            # Add pooling layer
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')

            # Add one linear layer
            dense_model = models.Dense(
                in_features=pooling_model.get_sentence_embedding_dimension(),
                out_features=1,
                activation_function=nn.Identity()
            )

            self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
            self.to(self.device)
        else:
            self.model = None

    def concatenate_text_graph_pairs(self, texts, graphs):
        """
        This function concatenates each text with its corresponding graph.
        It assumes that texts and graphs are lists of the same length.
        Output:
            cross_text: a list of strings (length: N)
        """
        assert len(texts) == len(graphs), "The length of texts and graphs must be equal."
        cross_text = [f"{text} [SEP] {graph}" for text, graph in zip(texts, graphs)]
        return cross_text

    def forward(self, texts, graphs):
        """
        texts: List[str]，图的文本描述
        graphs: List[str]，线性化后的 graph 形式
        return: shape = (batch_size,) 的张量, 表示匹配分数(logits)
        """
        # Concatenate the text and the linearized graph
        cross_data = self.concatenate_text_graph_pairs(texts, graphs)
        # 直接使用底层tokenize和模型调用，保留梯度
        cross_tokens = self.model.tokenize(cross_data)
        cross_tokens = {k: v.to(self.device) for k, v in cross_tokens.items()}
        cross_embeds = self.model(cross_tokens)["sentence_embedding"]

        logits = cross_embeds.squeeze(-1)
        return logits 

    def predict(self, texts, graphs):
        """推理模式，无梯度计算。"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(texts, graphs)
            return torch.sigmoid(logits)  # 在推理时添加sigmoid

    def save_pretrained(self, path):
        self.model.save_pretrained(path)

    @staticmethod
    def load(path) -> 'CrossEncoderClassifier':
        instance = CrossEncoderClassifier(encoder_model=None)
        instance.model = SentenceTransformer(path)  # 从path中加载预训练transformer
        instance.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        instance.model.to(instance.device)
        return instance

    def __str__(self):
        return f"CrossEncoderClassifier using device: {self.device}"

    def upload_model(self, repo_id):
        self.model.push_to_hub(repo_id=repo_id)
        print(f"Model uploaded successfully!")