import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from sentence_transformers import SentenceTransformer

class BiEncoder(nn.Module):
    def __init__(self, encoder_model='sentence-transformers/paraphrase-multilingual-mpnet-base-v2'):
        super(BiEncoder, self).__init__()
        self.model = SentenceTransformer(encoder_model)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)
        
    
    def forward(self, texts, graphs):
        # Encode texts
        if texts != []:
            encoded_texts = self.model.tokenize(texts)
            encoded_texts = {k: v.to(self.device) for k, v in encoded_texts.items()}
            text_embeds = self.model(encoded_texts)["sentence_embedding"]
            text_embeds = F.normalize(text_embeds, p=2, dim=1)
        else:
            text_embeds = None
        
        # Encode graphs
        if graphs != []:
            encoded_graphs = self.model.tokenize(graphs)
            encoded_graphs = {k: v.to(self.device) for k, v in encoded_graphs.items()}
            graph_embeds = self.model(encoded_graphs)["sentence_embedding"]
            graph_embeds = F.normalize(graph_embeds, p=2, dim=1)
        else:
            graph_embeds = None
        
        return text_embeds, graph_embeds
    
    def save_pretrained(self, path):
        self.model.save_pretrained(path)

    def upload_model(self, repo_id):
        self.model.push_to_hub(repo_id=repo_id)
        print(f"Model uploaded successfully!")
        
    @staticmethod
    def load(path) -> 'BiEncoder':
        instance = BiEncoder(encoder_model=path)
        return instance
    
        
    def __str__(self):
        return f"BiEncoder using device: {self.device}"
