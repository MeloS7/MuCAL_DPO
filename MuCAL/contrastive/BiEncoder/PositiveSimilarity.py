import torch
import torch.nn as nn
import torch.nn.functional as F

class PositiveSimilarity(nn.Module):
    def __init__(self):
        super(PositiveSimilarity, self).__init__()
        
    def forward(self, text_embeds, graph_embeds, batch_size, lang_set):
        # text_embeds: (batch_size * num_langs) x embed_dim
        # graph_embeds: batch_size x embed_dim
        num_langs = len(lang_set)
        text_embeds = text_embeds.view(batch_size, num_langs, -1)  # batch_size x num_langs x embed_dim
        graph_embeds = graph_embeds.unsqueeze(1)  # batch_size x 1 x embed_dim
        
        # Direct dot product since embeddings are normalized
        dot_product = torch.bmm(text_embeds, graph_embeds.transpose(1, 2))  # batch_size x num_langs x 1
        return dot_product.squeeze(2).mean(dim=1)  # batch_size