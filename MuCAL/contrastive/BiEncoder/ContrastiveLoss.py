import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, bidirection=False, g2t_weight=1.0):
        super(ContrastiveLoss, self).__init__()
        self.bidirection = bidirection
        self.g2t_weight = g2t_weight  # G2T loss的权重

    def forward(self, text_embeds, graph_embeds, batch_size, lang_set, hard_neg_embeds=None):
        # Reshape text_embeds to (batch_size, len(lang_set), embed_dim)
        text_embeds = text_embeds.view(batch_size, len(lang_set), -1)

        # Compute similarities between text_embeds and graph_embeds
        # text_embeds: (batch_size * len(lang_set), embed_dim)
        # graph_embeds: (batch_size, embed_dim)
        similarities = torch.matmul(text_embeds.view(batch_size * len(lang_set), -1), graph_embeds.T)
        # Reshape and sum over languages
        similarities = similarities.view(batch_size, len(lang_set), batch_size).sum(dim=1)  # (batch_size, batch_size)

        if hard_neg_embeds is not None:
            # Reshape hard_neg_embeds to (batch_size, num_hard_negatives, embed_dim)
            # hard_neg_embeds: (batch_size, num_hard_negatives, embed_dim)
            # Compute similarities using batch matrix multiplication
            hard_neg_similarities = torch.matmul(text_embeds, hard_neg_embeds.transpose(2, 1))  # (batch_size, len(lang_set), num_hard_negatives)
            # Sum over languages
            hard_neg_similarities = hard_neg_similarities.sum(dim=1)  # (batch_size, num_hard_negatives)
            # Concatenate similarities
            similarities = torch.cat([similarities, hard_neg_similarities], dim=1)  # (batch_size, batch_size + num_hard_negatives)
        
        # Apply softmax on the graph axis
        log_softmax_similarities = F.log_softmax(similarities, dim=1)  # batch_size x batch_size or batch_size x (batch_size + num_hard_negatives)
        
        # Create labels (0, 1, 2, ..., batch_size-1)
        labels = torch.arange(batch_size).to(similarities.device)
        
        # Compute the loss
        loss_t2g = -log_softmax_similarities[range(batch_size), labels].mean()

        loss = loss_t2g

        if self.bidirection:
            # 对每种语言分别计算相似度，然后求和
            similarities_g2t = torch.zeros(batch_size, batch_size).to(graph_embeds.device)
            
            for lang_idx in range(len(lang_set)):
                # 获取当前语言的所有文本嵌入
                text_embeds_lang = text_embeds[:, lang_idx, :]  # (batch_size, embed_dim)
                
                # 计算图嵌入和当前语言文本嵌入之间的相似度
                similarities_lang = torch.matmul(graph_embeds, text_embeds_lang.T)  # (batch_size, batch_size)
                similarities_g2t += similarities_lang
            
            # 应用softmax并计算损失
            log_softmax_similarities_g2t = F.log_softmax(similarities_g2t, dim=1)  # (batch_size, batch_size)
            loss_g2t = -log_softmax_similarities_g2t[range(batch_size), labels].mean()
            
            # 使用权重组合两个方向的损失
            loss = (loss_t2g + self.g2t_weight * loss_g2t) / (1 + self.g2t_weight)


        # if self.bidirection:
        #     # Compute similarities from graph to text
        #     similarities_g2t = torch.zeros(batch_size, batch_size).to(graph_embeds.device)
        #     for lang_idx in range(len(lang_set)):
        #         text_embeds_lang = text_embeds[:, lang_idx, :]  # (batch_size, embed_dim)
        #         similarities_lang = torch.matmul(graph_embeds, text_embeds_lang.T)  # (batch_size, batch_size)
        #         similarities_g2t += similarities_lang  # Sum over languages

        #     # Apply softmax over the text axis (dim=0)
        #     log_softmax_similarities_g2t = F.log_softmax(similarities_g2t, dim=1)  # (batch_size, batch_size)

        #     # Compute the G2T loss
        #     loss_g2t = -log_softmax_similarities_g2t[range(batch_size), labels].mean()

        #     # Average the T2G and G2T losses
        #     loss = (loss_t2g + loss_g2t) / 2

        return loss