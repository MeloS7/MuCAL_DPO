import torch
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler

from CrossEncoderContrastiveLoss import CrossEncoderContrastiveLoss
from RankAccuracy import RankAccuracy

class Trainer:
    def __init__(self, encoder, train_data_loader, lr=2e-5, num_epochs=1, train_batch=128, eval_batch=128,
                patience=3, save_path='ckpt/best_cr_model_tran3', wandb=None):
        super(Trainer, self).__init__()
        # Initialize instances
        self.encoder = encoder

        # Initialize Data Loader
        self.train_data_loader = train_data_loader
        
        # Set up parameters
        self.num_epochs = num_epochs
        self.lr = lr
        self.train_batch=train_batch
        self.eval_batch=eval_batch
        self.lang_set = ['en', 'zh', 'fr', 'ar', 'es', 'ru']
        
        # Early stopping parameters
        self.patience = patience
        self.best_mrr = -float('inf')
        self.patience_counter = 0
        self.save_path = save_path
        
        # Set up optimizer and loss function
        self.optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=self.lr, weight_decay=1e-3)
        self.loss = CrossEncoderContrastiveLoss()
        
        # Learning rate scheduler
        self.set_up_scheduler()

        # Initialize GradScaler
        self.scaler = GradScaler()
        
        # Print encoder info
        print(self.encoder)
        
        self.wandb = wandb
    
    def set_up_scheduler(self):
        num_training_steps_per_epoch = len(self.train_data_loader.get_train_data()["graphs"]) // self.train_batch
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=num_training_steps_per_epoch,  # 每个epoch重启一次
            T_mult=2,  # 每次重启后周期翻倍
            eta_min=1e-6  # 最小学习率
        )
    
    def train(self): 
        train_data = self.train_data_loader.get_train_data()
        # dev_data = self.train_data_loader.get_dev_data()
        dev_data = None
        dev_mrr_data = self.train_data_loader.get_graphset_dev()
        
        num_samples = len(train_data["graphs"])
        num_training_steps_per_epoch = num_samples // self.train_batch
        
        for epoch in tqdm(range(self.num_epochs)):
            self.encoder.train()
            total_loss = 0
            num_batches = 0
            peak_memory = 0
            self.optimizer.zero_grad()
            
            for i in tqdm(range(0, num_samples, self.train_batch)):
                end_index = min(i + self.train_batch, num_samples)
                actual_batch_size = end_index - i
                
                if actual_batch_size != self.train_batch:
                    print(f"Last Batch size: {actual_batch_size}.")
                    
                batched_texts = train_data["texts"][i * len(self.lang_set):end_index * len(self.lang_set)]
                batched_graphs = train_data["graphs"][i:end_index]

                # Encoder text and compute in-batch outputs
                with autocast():
                    logits = self.encoder(texts=batched_texts, graphs=batched_graphs, train=True)
                loss = self.loss(logits, actual_batch_size, len(self.lang_set), device=self.encoder.device)
                
                # 直接反向传播，不需要缩放损失
                self.scaler.scale(loss).backward()
                
                # 记录实际的loss
                total_loss += loss.item()
                num_batches += 1
                
                # 检查梯度
                total_grad = 0
                for param in self.encoder.parameters():
                    if param.grad is not None:
                        total_grad += param.grad.norm().item()
                
                # 如果梯度太小则跳过更新
                if total_grad < 1e-8:
                    print(f"Warning: Very small gradient detected: {total_grad}")
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
                
                # 更新参数
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # 记录GPU内存使用
                current_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
                peak_memory = max(peak_memory, current_memory)
                
                self.wandb.log({
                    "total_gradient_norm": total_grad,
                    "train_loss": loss.item(),
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                    "memory_usage": current_memory,
                    "epoch": epoch + i / num_samples,
                    "step_in_epoch": i,
                    "peak_gpu_memory_mb": peak_memory,
                })

                # 释放内存
                del batched_texts, batched_graphs, logits, loss
                torch.cuda.empty_cache()
                
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{self.num_epochs}, Average Training Loss: {avg_loss:.4f}")
            
            # Save model
            self.save_model(self.save_path+f"_ep{epoch+1}")
            print(f"Model saved to {self.save_path+f'_ep{epoch+1}'}")

            # Add avg loss to wandb
            self.wandb.log({"avg_loss": avg_loss})

            # Stop training if epoch is 4
            if epoch == 4:
                exit()

            if dev_data:
                val_loss, avg_cosine_similarity, mrr_t2g, mrr_g2t, avg_mrr = self.validate(dev_data, dev_mrr_data)
                print(f"Epoch {epoch+1}/{self.num_epochs}, Validation Loss: {val_loss:.4f}, "
                      f"Average Matching Score: {avg_cosine_similarity:.4f}, "
                      f"MRR (t2g): {mrr_t2g:.4f}, MRR (g2t): {mrr_g2t:.4f}, "
                      f"Average MRR: {avg_mrr:.4f}")

                # Early Stopping and Checkpointing - 使用avg_mrr替代val_loss
                if avg_mrr > self.best_mrr:
                    self.best_mrr = avg_mrr
                    self.patience_counter = 0
                    print(f"MRR improved. Saving the model to {self.save_path}")
                    self.save_model(self.save_path)
                else:
                    self.patience_counter += 1
                    print(f"No improvement in MRR for {self.patience_counter} epoch(s).")
                    if self.patience_counter >= self.patience:
                        print("Early stopping triggered.")
                        break
                        
                # wandb记录
                self.wandb.log({
                    "epoch_avg_loss": avg_loss,
                    "val_loss": val_loss,
                    "val_avg_cosine_similarity": avg_cosine_similarity,
                    "val_mrr_t2g": mrr_t2g,
                    "val_mrr_g2t": mrr_g2t,
                    "val_avg_mrr": avg_mrr,
                    "peak_gpu_memory_mb": peak_memory,
                })
                
    def validate(self, dev_data, dev_mrr_data):
        # 首先计算验证损失和相似度
        self.encoder.eval()
        total_loss = 0
        num_batches = 0
        num_samples = len(dev_data["graphs"])
        total_matching_scores = 0

        with torch.no_grad():
            for i in tqdm(range(0, num_samples, self.eval_batch)):
                end_index = min(i + self.eval_batch, num_samples)
                actual_batch_size = end_index - i
                
                if actual_batch_size != self.eval_batch:
                    print(f"Last Batch size: {actual_batch_size}.")
                    
                batched_texts = dev_data["texts"][i * len(self.lang_set):end_index * len(self.lang_set)]
                batched_graphs = dev_data["graphs"][i:end_index]
                batch_size = actual_batch_size
                num_lang = len(self.lang_set)
                
                # Compute loss on the validation set
                # Encoder text and compute in-batch outputs
                with autocast():
                    logits = self.encoder(texts=batched_texts, graphs=batched_graphs, train=True)
                
                # Compute the loss
                with autocast():
                    loss = self.loss(logits, batch_size, num_lang, device=self.encoder.device)
                total_loss += loss.item()
                
                # Compute matching score for all positive instances
                # Expand Graphs
                expanded_graphs = []
                for graph in batched_graphs:
                    expanded_graphs.extend([graph] * num_lang)
                    
                
                # Get the matching score (proba)
                with autocast():
                    matching_scores = self.encoder.predict(batched_texts, expanded_graphs)
                total_matching_scores += matching_scores.sum().item()
                
                num_batches += 1
                
                # 记录GPU内存使用
                current_memory = torch.cuda.memory_allocated() / (1024 ** 2)
                self.wandb.log({"validate_gpu_memory_usage_mb": current_memory})

                print(f"Loss: {loss.item():.4f}, Matching Score: {matching_scores.sum().item():.4f}")

                # Free memory
                del batched_texts, batched_graphs, logits, loss, matching_scores
                torch.cuda.empty_cache()
                
        avg_loss = total_loss / num_batches
        avg_matching_score = total_matching_scores / len(dev_data["texts"])

        # 计算MRR
        ranker = RankAccuracy(self.encoder, dev_mrr_data, self.lang_set)
        
        # 计算text-to-multilang-graph MRR
        _, mrr_t2g = ranker.compute_text_to_multilang_graph_rank_n(n=[1, 3, 10])

        # 计算graph-to-multilang-text MRR
        _, mrr_g2t = ranker.compute_graph_to_multilang_text_rank_n(n=[1, 3, 10])

        # 计算平均MRR
        avg_mrr = (mrr_t2g + mrr_g2t) / 2

        # 记录验证结果
        self.wandb.log({
            "val_loss": avg_loss,
            "val_avg_cosine_similarity": avg_matching_score,
            "val_mrr_t2g": mrr_t2g,
            "val_mrr_g2t": mrr_g2t,
            "val_avg_mrr": avg_mrr,
        })
        
        return avg_loss, avg_matching_score, mrr_t2g, mrr_g2t, avg_mrr
    
    
    def evaluate(self, eval_data):
        # Compute the loss function and positive similarity
        avg_loss, avg_cosine_similarity, mrr_t2g, mrr_g2t, avg_mrr = self.validate(eval_data)
        return avg_loss, avg_cosine_similarity, mrr_t2g, mrr_g2t, avg_mrr
    
    
    def save_model(self, path):
        self.encoder.save_pretrained(path)
