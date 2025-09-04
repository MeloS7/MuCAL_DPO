import re
import random
import torch
import wandb
from tqdm import tqdm
from transformers import get_scheduler
from torch.cuda.amp import autocast, GradScaler

class Trainer:
    def __init__(self, encoder, train_data_loader, lr=2e-5, num_epochs=1, train_batch=128, eval_batch=128, 
                 patience=3, save_path='ckpt/best_regression_model', wandb=None, eval_steps=100,
                 gradient_accumulation_steps=4, max_grad_norm=1.0):
        super(Trainer, self).__init__()
        # 初始化实例
        self.encoder = encoder
        self.wandb = wandb
        
        # 设置训练参数
        self.num_epochs = num_epochs
        self.lr = lr
        self.train_batch = train_batch
        self.eval_batch = eval_batch
        self.lang_set = ['en', 'zh', 'fr', 'ar', 'es', 'ru']
        # self.lang_set = ['en']
        
        # 早停参数
        self.patience = patience
        self.best_score = float('inf')
        self.patience_counter = 0
        self.save_path = save_path
        self.eval_steps = eval_steps
        self.steps_since_last_eval = 0
        
        # 梯度累积参数
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # 梯度裁剪参数
        self.max_grad_norm = max_grad_norm
        
        # 设置优化器和损失函数

        self.optimizer = torch.optim.AdamW(
            self.encoder.parameters(), lr=self.lr ,weight_decay=2e-5
        )

        self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1., device=self.encoder.device))  # 使用 BCEWithLogitsLoss 替代 BCELoss
        
        # 设置学习率调度器
        self.set_up_scheduler(train_data_loader)
        
        # 设置混合精度训练
        # self.scaler = GradScaler()
        
        # 打印模型信息
        print(f"学习率设置为 {self.lr}")
        print(self.encoder)

    def set_up_scheduler(self, train_data_loader):
        # 计算每个epoch的训练步数
        num_training_steps_per_epoch = len(train_data_loader.get_train_data()["graphs"]) // self.train_batch
        
        # 使用CosineAnnealingWarmRestarts调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=num_training_steps_per_epoch * 2,  # 重启周期
            T_mult=1,  # 每次重启后周期长度的倍增因子
            eta_min=2e-6  # 最小学习率
        )

    def generate_in_batch_negatives(self, texts, graphs, batch_size):
        """生成in-batch negatives并创建对应的标签。
           修正：同一个 graph 的所有多语言文本统统视为正例 (label=1)。
        """
        expanded_texts = []
        expanded_graphs = []
        labels = []
        
        for i_graph in range(batch_size):
            current_graph = graphs[i_graph]

            pos_text_indices_for_i = range(i_graph * len(self.lang_set), (i_graph + 1) * len(self.lang_set))
            pos_text_indices_for_i = set(pos_text_indices_for_i)  # 便于快速判断
            
            for i_text in range(batch_size * len(self.lang_set)):
                # Add text and graphs
                expanded_texts.append(texts[i_text])
                expanded_graphs.append(current_graph)

                # Add label
                if i_text in pos_text_indices_for_i:
                    labels.append(1.0)
                else:
                    labels.append(0.0)
        
        return expanded_texts, expanded_graphs, torch.tensor(labels, device=self.encoder.device)

    def train(self, train_data_loader):
        train_data = train_data_loader.get_train_data()
        num_samples = len(train_data["graphs"])
        
        for epoch in tqdm(range(self.num_epochs)):
            self.encoder.train()
            total_loss = 0
            num_batches = 0
            global_step = 0
            
            # 在每个epoch开始时清零梯度
            self.optimizer.zero_grad()
            
            for i in tqdm(range(0, num_samples, self.train_batch)):
                end_index = min(i + self.train_batch, num_samples)
                actual_batch_size = end_index - i
                
                if actual_batch_size != self.train_batch:
                    print(f"最后一个batch大小: {actual_batch_size}")
                    
                batched_texts = train_data["texts"][i * len(self.lang_set):end_index * len(self.lang_set)]
                batched_graphs = train_data["graphs"][i:end_index]

                # 生成in-batch negatives和标签
                expanded_texts, expanded_graphs, labels = self.generate_in_batch_negatives(
                    batched_texts, batched_graphs, actual_batch_size
                )

                # 编码文本和图并计算分数
                scores = self.encoder(texts=expanded_texts, graphs=expanded_graphs)
                loss = self.loss(scores, labels)
                
                # 计算损失并缩放
                scaled_loss = loss / self.gradient_accumulation_steps  # 缩放损失
                scaled_loss.backward()  # 反向传播缩放后的损失
                
                # 每gradient_accumulation_steps步更新一次参数
                if (global_step + 1) % self.gradient_accumulation_steps == 0:
                    # 执行梯度裁剪
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
                    
                    # 记录梯度范数
                    if self.wandb:
                        self.wandb.log({
                            "gradient_norm": grad_norm.item(),
                        })
                    
                    # 更新参数
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                total_loss += loss.item()
                num_batches += 1
                global_step += 1

                if i % (100 * self.train_batch) == 0:
                    # print(self.encoder.check_encoder_training())
                    with torch.no_grad():
                        logits = self.encoder(texts=expanded_texts, graphs=expanded_graphs)
                        probs  = torch.sigmoid(logits)
                        print("pos  mean:", probs[labels==1].mean().item(),
                              "neg  mean:", probs[labels==0].mean().item())
                
                # 记录训练信息
                current_lr = self.optimizer.param_groups[0]['lr']
                current_memory = torch.cuda.memory_allocated() / (1024 ** 2)
                
                self.wandb.log({
                    "train_loss": scaled_loss.item(),  # loss.item() 是损失的值
                    "learning_rate": current_lr,
                    "gpu_memory_usage_mb": current_memory
                })
                
                # 更新步数计数器
                if (global_step + 1) % self.gradient_accumulation_steps == 0:
                    self.steps_since_last_eval += 1
                
                # 每eval_steps步进行一次验证
                if self.steps_since_last_eval >= self.eval_steps:
                    print(f"\n执行验证 在step {epoch * (num_samples // self.train_batch) + num_batches}")
                    # 验证
                    dev_data = train_data_loader.get_dev_data()
                    val_loss = self.validate(dev_data)
                    
                    print(f"验证损失: {val_loss:.4f}")
                    
                    # 记录验证指标
                    self.wandb.log({
                        "val_loss": val_loss,
                    })
                    
                    # 早停检查
                    if val_loss < self.best_score:
                        self.best_score = val_loss
                        self.patience_counter = 0
                        print(f"损失值改善。保存模型到 {self.save_path}")
                        self.save_model(self.save_path)
                    else:
                        self.patience_counter += 1
                        print(f"损失值未改善 {self.patience_counter} 次验证")
                        if self.patience_counter >= self.patience:
                            print("触发早停机制")
                            return  # 直接返回,结束训练
                    
                    # 重置步数计数器
                    self.steps_since_last_eval = 0
                    # 恢复训练模式
                    self.encoder.train()
                
                # 释放内存
                del expanded_texts, expanded_graphs, scores, loss, labels
                torch.cuda.empty_cache()
            
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{self.num_epochs}, 平均训练损失: {avg_loss:.4f}")

    def validate(self, dev_data):
        self.encoder.eval()
        total_loss = 0
        num_batches = 0
        num_samples = len(dev_data["graphs"])
        num_lang = len(self.lang_set)
        
        with torch.no_grad():
            print("计算验证loss...")
            for i in tqdm(range(0, num_samples, self.eval_batch)):
                end_idx = min(i + self.eval_batch, num_samples)
                batch_size = end_idx - i
                
                # 获取当前batch的texts和graphs
                batch_texts = dev_data["texts"][i * num_lang:end_idx * num_lang]
                batch_graphs = dev_data["graphs"][i:end_idx]
                
                # 使用与训练相同的in-batch negatives生成方法
                expanded_texts, expanded_graphs, labels = self.generate_in_batch_negatives(
                    batch_texts, batch_graphs, batch_size
                )
                
                # 直接使用encoder计算分数
                # with autocast():
                scores = self.encoder(texts=expanded_texts, graphs=expanded_graphs)
                loss = self.loss(scores, labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                # 记录GPU内存使用
                current_memory = torch.cuda.memory_allocated() / (1024 ** 2)
                self.wandb.log({"validate_gpu_memory_usage_mb": current_memory})
                
                # 释放内存
                del batch_texts, batch_graphs, expanded_texts, expanded_graphs, scores, labels
                torch.cuda.empty_cache()
        
        return total_loss / num_batches

    def save_model(self, path):
        self.encoder.save_pretrained(path)

    def predict(self, texts, graphs):
        """推理时需要添加sigmoid来获得概率分数"""
        self.encoder.eval()
        with torch.no_grad():
            logits = self.encoder(texts, graphs)
            return torch.sigmoid(logits)  # 在推理时添加sigmoid
