import torch
import wandb
import random
from tqdm import tqdm
from transformers import get_scheduler

from ContrastiveLoss import ContrastiveLoss
from PositiveSimilarity import PositiveSimilarity
from RankAccuracy import RankAccuracy

from torch.cuda.amp import autocast, GradScaler

class Trainer:
    def __init__(self, encoder, train_data_loader, lang_set, lr=2e-5, num_epochs=1, train_batch=128, eval_batch=128, patience=3, save_path='ckpt/best_bi_model', wandb=None, bidirection=False):
        super(Trainer, self).__init__()
        # Initialize instances
        self.encoder = encoder
        self.wandb = wandb
        self.bidirection = bidirection

        # Set up parameters
        self.num_epochs = num_epochs
        self.lr = lr
        self.train_batch=train_batch
        self.eval_batch=eval_batch
        self.lang_set = lang_set
        
        # Early stopping parameters
        self.patience = patience
        self.best_mrr = -float('inf')
        self.patience_counter = 0
        self.save_path = save_path
        
        # Set up optimizer and loss function
        self.optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=self.lr, weight_decay=1e-3)
        self.loss = ContrastiveLoss(bidirection=self.bidirection)
        self.pos_sim = PositiveSimilarity()

        # Set up scheduler
        self.set_up_scheduler(train_data_loader)

        # Set up autocast and scaler
        self.scaler = GradScaler()
        
        # Print encoder info
        print(f"The learning rate is {self.lr}")
        print(self.encoder)
    
    def set_up_scheduler(self, train_data_loader):
        # Total training steps
        num_training_steps_per_epoch = len(train_data_loader.get_train_data()["graphs"]) // self.train_batch

        # CosineAnnealingWarmRestarts scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=num_training_steps_per_epoch * 2,  # Restart period
            T_mult=1,  # The factor by which the period length is multiplied after each restart
            eta_min=1e-6  # The minimum learning rate
        )

    # def set_up_scheduler(self, train_data_loader):
    #     num_training_steps_per_epoch = len(train_data_loader.get_train_data()["graphs"]) // self.train_batch

    #     # Linear Decay with Warm-Up Scheduler
    #     self.scheduler = get_scheduler(
    #         "linear",
    #         optimizer=self.optimizer,
    #         num_warmup_steps=int(0.15 * num_training_steps_per_epoch * self.num_epochs),
    #         num_training_steps=num_training_steps_per_epoch * self.num_epochs
    #     )


    def train(self, train_data_loader, use_hard_negatives=False, num_hard_negatives=6): 
        train_data = train_data_loader.get_train_data()
        num_samples = len(train_data["graphs"])
        num_training_steps_per_epoch = num_samples // self.train_batch

        for epoch in tqdm(range(self.num_epochs)):
            self.encoder.train()
            total_loss = 0
            num_batches = 0
            step = 0

            # Generate hard negatives if needed (they are various for each graph)
            if use_hard_negatives:
                if not train_data_loader.generate_hard_negatives(num_hard_negatives):
                    print("Failed to generate hard negatives.")
                    exit()

            for i in tqdm(range(0, num_samples, self.train_batch)):
                end_index = min(i + self.train_batch, num_samples)
                actual_batch_size = end_index - i
                
                if actual_batch_size != self.train_batch:
                    print(f"Last Batch size: {actual_batch_size}.")
                    
                batched_texts = train_data["texts"][i * len(self.lang_set):end_index * len(self.lang_set)]
                batched_graphs = train_data["graphs"][i:end_index]

                # Encoder graph and text
                with autocast():
                    text_embeds, graph_embeds = self.encoder(texts=batched_texts, graphs=batched_graphs)

                if use_hard_negatives:
                    batched_indices = [j for j in range(i, end_index)]
                    assert len(batched_indices) == actual_batch_size
                    # Select hard negatives
                    batched_hard_negatives = train_data_loader.select_hard_negatives_in_batch(batched_indices)
                    # Flatten the hard negatives
                    batched_hard_negatives_flat = [item for sublist in batched_hard_negatives for item in sublist]
                    # Encode hard negatives
                    _, batched_hard_neg_embeds_flat = self.encoder(texts=[], graphs=batched_hard_negatives_flat)
                    # Determine num_hard_negatives
                    num_hard_negatives = len(batched_hard_negatives[0])
                    # Reshape to (batch_size, num_hard_negatives, embed_dim)
                    batched_hard_neg_embeds = batched_hard_neg_embeds_flat.view(actual_batch_size, num_hard_negatives, -1)
                    
                else:
                    batched_hard_neg_embeds = None
                
                # Compute the loss
                with autocast():
                    loss = self.loss(text_embeds, graph_embeds, actual_batch_size, self.lang_set, batched_hard_neg_embeds)
                total_loss += loss.item()
                num_batches += 1

                # Backpropagation
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Update learning rate using CosineAnnealingWarmRestarts
                self.scheduler.step()

                # Log loss and learning rate
                current_lr = self.optimizer.param_groups[0]['lr']

                current_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
                self.wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": current_lr,
                    "memory_usage": current_memory,
                    "epoch": epoch + step / num_training_steps_per_epoch,
                    "step_in_epoch": step
                })

                # Free up memory
                del text_embeds, graph_embeds, loss
                torch.cuda.empty_cache()

                # Increment step counter
                step += 1
                
            avg_loss = total_loss / num_batches
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Peak memory in MB
            print(f"Epoch {epoch+1}/{self.num_epochs}, Average Training Loss: {avg_loss:.4f}")
            
            # Log peak memory
            self.wandb.log({
                "epoch_avg_loss": avg_loss,
                "peak_gpu_memory_mb": peak_memory,
            })

            # Perform validation at the end of each epoch
            self.perform_validation(epoch, train_data_loader)
                
        self.wandb.finish()

    def perform_validation(self, epoch, train_data_loader):
        # Validation
        dev_data = train_data_loader.get_dev_data()
        dev_mrr_data = train_data_loader.get_graphset_dev()
        val_loss, avg_cosine_similarity, mrr_t2g, mrr_g2t, avg_mrr = self.validate(dev_data, dev_mrr_data)
        print(f"Epoch {epoch+1}/{self.num_epochs}, Validation Loss: {val_loss:.4f}, Average Cosine Similarity: {avg_cosine_similarity:.4f},\
            MRR (t2g): {mrr_t2g:.4f}, MRR (g2t): {mrr_g2t:.4f}, Average MRR: {avg_mrr:.4f}")

        # Log validation metrics
        self.wandb.log({
            "val_loss": val_loss,
            "val_avg_cosine_similarity": avg_cosine_similarity,
            "val_avg_mrr": avg_mrr,
            "val_mrr_t2g": mrr_t2g,
            "val_mrr_g2t": mrr_g2t,
        })

        # Early Stopping and Checkpointing
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
                exit() # Exit the training loop
        
    def validate(self, dev_data, dev_mrr_data):
        self.encoder.eval()
        total_loss = 0
        num_batches = 0
        num_samples = len(dev_data["graphs"])
        total_cosine_similarity = 0
        num_pairs = 0

        with torch.no_grad():
            for i in tqdm(range(0, num_samples, self.eval_batch)):
                end_index = min(i + self.eval_batch, num_samples)
                actual_batch_size = end_index - i
                
                if actual_batch_size != self.eval_batch:
                    print(f"Last Batch size: {actual_batch_size}.")
                    
                batched_texts = dev_data["texts"][i * len(self.lang_set):end_index * len(self.lang_set)]
                batched_graphs = dev_data["graphs"][i:end_index]
                
                # Encode graph and text
                with autocast():
                    text_embeds, graph_embeds = self.encoder(texts=batched_texts, graphs=batched_graphs)

                    # Compute the loss
                    loss = self.loss(text_embeds, graph_embeds, actual_batch_size, self.lang_set)
                total_loss += loss.item()
                
                # Compute the cosine similarity of positive examples
                pos_sim = self.pos_sim(text_embeds, graph_embeds, actual_batch_size, self.lang_set)
                total_cosine_similarity += pos_sim.sum().item()
                num_pairs += pos_sim.size(0)
                
                num_batches += 1

                # Log GPU memory usage
                current_memory = torch.cuda.memory_allocated() / (1024 ** 2)
                self.wandb.log({"validate_gpu_memory_usage_mb": current_memory})

                # Free up memory
                del text_embeds, graph_embeds, pos_sim
                torch.cuda.empty_cache()
         
        avg_loss = total_loss / num_batches
        avg_cosine_similarity = total_cosine_similarity / num_pairs

        # Compute MRR for 'en' language using RankAccuracy
        ranker = RankAccuracy(self.encoder, dev_mrr_data, self.lang_set)
        
        # Compute text-to-multilang-graph MRR
        t2g_results, mrr_t2g = ranker.compute_text_to_multilang_graph_rank_n(n=[1, 3, 10])

        # Compute graph-to-multilang-text MRR
        g2t_results, mrr_g2t = ranker.compute_graph_to_multilang_text_rank_n(n=[1, 3, 10])

        # Compute the average MRR
        avg_mrr = (mrr_t2g + mrr_g2t) / 2

        return avg_loss, avg_cosine_similarity, mrr_t2g, mrr_g2t, avg_mrr
    
    
    def evaluate(self, eval_data):
        # Compute the loss function and positive similarity
        avg_loss, avg_cosine_similarity, mrr_t2g, mrr_g2t, avg_mrr = self.validate(eval_data)
        return avg_loss, avg_cosine_similarity, mrr_t2g, mrr_g2t, avg_mrr
    
    def save_model(self, path):
        self.encoder.save_pretrained(path)