import argparse
from Trainer import Trainer
from DataLoader import DataLoader
from CrossEncoderClassifier import CrossEncoderClassifier
from RankAccuracy import RankAccuracy
from RankAccuracy_FCT import RankAccuracy_FCT
import wandb
import os

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('-it', '--input_train_path', type=str, default='data/train/merged_train_with_MT.json')
    args.add_argument('-id', '--input_dev_path', type=str, default='data/dev/merged_dev_with_MT.json')
    args.add_argument('-ie', '--input_eval_path', type=str, default='data/test/new_test_with_MT.json')
    args.add_argument('-tbs', '--train_batch_size', type=int, default=32)
    args.add_argument('-vbs', '--val_batch_size', type=int, default=128)
    args.add_argument('-ep', '--epochs', type=int, default=5)
    args.add_argument('-sp', '--save_path', type=str, default='ckpt/regression_model')
    args.add_argument('-lp', '--load_path', type=str, help='Path to the checkpoint')
    args.add_argument('-tr', '--train', action='store_true', help='Flag to train the model')
    args.add_argument('-val', '--validate', action='store_true', help='Flag to validate the model')
    args.add_argument('-eval', '--evaluate', action='store_true', help='Flag to evaluate the model')
    args.add_argument('-upload', '--upload_model', action='store_true', help='Flag to upload the model to Huggingface')
    args.add_argument('-upload_name', '--upload_name', type=str, default='biencoder_ep10_bs32_trans3')
    args.add_argument('-wandb', '--use_wandb', action='store_true', help='Flag to use wandb')
    args.add_argument('-lr', '--learning_rate', type=float, default=2e-5, help='Learning rate')
    args.add_argument('-gac', '--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    args.add_argument('-mgc', '--max_grad_norm', type=float, default=1.0, help='Max gradient norm')

    
    args = args.parse_args()
    
    # Parse args
    train_path = args.input_train_path
    dev_path = args.input_dev_path
    eval_path = args.input_eval_path
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    epochs = args.epochs
    save_path = args.save_path
    load_path = args.load_path
    train = args.train
    validate = args.validate
    evaluate = args.evaluate
    upload = args.upload_model
    learning_rate = args.learning_rate
    gradient_accumulation_steps = args.gradient_accumulation_steps
    max_grad_norm = args.max_grad_norm

    
    if args.use_wandb:
        # Initialize W&B
        wandb.init(
            project="Multi-Align-G2T",
            name=f"regression_in_batch_lr_{learning_rate}_batch_{train_batch_size}_epoch_{epochs}",
            group="BiEncoder",
            config={
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": train_batch_size,
            }
        )
    else:
        class DummyWandb:
            def log(self, *args, **kwargs):
                pass
            def finish(self):
                pass

        wandb = DummyWandb()

    print(f"Train Batch size: {train_batch_size}, Val Batch size: {val_batch_size}, \nEpochs: {epochs}, \nSave path: {save_path}\nLoad path: {load_path}")
    
    # Set up language set
    lang_set = ['en', 'zh', 'fr', 'ar', 'es', 'ru']
    # lang_set = ['en']
    
    # Load Data
    data_loader = DataLoader(train_path, dev_path, eval_path, lang_set)
    train_data = data_loader.get_train_data()
    dev_data = data_loader.get_dev_data()
    
    # Load Trainer
    model = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    if load_path:
        encoder = CrossEncoderClassifier.load(load_path)
    else:
        encoder = CrossEncoderClassifier(model)

    # Load Trainer
    trainer = Trainer(
        encoder, 
        data_loader, 
        lr=learning_rate,
        num_epochs=epochs, 
        train_batch=train_batch_size, 
        eval_batch=val_batch_size, 
        patience=3, 
        save_path=save_path+'_best',
        wandb=wandb,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
    )
    
    # Train the model
    if train:
        trainer.train(data_loader)
        # Save the checkpoint
        # trainer.save_model(save_path)
    
    if validate:
        dev_mrr_data = data_loader.get_graphset_dev()
        val_loss, avg_cosine_similarity, mrr_t2g, mrr_g2t, avg_mrr = trainer.validate(dev_data, dev_mrr_data)
        print(f"Validation Loss: {val_loss:.4f}, Average Cosine Similarity: {avg_cosine_similarity:.4f},\
         MRR (t2g): {mrr_t2g:.4f}, MRR (g2t): {mrr_g2t:.4f}, Average MRR: {avg_mrr:.4f}")
        
    if evaluate:
        # Compute loss and positive similarity
        eval_data = data_loader.get_eval_data()
        # val_loss, avg_cosine_similarity = trainer.evaluate(eval_data)
        # print(f"Validation Loss: {val_loss:.4f}, Average Cosine Similarity: {avg_cosine_similarity:.4f}")
        
      # =========== T2G Part ==============
        # Compute Rank@N Accuracy
        graphset_texts = data_loader.get_graphset_texts()
        # ranker = RankAccuracy(encoder, graphset_texts, lang_set)
        ranker = RankAccuracy_FCT(graphset_texts, lang_set)
        # multi_accuracy_results, mrr = ranker.compute_text_to_multilang_graph_rank_n(n=[1, 3, 10])
        
        # # Print Rank@N results and average rank
        # print("Multilingual T2G - Rank@N Accuracy Results:")
        # for rank, accuracy in multi_accuracy_results.items():
        #     print(f"Rank@{rank}: {accuracy*100:.2f}%")
        # print(f"MRR: {mrr:.4f}")
        
        # Compute Monolingual Rank@N Accuracy for each language
        # mono_lang_results, mono_mrr = ranker.compute_text_to_monolang_graph_rank_n(n=[1, 3, 10])
        
        # # Print Monolingual Rank@N results and average ranks for each language
        # print("Monolingual T2G - Rank@N Accuracy Results by Language:")
        # for lang, results in mono_lang_results.items():
        #     print(f"\nLanguage: {lang}")
        #     for rank in [1, 3, 10]: 
        #         print(f"Rank@{rank}: {results[rank]*100:.2f}%")  # Convert to percentage
        #     print(f"MRR: {mono_mrr[lang]:.4f}") 
            
        # =========== G2T Part ==============
        # multi_accuracy_results, mrr = ranker.compute_graph_to_multilang_text_rank_n(n=[1, 3, 10])
        
        # # Print Rank@N results and average rank
        # print("Multilingual G2T - Rank@N Accuracy Results:")
        # for rank, accuracy in multi_accuracy_results.items():
        #     print(f"Rank@{rank}: {accuracy*100:.2f}%")
        # print(f"MRR: {mrr:.4f}")
        
        # Compute Monolingual Rank@N Accuracy for each language (Graph to Text)
        mono_lang_results = ranker.compute_graph_to_monolang_text_rank_n(n=[1, 3, 10])
        
        # Print Monolingual results for each language
        print("\nMonolingual G2T - Results by Language:")
        for lang, results in mono_lang_results.items():
            print(f"\nLanguage: {lang}")
            for rank in [1, 3, 10]: 
                print(f"Rank@{rank}: {results[rank]*100:.2f}%")
            print(f"MRR: {results['MRR']:.4f}")

    if upload:
        encoder.upload_model(args.upload_name)
