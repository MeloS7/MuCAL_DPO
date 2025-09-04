import json
import argparse
import wandb
import torch
from typing import Dict
import transformers
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    EarlyStoppingCallback,
)
from datasets import Dataset
from trl import SFTTrainer, SFTConfig, setup_chat_format, DataCollatorForCompletionOnlyLM

def read_jsonl_file(file_path):
    # Read JSONL file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return data

def make_hf_dataset(data):
    # Make HF dataset
    dataset = Dataset.from_list(data)
    return dataset

def create_conversation(sample, system_prompt):
    # Create ChatML format
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample['graph']},
            {"role": "assistant", "content": sample["text"]}
        ]
    }

def train_prompt_completion(dataset_train, dataset_dev, model, tokenizer, args):
    # 初始化wandb
    wandb.init(
        project=args.wandb_project,
        name=f"sft-{args.model_name.split('/')[-1]}-{args.mode}",
        config=vars(args)
    )

    # 设置训练参数
    training_args = SFTConfig(
        output_dir=args.save_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.valid_batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,  # 只保存最后3个checkpoint
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        remove_unused_columns=True,
        disable_tqdm=False,  # 显示进度条
        max_seq_length=1024,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_dev,
        args=training_args,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=5,
                early_stopping_threshold=0.001
            )
        ],
    )

    # Print trainer information
    train_result = trainer.train()
    
    # Save final model
    trainer.save_model(f"{args.save_dir}/{args.model_name.split('/')[-1]}-{args.mode}")
    
    # Save training state
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Evaluate
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    # Close wandb
    wandb.finish()

def train_completion_only(dataset_train, dataset_dev, model, tokenizer, args):
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=f"sft-{args.model_name.split('/')[-1]}-{args.mode}",
        config=vars(args)
    )
        
    # Set training parameters
    training_args = SFTConfig(
        output_dir=args.save_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.valid_batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        remove_unused_columns=True,
        disable_tqdm=False,
        max_seq_length=1024,
    )

    # Qwen2.5's response template
    model_name = args.model_name
    if "qwen" in model_name.lower():
        response_template = "<|im_start|>assistant\n"
        instruction_template = "<|im_start|>system\n"
    elif "llama" in model_name.lower():
        response_template = "<|start_header_id|>user<|end_header_id|>"
        instruction_template = "<|start_header_id|>system<|end_header_id|>"
    elif "smollm" in model_name.lower():
        response_template = "<|im_start|>assistant\n"
        instruction_template = "<|im_start|>system\n"
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_dev,
        args=training_args,
        data_collator=collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=5,
                early_stopping_threshold=0.001
            )
        ],
    )
    
    # Train model
    train_result = trainer.train()
    
    # Save final model
    trainer.save_model(f"{args.save_dir}/{args.model_name.split('/')[-1]}-{args.mode}")
    
    # Save training state
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Evaluate
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    # Close wandb
    wandb.finish()

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-it", "--input_train_path", type=str, help="Path to the input JSONL file.", default="data/SFT/train/filtered_kelm_by_Q1_DQE_train.jsonl")
    parser.add_argument("-id", "--input_dev_path", type=str, help="Path to the input JSONL file.", default="data/SFT/dev/filtered_kelm_by_Q1_DQE_dev.jsonl")
    parser.add_argument("-o", "--output_path", type=str, help="Path to the output file.", default="data/generations/inference_only/kelm_clean_test_all_en_qwen2.5-1.5B-Instruct-512_0-shot.txt")
    parser.add_argument("-tb", "--train_batch_size", type=int, help="Train batch size.", default=16)
    parser.add_argument("-vb", "--valid_batch_size", type=int, help="Valid batch size.", default=32)
    parser.add_argument("-lr", "--learning_rate", type=float, help="Learning rate.", default=1e-5)
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs.", default=10)
    parser.add_argument("-model", "--model_name", type=str, help="Model name.", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("-mode", "--mode", type=str, help="Completion-Only or Prompt-Completion.", default="Completion-Only")
    parser.add_argument("--wandb_project", type=str, default="RDF-to-Text-SFT", help="WandB project name")
    parser.add_argument("--save_dir", type=str, default="ckpt", help="Directory to save the model")
    args = parser.parse_args()

    # Receive arguments
    input_train_path = args.input_train_path
    input_dev_path = args.input_dev_path
    model_name = args.model_name
    train_batch_size = args.train_batch_size
    valid_batch_size = args.valid_batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    mode = args.mode

    # Read JSONL file
    data_train = read_jsonl_file(input_train_path)
    data_dev = read_jsonl_file(input_dev_path)

    # Create HF Dataset
    dataset_train = make_hf_dataset(data_train)
    dataset_dev = make_hf_dataset(data_dev)

    print(f"Train dataset size: {len(dataset_train)}")
    print(f"Dev dataset size: {len(dataset_dev)}")

    # Define system prompt
    system_prompt = f"""You are a RDF to text converter. Users will give you a RDF graph represented as a set of triples. Each triple provides a fact in the form '[S] subject [P] predicate [O] object'. Please convert this graph into fluent and natural language text. The output should be a concise and coherent description, consisting of one or a few sentences. Ensure that:
1. All facts from the graph are included in the description.
2. The text is fluent, natural, and easy to understand.
3. There is no repetition or missing details."""

    # Convert dataset to conversation format
    dataset_train = dataset_train.map(create_conversation, remove_columns=dataset_train.column_names, fn_kwargs={"system_prompt": system_prompt}, batched=False)
    dataset_dev = dataset_dev.map(create_conversation, remove_columns=dataset_dev.column_names, fn_kwargs={"system_prompt": system_prompt}, batched=False)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        # attn_implementation="flash_attention_2", # Accelerate attention
        cache_dir="ckpt_qwen",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Default padding side is right, which is for training
    # For inference, we need to change it to left !!!
    tokenizer.padding_side = "right"

    # Print chat template
    print(tokenizer.chat_template)

    # If model is llama series, we add a pad token
    if "llama" in model_name.lower():
        if tokenizer.pad_token is None:
            print(f"Adding pad token as '<pad>'")
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="<pad>"),
                tokenizer=tokenizer,
                model=model,
            )
            model.config.pad_token_id = tokenizer.pad_token_id
        # Set up the chat format with default 'chatml' format
        # model, tokenizer = setup_chat_format(model, tokenizer)  

    # Train model
    if mode == "Completion-Only":
        train_completion_only(dataset_train, dataset_dev, model, tokenizer, args)
    elif mode == "Prompt-Completion":
        train_prompt_completion(dataset_train, dataset_dev, model, tokenizer, args)
    

if __name__ == "__main__":
    main()