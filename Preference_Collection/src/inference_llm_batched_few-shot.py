import json
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

def read_jsonl_file(file_path):
    # Read JSONL file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return data

def make_hf_dataset(data):
    # Make HF dataset
    dataset = Dataset.from_list(data)
    return dataset

def inference_llm(dataset, batch_size=16, model_name="Qwen/Qwen2.5-7B-Instruct", cache_dir=None):
    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2", # Accelerate attention
        cache_dir=cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        padding_side="left",
        cache_dir=cache_dir,
    )
    
    if 'llama' in model_name:
        tokenizer.pad_token = tokenizer.eos_token

    # Define system prompt
    system_prompt = {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}

    # Define instruction
    instruction = f"""The following is a graph represented as a set of triples. Each triple provides a fact in the form '[S] subject [P] predicate [O] object'. Please convert this graph into fluent and natural language text. The output should be a concise and coherent description, consisting of one or a few sentences. Ensure that:
1. All facts from the graph are included in the description.
2. The text is fluent, natural, and easy to understand.
3. There is no repetition or missing details.

Graph:
"""

    # Define few-shot examples
    def add_newlines(graph):
        # We will add newlines to the graph with more than one triple to make it more readable
        graph = graph.split(" [S] ")
        graph = "\n[S] ".join([triple.strip() for triple in graph])
        return graph
    
    graph_example1 = "[S] Jukka Raitala [P] team [O] TSG 1899 Hoffenheim"
    graph_example1 = add_newlines(graph_example1)
    text_example1 = "Jukka Raitala's club is TSG 1899 Hoffenheim."

    # Split triples with newlines
    graph_example2 = "[S] Kentland, Maryland [P] locatedInArea [O] Prince George's County, Maryland [S] Kentland, Maryland [P] country [O] United States [S] Kentland, Maryland [P] instanceOf [O] unincorporated community"
    graph_example2 = add_newlines(graph_example2)
    text_example2 = "Kentland is an unincorporated community located in Prince George's County, Maryland, United States."

    graph_example3 = "[S] Iridium anomaly [P] instanceOf [O] physical phenomenon [S] Iridium anomaly [P] location [O] Cretaceous–Paleogene boundary [S] Iridium anomaly [P] facetOf [O] Iridium [S] Iridium anomaly [P] hasCause [O] Impact event [S] Iridium anomaly [P] facetOf [O] Cretaceous–Paleogene boundary"
    graph_example3 = add_newlines(graph_example3)
    text_example3 = "The iridium anomaly is a physical phenomenon caused by an impact event. It is located at the Cretaceous-Paleogene boundary."

    few_shot_examples = [
        {"role": "user", "content": f"""{instruction}{graph_example1}"""},
        {"role": "assistant", "content": f"""{text_example1}"""},
        {"role": "user", "content": f"""{instruction}{graph_example2}"""},
        {"role": "assistant", "content": f"""{text_example2}"""},
        {"role": "user", "content": f"""{instruction}{graph_example3}"""},
        {"role": "assistant", "content": f"""{text_example3}"""},
    ]

    results = []
    # Iterate over dataset and do the inference in batches
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]
        
        # Iterate over each sample in the batch
        message_batch = []
        for graph in batch["graph"]:
            prompt = f"""{instruction}{add_newlines(graph)}"""
            # Add system prompt, few-shot examples, and user prompt
            single_message = [system_prompt]
            single_message.extend(few_shot_examples)
            single_message.append({"role": "user", "content": prompt})
            message_batch.append(single_message)

        text_batch = tokenizer.apply_chat_template(
            message_batch,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs_batch = tokenizer(text_batch, return_tensors="pt", padding=True).to(model.device)

        generated_ids_batch = model.generate(
            **model_inputs_batch,
            max_new_tokens=512,
        )
        generated_ids_batch = generated_ids_batch[:, model_inputs_batch.input_ids.shape[1]:]

        response_batch = tokenizer.batch_decode(generated_ids_batch, skip_special_tokens=True)
        
        # Remove the '\n' in each response
        response_batch = [response.replace('\n', '\\n') for response in response_batch]

        # Add the response to the results
        results.extend(response_batch)
        print(response_batch[0])

    assert len(results) == len(dataset), f"Inference finished. Total samples: {len(dataset)}, Total batches: {len(results) / batch_size}"
    return results

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, help="Path to the input JSONL file.", default="data/SFT/test/kelm_clean_test_all_en.jsonl")
    parser.add_argument("-o", "--output_path", type=str, help="Path to the output file.", default="data/generations/inference_only/kelm_clean_test_all_en_qwen2.5-1.5B-Instruct-512_0-shot.txt")
    parser.add_argument("-b", "--batch_size", type=int, help="Batch size.", default=16)
    parser.add_argument("--model_name", type=str, help="Model name.", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--cache_dir", type=str, help="Directory to store the downloaded model.", default=None)
    args = parser.parse_args()

    # Read JSONL file
    data = read_jsonl_file(args.input_path)

    # Create HF Dataset
    dataset = make_hf_dataset(data)

    # Inference LLM
    generations = inference_llm(dataset, args.batch_size, args.model_name, args.cache_dir)

    # Save generations
    with open(args.output_path, "w") as f:
        for generation in generations:
            f.write(generation + "\n")

if __name__ == "__main__":
    main()