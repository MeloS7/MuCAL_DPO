import os
import json
import argparse
from together import Together
from datasets import Dataset
from tqdm import tqdm

def read_jsonl_file(file_path):
    # Read JSONL file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return data

def make_hf_dataset(data):
    # Make HF dataset
    dataset = Dataset.from_list(data)
    return dataset

# Define few-shot examples
def add_newlines(graph):
    # We will add newlines to the graph with more than one triple to make it more readable
    graph = graph.split(" [S] ")
    graph = "\n[S] ".join([triple.strip() for triple in graph])
    return graph

def generate_via_api(client, prompt, dataset):
    # Generate text
    results = []
    
    for sample in tqdm(dataset, total=len(dataset)):
        graph = sample["graph"]
        graph = add_newlines(graph)

        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{"role": "user", "content": prompt+graph+"\nText:\n"}],
            temperature=0.7,
            max_tokens=1024,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1.0,
            stop=["<|eot_id|>","<|eom_id|>"],
        )

        generated_text = response.choices[0].message.content

        # Remove the '\n' in each response
        generated_text = generated_text.replace('\n', '\\n')

        # Add the response to the results
        results.append(generated_text)

    assert len(results) == len(dataset), f"Inference finished. Total samples: {len(dataset)}, Total batches: {len(results)}"
    return results



def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, help="Path to the input JSONL file.", default="data/SFT/test/kelm_clean_test_all_en.jsonl")
    parser.add_argument("-o", "--output_path", type=str, help="Path to the output file.")
    args = parser.parse_args()

    TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
    client = Together(api_key=TOGETHER_API_KEY)

    # Read JSONL file
    data = read_jsonl_file(args.input_path)

    # Create HF Dataset
    dataset = make_hf_dataset(data)

    prompt = f"""The following is a graph represented as a set of triples. Each triple provides a fact in the form '[S] subject [P] predicate [O] object'. Please convert this graph into fluent and natural language text. The output should be a concise and coherent description, consisting of one or a few sentences converted from the given graph without any additional information. Ensure that:
1. All facts from the graph are included in the description.
2. The text is fluent, natural, and easy to understand.
3. There is no repetition or missing details.

Here are some examples with different graph sizes:
## Example 1 ##
Graph:
[S] Jukka Raitala [P] team [O] TSG 1899 Hoffenheim
Text:
Jukka Raitala's club is TSG 1899 Hoffenheim.

## Example 2 ##
Graph:
[S] Kentland, Maryland [P] locatedInArea [O] Prince George's County, Maryland
[S] Kentland, Maryland [P] country [O] United States
[S] Kentland, Maryland [P] instanceOf [O] unincorporated community
Text:
Kentland is an unincorporated community located in Prince George's County, Maryland, United States.

## Example 3 ##
Graph:
[S] Iridium anomaly [P] instanceOf [O] physical phenomenon
[S] Iridium anomaly [P] location [O] Cretaceous–Paleogene boundary
[S] Iridium anomaly [P] facetOf [O] Iridium
[S] Iridium anomaly [P] hasCause [O] Impact event
[S] Iridium anomaly [P] facetOf [O] Cretaceous–Paleogene boundary
Text:
The iridium anomaly is a physical phenomenon caused by an impact event. It is located at the Cretaceous-Paleogene boundary.

Graph to convert:
"""
    # Generate text
    generated_texts = generate_via_api(client, prompt, dataset)

    # Save the results
    with open(args.output_path, 'w', encoding='utf-8') as file:
        for text in generated_texts:
            file.write(text + '\n')

if __name__ == "__main__":
    main()