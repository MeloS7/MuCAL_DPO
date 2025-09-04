import re
import gc
import json
import torch
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def load_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [line.strip() for line in file.readlines()]
    return data

def from_camel(s):
    return re.sub('([A-Z])', r' \1', s).lower()

def to_camel(s):
    s = re.sub(r"(_|-)+", " ", s).title().replace(" ", "")
    
    # Join the string, ensuring the first letter is lowercase
    return ''.join([s[0].lower(), s[1:]])

def compute_score(hypothesis, graphs, model_name='OneFly7/biencoder_ep10_bs32_lr2e5_cosine_annealing_hard_neg_2'):
    def _text(text):
        # Remove the '\n' and extra spaces
        return text.strip()
    
    def _table_linearize(table):
        '''
        Linearize table data into one string
        '''
        # Convert strings to list
        list_table = eval(table)

        graph = [triple.split(" | ") for triple in list_table]
        linearized_triple = ""
        for triple in graph:
            subject, predicate, object = triple
            subject = subject.strip('\"')
            predicate = to_camel(from_camel(predicate))
            object = object.strip('\"')
            linearized_triple += " [S] " + subject + " [P] " + predicate + " [O] " + object
        return linearized_triple.replace("_", " ").strip()
    
    def encode_batched(model, sentences):
        # Encode sentences in batches and show progress
        return model.encode(sentences, show_progress_bar=True)
        
    def compute_cosine(embed_pred, embed_graph):
        import numpy as np
        # Normalize the vectors to unit vectors (divide each vector by its norm)
        norm_embed_pred = embed_pred / np.linalg.norm(embed_pred, axis=1, keepdims=True)
        norm_embed_graph = embed_graph / np.linalg.norm(embed_graph, axis=1, keepdims=True)

        # Compute the cosine similarity only for i=j
        cosine_scores = np.sum(norm_embed_pred * norm_embed_graph, axis=1)

        # Convert numpy array to list
        return cosine_scores.tolist()
    
    # Load model
    model = SentenceTransformer(model_name)

    predictions = [_text(pred) for pred in hypothesis]
    graphs = [_table_linearize(table) for table in graphs]
    assert len(predictions) == len(graphs)

    # Compute embeddings
    embed_pred = encode_batched(model, predictions)
    embed_graph = encode_batched(model, graphs)

    # Compute similarity
    cosine_scores = compute_cosine(embed_pred, embed_graph)

    # Unload model
    del model, embed_graph, embed_pred, predictions
    gc.collect()
    torch.cuda.empty_cache()

    return cosine_scores, graphs

def make_preference_dataset(graphs, scores_txt1, scores_txt2, data_txt1, data_txt2):
    dataset = []
    stats = {"from_txt1": 0, "from_txt2": 0}
    for i, graph in enumerate(graphs):
        # Compare the scores, the higher score is "chosen", the lower score is "rejected"
        if scores_txt1[i] > scores_txt2[i]:
            dataset.append({
                "graph": graph,
                "chosen": data_txt1[i],
                "rejected": data_txt2[i],
                "chosen_score": scores_txt1[i],
                "rejected_score": scores_txt2[i],
                "preference": 1
            })
            stats["from_txt1"] += 1
        else:
            dataset.append({
                "graph": graph,
                "chosen": data_txt2[i],
                "rejected": data_txt1[i],
                "chosen_score": scores_txt2[i],
                "rejected_score": scores_txt1[i],
                "preference": 2
            })
            stats["from_txt2"] += 1
    return dataset, stats

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Model name.", default="OneFly7/biencoder_ep10_bs32_lr2e5_cosine_annealing_hard_neg_2")
    parser.add_argument("--input_graph_file", type=str, help="Input graph file.", default="data/RL/train/kelm_Q1_graphs.txt")
    parser.add_argument("--input_txt1_file", type=str, help="Input txt1 file.", default="data/RL/train/kelm_Q1_Prop_Overlap_Dir.txt")
    parser.add_argument("--input_txt2_file", type=str, help="Input txt2 file.", default="data/RL/train/filtered_kelm_by_Q1_DQE_train.txt")
    parser.add_argument("--output_file", type=str, help="Output file.", default="data/RL/train/kelm_Q1_graphs_preference.jsonl")
    args = parser.parse_args()

    # Load data
    data_txt1 = load_txt_file(args.input_txt1_file)
    data_txt2 = load_txt_file(args.input_txt2_file)
    graphs = load_txt_file(args.input_graph_file)
    model_name = args.model_name

    assert len(data_txt1) == len(data_txt2), "The length of data_txt1 and data_txt2 is not the same"
    assert len(data_txt1) == len(graphs), "The length of data_txt1 and graphs is not the same"
    print(f"The length of data_txt1 is {len(data_txt1)}")
    print(f"The length of data_txt2 is {len(data_txt2)}")
    print(f"The length of graphs is {len(graphs)}")

    # Compute metric score
    scores_txt1, linearized_graphs = compute_score(data_txt1, graphs, model_name)
    scores_txt2, linearized_graphs = compute_score(data_txt2, graphs, model_name)

    # Make preference dataset
    preference_dataset, stats = make_preference_dataset(linearized_graphs, scores_txt1, scores_txt2, data_txt1, data_txt2)

    # Print stats
    print(f"From txt1: {stats['from_txt1']}")
    print(f"From txt2: {stats['from_txt2']}")

    # Save to jsonl file
    with open(args.output_file, 'w', encoding='utf-8') as file:
        for item in preference_dataset:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()