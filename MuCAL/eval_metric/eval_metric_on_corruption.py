import json
import torch
import argparse
import numpy as np
import nltk
import re
from tqdm import tqdm
import os
import sys

from sentence_transformers import SentenceTransformer

# Add current directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.append(project_root)

def load_json_file(json_file):
    with open(json_file, 'r') as f:
        json_dict = json.load(f)
    return json_dict

def process_data(data, lang='en'):
    predictions = []
    graphs_linears = []
    indices = []
    data_items = []
    for item in data:
        refs = item['ref']
        # We check if the language is in the references
        if lang in refs and refs[lang]:
            # Always use the first reference !!!
            ref_text = refs[lang][0]['lex']
        else:
            # If the language is not in the references, raise an error
            raise ValueError(f"Language {lang} not found in references")
        # Get triples
        triples = item['triples']  # [[subject, predicate, object], ...]
        # 线性化图
        graph_linear = _table_linearize(triples)
        # 将线性化的图添加到数据项
        item['graph'] = graph_linear
        predictions.append(ref_text)
        graphs_linears.append(graph_linear)
        indices.append(item['index'])
        data_items.append(item)
    return predictions, graphs_linears, indices, data_items


def from_camel(s):
    return re.sub('([A-Z])', r' \1', s).lower()

def to_camel(s):
    s = re.sub(r"(_|-)+", " ", s).title().replace(" ", "")
    
    # Join the string, ensuring the first letter is lowercase
    return ''.join([s[0].lower(), s[1:]])

def _table_linearize(table):
        '''
        Linearize table data into one string
        '''
        # print(type(table))
        linearized_triple = ""
        for triple in table:
            subject, predicate, object = triple
            subject = subject.strip('\"')
            predicate = to_camel(from_camel(predicate))
            object = object.strip('\"')
            linearized_triple += " [S] " + subject + " [P] " + predicate + " [O] " + object
        return linearized_triple.replace("_", " ").strip()
    
def biEncoder(hypothesis, graphs, batch_size, model='teven/bi_all_bs192_hardneg_finetuned_WebNLG2017'):
    def _text(text):
        return " ".join(nltk.word_tokenize(text)).strip()
    
    def encode_batched(model, sentences, batch_size=32, device=None):
        return model.encode(sentences, batch_size=batch_size, device=device, show_progress_bar=True)
        
    def compute_cosine(embed_pred, embed_graph):
        import numpy as np
        # Normalize the vectors to unit vectors (divide each vector by its norm)
        norm_embed_pred = embed_pred / np.linalg.norm(embed_pred, axis=1, keepdims=True)
        norm_embed_graph = embed_graph / np.linalg.norm(embed_graph, axis=1, keepdims=True)

        # Compute the cosine similarity only for i=j
        cosine_scores = np.sum(norm_embed_pred * norm_embed_graph, axis=1)
        return cosine_scores
    
    # Load model
    model = SentenceTransformer(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # model = SentenceTransformer('OneFly7/biencoder_ep10_bs64_trans3')

    predictions = [_text(pred) for pred in hypothesis]
    # graphs = [_table_linearize(table) for table in graphs]
    assert len(predictions) == len(graphs)

    # Compute embeddings
    embed_pred = encode_batched(model, predictions, batch_size)
    embed_graph = encode_batched(model, graphs, batch_size)

    # Compute similarity
    cosine_scores = compute_cosine(embed_pred, embed_graph)

    return cosine_scores

def crossEncoder(hypothesis, graphs, batch_size, model='teven/bi_all_bs192_hardneg_finetuned_WebNLG2017'):
    from src.contrastive.CrossEncoder.CrossEncoder import CrossEncoder

    def _text(text):
        return " ".join(nltk.word_tokenize(text)).strip()

    def predict_batched(cross_encoder, hypothesis, graphs, batch_size=8):
        scores = []
        for i in tqdm(range(0, len(hypothesis), batch_size)):
            scores.extend(cross_encoder.predict(hypothesis[i:i+batch_size], graphs[i:i+batch_size]))
        return scores

    # Load model
    cross_encoder = CrossEncoder.load(model)

    predictions = [_text(pred) for pred in hypothesis]
    assert len(predictions) == len(graphs)

    # Compute scores
    scores = predict_batched(cross_encoder, predictions, graphs, batch_size)

    return scores
    

def main():
    # Argument parser
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--file_path", help="Path to processed json file", default="data/test/corrupt/corrupted_all.json")
    argParser.add_argument("-b", "--batch_size", help="Batch size for processing", type=int, default=32)
    argParser.add_argument("-m", "--model_name", help="Model name", type=str, default='teven/bi_all_bs192_hardneg_finetuned_WebNLG2017')
    argParser.add_argument("-mt", "--metric", help="Metric to eval", type=str, default='cross_encoder')
    argParser.add_argument("-s", "--save_path", help="Save path", type=str, default="data/test/corrupt/crossencoder_score/ep10_bs4")
    argParser.add_argument("-f", "--save_file", help="Save file", type=str, default="corrupt_score_cross_ep10_bs4")
    argParser.add_argument("-l", "--language", help="Language", type=str, default="en")

    args = argParser.parse_args()
    file_path = args.file_path
    batch_size = args.batch_size
    model = args.model_name
    metric = args.metric
    save_path = args.save_path
    save_file = args.save_file
    language = args.language

    # Load data
    train_data = load_json_file(file_path)
    
    # Process data
    predictions, graphs_linear, indices, data_items = process_data(train_data, lang=language)
    
    # Compute metric scores
    if metric == 'bi_encoder':
        scores = biEncoder(predictions, graphs_linear, batch_size, model=model)
    elif metric == 'cross_encoder':
        scores = crossEncoder(predictions, graphs_linear, batch_size, model=model)
    else:
        raise ValueError(f"Metric {metric} not supported")
    
    # Add socre into data items
    for data, score in zip(data_items, scores):
        data["score_"+language] = float(score)

    # Save filtered data
    with open(f"{save_path}/{save_file}_{language}.json", "w", encoding='utf-8') as f:
        json.dump(data_items, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()

