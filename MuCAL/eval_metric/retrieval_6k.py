import re
import os
import sys
import json
import torch
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gc

# Add current directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.append(project_root)

def merge_json_dict(data, json_dict, lang):
    for idx, item in enumerate(data):
        assert item["graph"] == json_dict[idx]["graph"] and item["index"] == json_dict[idx]["index"], "The graph or index is not the same"
        item["score_"+lang] = json_dict[idx]["score_"+lang]
    return data

def parse_lang(file_name):
    return file_name.split("_")[-1].split(".")[0]

def load_all_json_file(folder_path):
    # Read all json file in this file folder
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    data = None

    # Load the first json file
    # Analyze language from file name
    with open(os.path.join(folder_path, json_files[0]), 'r') as f:
        json_dict = json.load(f)
        data = json_dict

    # Load the rest json files
    for json_file in json_files[1:]:
        lang = parse_lang(json_file)
        assert lang in ["ar", "en", "es", "fr", "ru", "zh"], "Language not supported"
        with open(os.path.join(folder_path, json_file), 'r') as f:
            json_dict = json.load(f)
            # Merge the json_dict into data
            data = merge_json_dict(data, json_dict, lang)

    # Parse good lexs of each language
    lexs = []
    lang_order = ["en", "ar", "es", "fr", "ru", "zh"]
    for item in data:
        if item["type"] == "good":
            for lang in lang_order:
                lexs.append(item["ref"][lang][0]['lex'])

    print(f"There are {len(lexs)} good lexs")

    # Parse all graph
    graphs = []
    type_order = ["good", "replace_pred", "replace_obj", "added", "swapped", "removed"]
    type_dict = {}
    for type in type_order:
        type_dict[type] = 0
        for item in data:
            if item["type"] == type:
                graphs.append(item["graph"])
                type_dict[type] += 1

    print(f"There are {len(graphs)} graphs")

    for type in type_order:
        print(f"{type}: {type_dict[type]}")
    return graphs, lexs

def compute_cosine_similarity(text_embeds, graph_embeds):
    # Normalize the embeddings to unit vectors
    text_embeds = torch.nn.functional.normalize(text_embeds, p=2, dim=1)
    graph_embeds = torch.nn.functional.normalize(graph_embeds, p=2, dim=1)
    
    # Cosine similarity as dot product of normalized vectors
    return torch.mm(text_embeds, graph_embeds.t())

def compute_text_to_graph_biencoder(graphs, lexs, model_path, batch_size=32, n=[1]):
    num_graphs = len(graphs)
    num_texts = len(lexs)
    languages = ["en", "ar", "es", "fr", "ru", "zh"]
    
    
    # 初始化指标存储
    metrics = {lang: {"mrr": 0.0, "hits": {k: 0 for k in n}} for lang in languages}
    num_samples_per_lang = {lang: 0 for lang in languages}

    # 初始化encoder
    from sentence_transformers import SentenceTransformer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = SentenceTransformer(model_path).to(device)
    
    with torch.no_grad():
        # Encode all graphs at once since they are constant across languages
        # Encode in batch
        graph_embeds = []
        for i in range(0, num_graphs, batch_size):
            end_idx = min(i + batch_size, num_graphs)
            batch_embeds = encoder.encode(graphs[i:end_idx], convert_to_tensor=True, device=device)
            graph_embeds.append(batch_embeds)
        
        # 将所有batch的embeds拼接起来
        graph_embeds = torch.cat(graph_embeds, dim=0)

        # 对每个text进行处理
        for i in tqdm(range(num_texts)):
            lang_idx = i % 6
            graph_idx = i // 6  # 每6个文本（不同语言）对应同一个图
            current_lang = languages[lang_idx]
            
            # 确保graph_idx在有效范围内
            if graph_idx >= len(graphs):
                continue
            
            # 编码当前text
            text_embed = encoder.encode([lexs[i]], convert_to_tensor=True, device=device)
            
            # 计算相似度
            similarities = compute_cosine_similarity(text_embed, graph_embeds)
            
            # 计算排名
            sorted_indices = similarities.squeeze().argsort(descending=True)
            correct_rank = (sorted_indices == graph_idx).nonzero(as_tuple=True)[0].item() + 1
            
            # 更新指标
            metrics[current_lang]["mrr"] += 1.0 / correct_rank
            for k in n:
                if correct_rank <= k:
                    metrics[current_lang]["hits"][k] += 1
            num_samples_per_lang[current_lang] += 1
    
    # 计算每种语言的平均值
    results = {}
    for lang in languages:
        num_samples = num_samples_per_lang[lang]
        lang_metrics = {
            "mrr": metrics[lang]["mrr"] / num_samples,
            "hits": {k: metrics[lang]["hits"][k] / num_samples for k in n}
        }
        results[lang] = lang_metrics
    
    # 计算所有语言的平均值
    overall_metrics = {
        "mrr": sum(lang_metrics["mrr"] for lang_metrics in results.values()) / len(languages),
        "hits": {
            k: sum(lang_metrics["hits"][k] for lang_metrics in results.values()) / len(languages)
            for k in n
        }
    }
    results["overall"] = overall_metrics
    
    return results

def compute_text_to_graph_crossencoder(graphs, lexs, model_path, batch_size=64, n=[1]):
    num_graphs = len(graphs)  # 5.8k
    num_texts = len(lexs)     # 6k
    languages = ["en", "ar", "es", "fr", "ru", "zh"]
    
    # 初始化指标存储
    metrics = {lang: {"mrr": 0.0, "hits": {k: 0 for k in n}} for lang in languages}
    num_samples_per_lang = {lang: 0 for lang in languages}

    # Initialize encoder
    from src.contrastive.CrossEncoder.CrossEncoder import CrossEncoder
    cross_encoder = CrossEncoder.load(model_path)

    def predict_batched(cross_encoder, text, graphs, batch_size=64):
        all_scores = []
        for i in range(0, len(graphs), batch_size):
            end_idx = min(i + batch_size, len(graphs))
            batch_graphs = graphs[i:end_idx]
            batch_texts = [text] * len(batch_graphs)  # 复制当前文本
            scores = cross_encoder.predict(batch_texts, batch_graphs)
            # 确保scores是在CPU上的numpy数组
            if isinstance(scores, torch.Tensor):
                scores = scores.detach().cpu().numpy()
            all_scores.extend(scores)
        return np.array(all_scores)

    # 对所有文本和所有Graphs进行评估
    with torch.no_grad():
        for i in tqdm(range(num_texts)): 
            lang_idx = i % 6
            graph_idx = i // 6 
            current_lang = languages[lang_idx]
            current_text = lexs[i]
            
            # 计算当前文本与所有graph的得分
            scores = predict_batched(
                cross_encoder, 
                current_text, 
                graphs,
                batch_size=batch_size
            )
            
            # 计算在全部graphs中的排名
            sorted_indices = np.argsort(scores)[::-1]  # 降序排序
            correct_rank = np.where(sorted_indices == graph_idx)[0][0] + 1
            
            # 更新指标
            metrics[current_lang]["mrr"] += 1.0 / correct_rank
            for k in n:
                if correct_rank <= k:
                    metrics[current_lang]["hits"][k] += 1
            num_samples_per_lang[current_lang] += 1

    # 计算每种语言的平均值
    results = {}
    for lang in languages:
        num_samples = num_samples_per_lang[lang]  # 应该是1000
        lang_metrics = {
            "mrr": metrics[lang]["mrr"] / num_samples,
            "hits": {k: metrics[lang]["hits"][k] / num_samples for k in n}
        }
        results[lang] = lang_metrics
    
    # 计算overall（所有语言的平均值）
    overall_mrr = sum(results[lang]["mrr"] for lang in languages) / len(languages)
    overall_hits = {
        k: sum(results[lang]["hits"][k] for lang in languages) / len(languages)
        for k in n
    }
    results["overall"] = {
        "mrr": overall_mrr,
        "hits": overall_hits
    }

    return results


def compute_text_to_graph_classifier(graphs, lexs, model_path, batch_size=64, n=[1]):
    num_graphs = len(graphs)  # 5.8k
    num_texts = len(lexs)     # 6k
    languages = ["en", "ar", "es", "fr", "ru", "zh"]
    
    # 初始化指标存储
    metrics = {lang: {"mrr": 0.0, "hits": {k: 0 for k in n}} for lang in languages}
    num_samples_per_lang = {lang: 0 for lang in languages}

    # Initialize encoder
    from src.regression.in_batch.CrossEncoderClassifier import CrossEncoderClassifier
    cross_encoder = CrossEncoderClassifier.load(model_path)

    def predict_batched(cross_encoder, text, graphs, batch_size=64):
        all_scores = []
        for i in range(0, len(graphs), batch_size):
            end_idx = min(i + batch_size, len(graphs))
            batch_graphs = graphs[i:end_idx]
            batch_texts = [text] * len(batch_graphs)  # 复制当前文本
            scores = cross_encoder.predict(batch_texts, batch_graphs)
            # 确保scores是在CPU上的numpy数组
            if isinstance(scores, torch.Tensor):
                scores = scores.detach().cpu().numpy()
            all_scores.extend(scores)
        return np.array(all_scores)

    # 对所有文本和所有Graphs进行评估
    with torch.no_grad():
        for i in tqdm(range(num_texts)): 
            lang_idx = i % 6
            graph_idx = i // 6 
            current_lang = languages[lang_idx]
            current_text = lexs[i]
            
            # 计算当前文本与所有graph的得分
            scores = predict_batched(
                cross_encoder, 
                current_text, 
                graphs,
                batch_size=batch_size
            )
            
            # 计算在全部graphs中的排名
            sorted_indices = np.argsort(scores)[::-1]  # 降序排序
            correct_rank = np.where(sorted_indices == graph_idx)[0][0] + 1
            
            # 更新指标
            metrics[current_lang]["mrr"] += 1.0 / correct_rank
            for k in n:
                if correct_rank <= k:
                    metrics[current_lang]["hits"][k] += 1
            num_samples_per_lang[current_lang] += 1

    # 计算每种语言的平均值
    results = {}
    for lang in languages:
        num_samples = num_samples_per_lang[lang]  # 应该是1000
        lang_metrics = {
            "mrr": metrics[lang]["mrr"] / num_samples,
            "hits": {k: metrics[lang]["hits"][k] / num_samples for k in n}
        }
        results[lang] = lang_metrics
    
    # 计算overall（所有语言的平均值）
    overall_mrr = sum(results[lang]["mrr"] for lang in languages) / len(languages)
    overall_hits = {
        k: sum(results[lang]["hits"][k] for lang in languages) / len(languages)
        for k in n
    }
    results["overall"] = {
        "mrr": overall_mrr,
        "hits": overall_hits
    }

    return results

def compute_text_to_graph_FCT(graphs, lexs, n=[1]):
    num_graphs = len(graphs)  # 5.8k
    num_texts = len(lexs)     # 6k
    languages = ["en", "ar", "es", "fr", "ru", "zh"]
    
    # 初始化指标存储
    metrics = {lang: {"mrr": 0.0, "hits": {k: 0 for k in n}} for lang in languages}
    num_samples_per_lang = {lang: 0 for lang in languages}

    def predict_batched(text, graphs):
        batch_texts = [text] * len(graphs)
        batch_graphs = graphs
        scores = factspotter(batch_texts, batch_graphs)
        return np.array(scores)

    # 对所有文本和所有Graphs进行评估
    with torch.no_grad():
        for i in tqdm(range(num_texts)): 
            lang_idx = i % 6
            graph_idx = i // 6 
            current_lang = languages[lang_idx]
            current_text = lexs[i]

            if current_lang != "en":
                continue
            
            # 计算当前文本与所有graph的得分
            scores = predict_batched(
                current_text, 
                graphs
            )
            
            # 计算在全部graphs中的排名
            sorted_indices = np.argsort(scores)[::-1]  # 降序排序
            correct_rank = np.where(sorted_indices == graph_idx)[0][0] + 1
            
            # 更新指标
            metrics[current_lang]["mrr"] += 1.0 / correct_rank
            for k in n:
                if correct_rank <= k:
                    metrics[current_lang]["hits"][k] += 1
            num_samples_per_lang[current_lang] += 1

    # 计算每种语言的平均值
    results = {}
    for lang in languages:
        if lang != "en":
            continue
        num_samples = num_samples_per_lang[lang]  # 应该是1000
        lang_metrics = {
            "mrr": metrics[lang]["mrr"] / num_samples,
            "hits": {k: metrics[lang]["hits"][k] / num_samples for k in n}
        }
        results[lang] = lang_metrics
    

    results["overall"] = {
        "mrr": 0.0,
        "hits": {k: 0 for k in n}
    }

    return results

def factspotter(hypothesis, graphs):
        def _text(text):
            return text.strip()

        def _table_linearize(linearized_graph):
            '''
            Replace all special seperators ([S], [P], [O]) with ','
            '''
            # Split the graph into triples by [S]
            triples = linearized_graph.split("[S]")
            triples = triples[1:]

            # Replace all special seperators ([O], [P]) with ','
            linearized_triples = []
            for triple in triples:
                triple_replaced = triple.replace(" [O]", ",").replace(" [P]", ",").strip()
                linearized_triples.append(triple_replaced)
            return linearized_triples

        def sentence_cls_score(input_strings, predicate_cls_model, predicate_cls_tokenizer):
            tokenized_cls_input = predicate_cls_tokenizer(input_strings, truncation=True, padding=True,
                                                        return_token_type_ids=True)
            input_ids = torch.Tensor(tokenized_cls_input['input_ids']).long().to(torch.device("cuda"))
            token_type_ids = torch.Tensor(tokenized_cls_input['token_type_ids']).long().to(torch.device("cuda"))
            attention_mask = torch.Tensor(tokenized_cls_input['attention_mask']).long().to(torch.device("cuda"))
            prev_cls_output = predicate_cls_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            softmax_cls_output = torch.softmax(prev_cls_output.logits, dim=1, )
            return softmax_cls_output

        def mapped_score_in_batch(gt_cls_input, gt_count, model, tokenizer, batch_size=16):
            # Calculate GT Entailment score in batch
            torch.cuda.empty_cache()
            batch_size = batch_size
            # get cls score for each batch of GT
            batched_gt_cls = [gt_cls_input[i:i + batch_size] for i in range(0, len(gt_cls_input), batch_size)]
            # cls_gt = []
            cls_entail_score = []
            for golden_batch in tqdm(batched_gt_cls, 'GT CLS Progress'):
                tmp_cls = sentence_cls_score(golden_batch, model, tokenizer)
                cls_entail_score.extend([float(x[0]) for x in tmp_cls])

            # Map the count number into the scores
            index = 0
            avg_mapped_score = []
            for nb in gt_count:
                avg = sum(cls_entail_score[index:index+nb]) / nb
                avg_mapped_score.append(avg)
                index += nb

            assert len(avg_mapped_score) == len(gt_count)

            # factSpotter_Score = sum(avg_mapped_score) / len(avg_mapped_score)
            return avg_mapped_score
        
        def make_pairs(preds, graphs):
            pairs = []
            graph_size = []
            for i, pred in enumerate(preds):
                pairs.extend([(pred, triple) for triple in graphs[i]])
                graph_size.append(len(graphs[i]))
            return pairs, graph_size

        # Processe pred and graph
        predictions = [_text(pred) for pred in hypothesis]
        graphs = [_table_linearize(table) for table in graphs]
        assert len(predictions) == len(graphs)

        # Combine pred and graph as pairs
        pairs, graph_size = make_pairs(predictions, graphs)

        # Load Models
        tokenizer = AutoTokenizer.from_pretrained("Inria-CEDAR/FactSpotter-DeBERTaV3-Base")
        model = AutoModelForSequenceClassification.from_pretrained("Inria-CEDAR/FactSpotter-DeBERTaV3-Base")
        model.to(torch.device("cuda"))

        # Compute FactSpotter score
        factscore = mapped_score_in_batch(pairs, graph_size, model, tokenizer, 16)

        # Unload model
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        return factscore

def DQE(hypothesis, graphs, path='/home/ysong/intern24_eval/webnlg_toolkit/eval/metrics/DQE'):
    # Add DQE directory into the working repo
    sys.path.append(path)
    from questeval.questeval_metric import QuestEval
    import nltk

    def _text(text):
        return " ".join(nltk.word_tokenize(text)).strip()

    def _table_linearize(linearized_graph):
            '''
            Replace all special seperators ([S], [P], [O]) with '|'
            '''
            # Split the graph into triples by [S]
            triples = linearized_graph.split("[S]")
            triples = triples[1:]

            # Replace all special seperators ([O], [P]) with '|'
            linearized_triples = []
            for triple in triples:
                triple_replaced = triple.replace(" [O]", "|").replace(" [P]", "|").strip()
                linearized_triples.append(triple_replaced)
            return linearized_triples
    
    # Load Model
    questeval = QuestEval(task="data2text", no_cuda=False)
    
    # Processe pred and graph
    predictions = [_text(pred) for pred in hypothesis]
    graphs = [_table_linearize(table) for table in graphs]
    assert len(predictions) == len(graphs)

    # Compute DQE score
    scores = questeval.corpus_questeval(
                hypothesis=predictions, 
                sources=graphs
            )

    # Unload model
    del questeval
    gc.collect()
    torch.cuda.empty_cache()
    
    return scores


def compute_text_to_graph_DQE(graphs, lexs, n=[1]):
    num_graphs = len(graphs)  # 5.8k
    num_texts = len(lexs)     # 6k
    languages = ["en", "ar", "es", "fr", "ru", "zh"]
    
    # 初始化指标存储
    metrics = {lang: {"mrr": 0.0, "hits": {k: 0 for k in n}} for lang in languages}
    num_samples_per_lang = {lang: 0 for lang in languages}

    def predict_batched(text, graphs):
        batch_texts = [text] * len(graphs)
        batch_graphs = graphs
        scores = DQE(batch_texts, batch_graphs)
        return np.array(scores)

    # 对所有文本和所有Graphs进行评估
    with torch.no_grad():
        for i in tqdm(range(num_texts)): 
            lang_idx = i % 6
            graph_idx = i // 6 
            current_lang = languages[lang_idx]
            current_text = lexs[i]

            # if current_lang != "en":
            #     continue
            
            # 计算当前文本与所有graph的得分
            scores = predict_batched(
                current_text, 
                graphs
            )
            
            # 计算在全部graphs中的排名
            sorted_indices = np.argsort(scores)[::-1]  # 降序排序
            correct_rank = np.where(sorted_indices == graph_idx)[0][0] + 1
            
            # 更新指标
            metrics[current_lang]["mrr"] += 1.0 / correct_rank
            for k in n:
                if correct_rank <= k:
                    metrics[current_lang]["hits"][k] += 1
            num_samples_per_lang[current_lang] += 1

    # 计算每种语言的平均值
    results = {}
    for lang in languages:
        if lang != "en":
            continue
        num_samples = num_samples_per_lang[lang]  # 应该是1000
        lang_metrics = {
            "mrr": metrics[lang]["mrr"] / num_samples,
            "hits": {k: metrics[lang]["hits"][k] / num_samples for k in n}
        }
        results[lang] = lang_metrics
    

    results["overall"] = {
        "mrr": 0.0,
        "hits": {k: 0 for k in n}
    }

    return results

def visualize_retrieval_results(results):
    # Visualize the results
    for lang, metrics in results.items():
        print(f"\n{lang.upper()} Results:")
        print(f"MRR: {metrics['mrr']:.4f}")
        for k, hit_rate in metrics["hits"].items():
            print(f"Hits@{k}: {hit_rate:.4f}")

def main():
    # Argument parser
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--folder_path", help="Path to processed json file", default="data/test/corrupt/crossencoder_score/ep10_bs4")
    argParser.add_argument("-m", "--model", help="Model name", default="ckpt/bi_ep10_bs32_lr2e5_cosine_annealing_hard_neg_1_best")

    args = argParser.parse_args()
    folder_path = args.folder_path
    model_path = args.model

    print(f"Model path: {model_path}")
    print(f"Folder path: {folder_path}")
    
    # Check if the file folder exists
    if not os.path.exists(folder_path):
        print(f"File {folder_path} does not exist")
        return
    
    # Parse graphs and lexs
    # Text: 6k, Graph: 5.8k
    # Text = 1k(en) + 1k(ar) + 1k(es) + 1k(fr) + 1k(ru) + 1k(zh)
    # Graph = 1k(good) + 1k(replace_pred) + 1k(replace_obj) + 1k(added) + 1k(swapped) + 0.8k(removed)
    graphs, lexs = load_all_json_file(folder_path)

    # Compute the retrieval score (Rank@1 and MRR)
    results = compute_text_to_graph_biencoder(graphs, lexs, model_path)
    # results = compute_text_to_graph_crossencoder(graphs, lexs, model_path)
    # results = compute_text_to_graph_classifier(graphs, lexs, model_path)
    # results = compute_text_to_graph_FCT(graphs, lexs)
    # results = compute_text_to_graph_DQE(graphs, lexs)
    
    # Visualize the results
    visualize_retrieval_results(results)





if __name__ == "__main__":
    main()