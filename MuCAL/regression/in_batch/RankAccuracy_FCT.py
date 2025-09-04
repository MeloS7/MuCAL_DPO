import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
import gc
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class RankAccuracy_FCT:
    def __init__(self, graph_text_dict, lang_set):
        super(RankAccuracy_FCT, self).__init__()
        self.graphs = graph_text_dict['graphs']
        self.texts = graph_text_dict['texts']
        self.lang_set = lang_set

    def factspotter(self, hypothesis, graphs):
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
        
    def compute_text_to_multilang_graph_rank_n(self, n=[1, 3, 10], batch_size=16):
        num_graph = len(self.graphs)
        num_text = len(self.texts) // len(self.lang_set)
        assert num_graph == num_text
        accuracy_results = {k: 0 for k in n}
        total_reciprocal_rank = 0.0

        with torch.no_grad():
            # 预计算所有得分
            all_scores = []
            for i in tqdm(range(num_text)):
                batch_scores = []
                batched_texts = self.texts[i * len(self.lang_set):(i + 1) * len(self.lang_set)]
                
                for batch_start in range(0, num_graph, batch_size):
                    batch_end = min(batch_start + batch_size, num_graph)
                    current_graphs = self.graphs[batch_start:batch_end]
                    batch_graphs = []
                    batch_texts = []
                    
                    for graph in current_graphs:
                        batch_graphs.extend([graph] * len(self.lang_set))
                        batch_texts.extend(batched_texts)
                    
                    with autocast():
                        scores = self.factspotter(batch_texts, batch_graphs)
                    batch_scores.extend(scores.view(-1, len(self.lang_set)).mean(dim=1).tolist())

                    # 释放内存
                    del batch_texts, batch_graphs, scores
                    torch.cuda.empty_cache()
                
                all_scores.append(batch_scores)
                del batch_scores
                torch.cuda.empty_cache()

            # 使用预计算的得分进行排序和计算
            for i, similarities in enumerate(all_scores):
                sorted_indices = sorted(range(len(similarities)), 
                                     key=lambda k: similarities[k], 
                                     reverse=True)
                correct_rank = sorted_indices.index(i) + 1
                total_reciprocal_rank += 1.0 / correct_rank
                
                for rank in n:
                    if correct_rank <= rank:
                        accuracy_results[rank] += 1

        # 计算MRR
        mrr = total_reciprocal_rank / num_text

        # 将结果平均到文本数量
        for k in accuracy_results:
            accuracy_results[k] /= num_text

        return accuracy_results, mrr

    def compute_text_to_monolang_graph_rank_n(self, n=[1, 3, 10], batch_size=2000):
        num_graph = len(self.graphs)
        num_text = len(self.texts) // len(self.lang_set)
        lang_results = {lang: {k: 0 for k in n} for lang in self.lang_set}
        lang_mrr = {lang: 0.0 for lang in self.lang_set}

        with torch.no_grad():
            for lang_index, lang in enumerate(self.lang_set):
                # 预计算所有得分
                all_scores = []
                for i in tqdm(range(num_text), desc=f"处理 {lang} 文本"):
                    batch_scores = []
                    current_text = [self.texts[i * len(self.lang_set) + lang_index]]
                    
                    for batch_start in range(0, num_graph, batch_size):
                        batch_end = min(batch_start + batch_size, num_graph)
                        batched_graphs = self.graphs[batch_start:batch_end]
                        batched_texts = current_text * len(batched_graphs)
                        
                        scores = self.factspotter(batched_texts, batched_graphs)
                        batch_scores.extend(scores)

                        # 释放内存
                        del batched_texts, batched_graphs, scores
                        torch.cuda.empty_cache()
                    
                    all_scores.append(batch_scores)
                    del batch_scores
                    torch.cuda.empty_cache()

                # 使用预计算的得分进行排序和计算
                for i, similarities in enumerate(all_scores):
                    sorted_indices = sorted(range(len(similarities)), 
                                         key=lambda k: similarities[k], 
                                         reverse=True)
                    correct_rank = sorted_indices.index(i) + 1
                    lang_mrr[lang] += 1.0 / correct_rank
                    
                    for rank in n:
                        if correct_rank <= rank:
                            lang_results[lang][rank] += 1

                # 计算当前语言的百分比结果
                for rank in n:
                    lang_results[lang][rank] /= num_text
                lang_mrr[lang] /= num_text
                lang_results[lang]['MRR'] = lang_mrr[lang]

        return lang_results, lang_mrr
    

    def compute_graph_to_multilang_text_rank_n(self, n=[1, 3, 10], batch_size=2000):
        num_graph = len(self.graphs)
        num_lang = len(self.lang_set)
        num_text = len(self.texts) // len(self.lang_set)
        assert num_graph == num_text
        accuracy_results = {k: 0 for k in n}
        total_reciprocal_rank = 0.0

        with torch.no_grad():
            # 预计算所有得分
            all_scores = []
            for i in tqdm(range(num_graph)):
                batch_scores = []
                current_graph = [self.graphs[i]] * num_lang
                
                for batch_start in range(0, num_text, batch_size):
                    batch_end = min(batch_start + batch_size, num_text)
                    batch_texts = []
                    for j in range(batch_start, batch_end):
                        batch_texts.extend(self.texts[j * num_lang:(j + 1) * num_lang])
                    batch_graphs = current_graph * (batch_end - batch_start)
                    
                    with autocast():
                        scores = self.encoder.predict(batch_texts, batch_graphs)
                    batch_means = scores.view(-1, num_lang).mean(dim=1).tolist()
                    batch_scores.extend(batch_means)

                    # Free memory
                    del batch_texts, batch_graphs, scores
                    torch.cuda.empty_cache()
                
                all_scores.append(batch_scores)
                del batch_scores
                torch.cuda.empty_cache()

            # 使用预计算的得分进行排序和计算
            for i, similarities in enumerate(all_scores):
                sorted_indices = sorted(range(len(similarities)), 
                                     key=lambda k: similarities[k], 
                                     reverse=True)
                correct_rank = sorted_indices.index(i) + 1

                # Calculate reciprocal rank for MRR
                total_reciprocal_rank += 1.0 / correct_rank

                # Update accuracy results based on ranks
                for rank in n:
                    if correct_rank <= rank:
                        accuracy_results[rank] += 1

        # 计算最终结果
        mrr = total_reciprocal_rank / num_graph
        for k in accuracy_results:
            accuracy_results[k] /= num_graph

        return accuracy_results, mrr
    
    def compute_graph_to_monolang_text_rank_n(self, n=[1, 3, 10], batch_size=2000):
        num_graph = len(self.graphs)
        num_text = len(self.texts) // len(self.lang_set)
        num_lang = len(self.lang_set)
        assert num_graph == num_text
        lang_results = {lang: {k: 0 for k in n} for lang in self.lang_set}
        lang_mrr = {lang: 0.0 for lang in self.lang_set}

        with torch.no_grad():
            for lang_index, lang in enumerate(self.lang_set):
                all_scores = []
                for i in tqdm(range(num_graph)):
                    batch_scores = []
                    current_graph = [self.graphs[i]]
                    all_text_unique_lang = [self.texts[j * num_lang + lang_index] for j in range(num_text)]
                    
                    for batch_start in range(0, num_text, batch_size):
                        batch_end = min(batch_start + batch_size, num_text)
                        current_texts = all_text_unique_lang[batch_start:batch_end]
                        current_graphs = current_graph * len(current_texts)
                        
                        with autocast():
                            scores = self.factspotter(current_texts, current_graphs)
                        batch_scores.extend(scores)

                        # Free memory
                        del current_texts, current_graphs, scores
                        torch.cuda.empty_cache()
                    
                    all_scores.append(batch_scores)
                    del batch_scores
                    torch.cuda.empty_cache()

                # 使用预计算的得分进行排序和计算
                for i, similarities in enumerate(all_scores):
                    sorted_indices = sorted(range(len(similarities)), 
                                         key=lambda k: similarities[k], 
                                         reverse=True)
                    correct_rank = sorted_indices.index(i) + 1
                    lang_mrr[lang] += 1.0 / correct_rank
                    
                    for rank in n:
                        if correct_rank <= rank:
                            lang_results[lang][rank] += 1

                # Convert counts to percentages and calculate the average correct rank
                for k in lang_results[lang]:
                    lang_results[lang][k] /= num_graph
                lang_mrr[lang] /= num_graph
                lang_results[lang]['MRR'] = lang_mrr[lang]

        return lang_results

        


