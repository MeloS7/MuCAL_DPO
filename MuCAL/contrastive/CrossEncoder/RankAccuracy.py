import torch
from tqdm import tqdm
from torch.cuda.amp import autocast

class RankAccuracy:
    def __init__(self, encoder, graph_text_dict, lang_set):
        super(RankAccuracy, self).__init__()
        self.encoder = encoder
        self.graphs = graph_text_dict['graphs']
        self.texts = graph_text_dict['texts']
        self.lang_set = lang_set
        
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
                        scores = self.encoder.predict(batch_texts, batch_graphs)
                    batch_scores.extend(scores.view(-1, len(self.lang_set)).mean(dim=1).tolist())

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
                total_reciprocal_rank += 1.0 / correct_rank
                
                for rank in n:
                    if correct_rank <= rank:
                        accuracy_results[rank] += 1

        # Compute MRR
        mrr = total_reciprocal_rank / num_text

        # 将结果平均到文本数量
        for k in accuracy_results:
            accuracy_results[k] /= num_text

        return accuracy_results, mrr

    
    def compute_text_to_monolang_graph_rank_n(self, n=[1, 3, 10], batch_size=16):
        num_graph = len(self.graphs)
        num_text = len(self.texts) // len(self.lang_set)
        lang_results = {lang: {k: 0 for k in n} for lang in self.lang_set}
        lang_mrr = {lang: 0.0 for lang in self.lang_set}

        with torch.no_grad():
            for lang_index, lang in enumerate(self.lang_set):
                # 预计算所有得分
                all_scores = []
                for i in tqdm(range(num_text), desc=f"Processing {lang} texts"):
                    batch_scores = []
                    current_text = [self.texts[i * len(self.lang_set) + lang_index]]
                    
                    for batch_start in range(0, num_graph, batch_size):
                        batch_end = min(batch_start + batch_size, num_graph)
                        batched_graphs = self.graphs[batch_start:batch_end]
                        batched_texts = current_text * len(batched_graphs)
                        
                        with autocast():
                            scores = self.encoder.predict(batched_texts, batched_graphs)
                        batch_scores.extend(scores.tolist())

                        # Free memory
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

                    # Calculate reciprocal rank for MRR
                    lang_mrr[lang] += 1.0 / correct_rank
                    
                    for rank in n:
                        if correct_rank <= rank:
                            lang_results[lang][rank] += 1

                # Convert counts to percentages for the current language
                for rank in n:
                    lang_results[lang][rank] /= num_text

                # Calculate Mean Reciprocal Rank (MRR) for the current language
                lang_mrr[lang] /= num_text
                lang_results[lang]['MRR'] = lang_mrr[lang]

        return lang_results

    
    
    def compute_graph_to_multilang_text_rank_n(self, n=[1, 3, 10], batch_size=16):
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
    
    def compute_graph_to_monolang_text_rank_n(self, n=[1, 3, 10], batch_size=16):
        num_graph = len(self.graphs)
        num_text = len(self.texts) // len(self.lang_set)
        num_lang = len(self.lang_set)
        assert num_graph == num_text
        lang_results = {lang: {k: 0 for k in n} for lang in self.lang_set}
        lang_mrr = {lang: 0.0 for lang in self.lang_set}

        with torch.no_grad():
            for lang_index, lang in enumerate(self.lang_set):
                # 预计算所有得分
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
                            scores = self.encoder.predict(current_texts, current_graphs)
                        batch_scores.extend(scores.tolist())

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

        

