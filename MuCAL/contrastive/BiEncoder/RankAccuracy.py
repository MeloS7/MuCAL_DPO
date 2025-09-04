import torch
from tqdm import tqdm

class RankAccuracy:
    def __init__(self, encoder, graph_text_dict, lang_set):
        super(RankAccuracy, self).__init__()
        self.encoder = encoder
        self.graphs = graph_text_dict['graphs']
        self.texts = graph_text_dict['texts']
        self.lang_set = lang_set
        
    def compute_cosine_similarity(self, text_embeds, graph_embeds):
        # Normalize the embeddings to unit vectors
        text_embeds = torch.nn.functional.normalize(text_embeds, p=2, dim=1)
        graph_embeds = torch.nn.functional.normalize(graph_embeds, p=2, dim=1)
        
        # Cosine similarity as dot product of normalized vectors
        return torch.mm(text_embeds, graph_embeds.t())

    # def compute_cosine_similarity(self, text_embeds, graph_embeds):
    #     # Cosine similarity as dot product of normalized vectors
    #     return torch.mm(text_embeds, graph_embeds.t())
        
    def compute_text_to_multilang_graph_rank_n(self, n=[1, 3, 10]):
        num_graph = len(self.graphs)
        num_text = len(self.texts) // len(self.lang_set)
        assert num_graph == num_text
        accuracy_results = {k: 0 for k in n}
        total_reciprocal_rank = 0.0

        with torch.no_grad():
            # Encode all graphs at once since they are constant across languages
            _, graph_embeds = self.encoder(texts=[], graphs=self.graphs)  # No text input required here
            
            for i in tqdm(range(num_text)):
                # Prepare the batch of texts for each graph, considering the text translations as one unit
                batched_texts = self.texts[i * len(self.lang_set):(i + 1) * len(self.lang_set)]
                # batched_graphs = self.graphs  # Using all graphs for similarity computation

                # Encode texts and graphs
                text_embeds, _ = self.encoder(texts=batched_texts, graphs=[]) # No graph input here

                # Compute the similarity matrix
                similarities = self.compute_cosine_similarity(text_embeds, graph_embeds)

                # Aggregate similarities by taking the mean across the language set for each graph
                mean_similarities = similarities.mean(dim=0)  # Taking mean across texts for each graph

                # Sort indices by decreasing mean similarity
                sorted_indices = mean_similarities.argsort(descending=True)
                correct_rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
                
                # Calculate reciprocal rank
                reciprocal_rank = 1 / correct_rank
                total_reciprocal_rank += reciprocal_rank

                # Update accuracy results based on ranks
                for rank in n:
                    if correct_rank <= rank:
                        accuracy_results[rank] += 1

            # Average the results over the number of texts
            for k in accuracy_results:
                accuracy_results[k] /= num_text

            # Calculate the average reciprocal rank
            mrr = total_reciprocal_rank / num_text

        return accuracy_results, mrr
    
    def compute_text_to_monolang_graph_rank_n(self, n=[1, 3, 10], languages=None):
        if languages is None:
            languages = self.lang_set
        num_graph = len(self.graphs)
        num_text = len(self.texts) // len(self.lang_set)
        lang_results = {lang: {k: 0 for k in n} for lang in languages}
        lang_mrr = {lang: 0.0 for lang in languages}

        with torch.no_grad():
            # Encode all graphs at once since they are constant across languages
            _, graph_embeds = self.encoder(texts=[], graphs=self.graphs)  # No text input required here
            
            for lang in languages:
                if lang != 'en':
                    continue
                lang_index = self.lang_set.index(lang)
                for i in tqdm(range(num_text)):
                    # Selecting the text of the current language for each graph
                    current_text = [self.texts[i * len(self.lang_set) + lang_index]]

                    # Encode the current text (empty graphs since we already have their embeddings)
                    text_embed, _ = self.encoder(texts=current_text, graphs=[])  # No graph input required here

                    # Compute similarity between the current text and all graph embeddings
                    similarities = self.compute_cosine_similarity(text_embed, graph_embeds)

                    # Sort by decreasing similarity
                    sorted_indices = similarities.squeeze().argsort(descending=True)
                    correct_rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
                    lang_mrr[lang] += 1 / correct_rank

                    # Update rank accuracies for the current language
                    for rank in n:
                        if correct_rank <= rank:
                            lang_results[lang][rank] += 1
                            
                # Convert counts to percentages
                for rank in n:
                    lang_results[lang][rank] /= num_text

                # Calculate MRR for the current language    
                lang_mrr[lang] /= num_text

        return lang_results, lang_mrr
    
    
    def compute_graph_to_multilang_text_rank_n(self, batch_size=32, n=[1, 3, 10]):
        num_graph = len(self.graphs)
        num_text_groups = len(self.texts) // len(self.lang_set)
        assert num_graph == num_text_groups
        accuracy_results = {k: 0 for k in n}
        total_reciprocal_rank = 0.0

        with torch.no_grad():
            # Encode all multilingual text groups in batch
            text_embeds = []
            for i in range(0, len(self.texts), batch_size):
                end_index = min(i + batch_size, len(self.texts))
                batched_embeds, _ = self.encoder(texts=self.texts[i:end_index], graphs=[])
                text_embeds.extend(batched_embeds)

            assert len(text_embeds) == num_graph * len(self.lang_set)

            # Convert text_embeds to tensor
            text_embeds = torch.stack(text_embeds)

            for i in tqdm(range(num_graph)):
                current_graph = [self.graphs[i]]
                _, graph_embeds = self.encoder(texts=[], graphs=current_graph)  # Encode the current graph 

                # Compute similarity between the current graph and all multilingual text groups
                similarities = self.compute_cosine_similarity(text_embeds, graph_embeds)
                num_lang = len(self.lang_set)

                # Reshape similarities to (num_text_groups, num_lang) for easier averaging
                similarities = similarities.view(num_text_groups, num_lang)

                # Average the similarities for each text group
                averaged_similarities = similarities.mean(dim=1)

                # Sort by decreasing similarity
                sorted_indices = averaged_similarities.argsort(descending=True)
                correct_rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
                total_reciprocal_rank += 1 / correct_rank

                # Update accuracy results based on ranks
                for rank in n:
                    if correct_rank <= rank:
                        accuracy_results[rank] += 1

            # Convert counts to percentages and calculate the average correct rank
            for k in accuracy_results:
                accuracy_results[k] /= num_graph
            mrr = total_reciprocal_rank / num_graph

        return accuracy_results, mrr
    
    def compute_graph_to_monolang_text_rank_n(self, batch_size=32, n=[1, 3, 10]):
        num_graph = len(self.graphs)
        num_text_groups = len(self.texts) // len(self.lang_set)
        assert num_graph == num_text_groups
        lang_results = {lang: {k: 0 for k in n} for lang in self.lang_set}
        lang_mrr = {lang: 0.0 for lang in self.lang_set}

        with torch.no_grad():
            # Encode all multilingual text groups
            text_embeds = []
            for i in range(0, len(self.texts), batch_size):
                end_index = min(i + batch_size, len(self.texts))
                batched_embeds, _ = self.encoder(texts=self.texts[i:end_index], graphs=[])
                text_embeds.extend(batched_embeds)

            assert len(text_embeds) == num_graph * len(self.lang_set)

            # Convert text_embeds to tensor
            text_embeds = torch.stack(text_embeds)

            for lang_index, lang in enumerate(self.lang_set):
                if lang != 'en':
                    continue
                for i in tqdm(range(num_graph)):
                    current_graph = [self.graphs[i]]
                    _, graph_embeds = self.encoder(texts=[], graphs=current_graph)  # Encode the current graph

                    # Extract texts of the current language
                    lang_texts = self.texts[lang_index::len(self.lang_set)]
                    lang_text_embeds = text_embeds[lang_index::len(self.lang_set)]

                    # Compute similarity between the current graph and all texts of the current language
                    similarities = self.compute_cosine_similarity(lang_text_embeds, graph_embeds)

                    # Sort by decreasing similarity
                    sorted_indices = similarities.squeeze().argsort(descending=True)
                    correct_rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
                    lang_mrr[lang] += 1 / correct_rank

                    # Update accuracy results based on ranks
                    for rank in n:
                        if correct_rank <= rank:
                            lang_results[lang][rank] += 1

                # Convert counts to percentages and calculate the average correct rank
                for k in lang_results[lang]:
                    lang_results[lang][k] /= num_graph
                lang_mrr[lang] /= num_graph

        return lang_results, lang_mrr

        

