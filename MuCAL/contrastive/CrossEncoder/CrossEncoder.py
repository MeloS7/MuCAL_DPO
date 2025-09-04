import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from sentence_transformers import SentenceTransformer, models

class CrossEncoder(nn.Module):
    def __init__(self, encoder_model=None):
        super(CrossEncoder, self).__init__()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if encoder_model is not None:
            # Initialize Model
            word_embedding_model = models.Transformer(encoder_model)
            
            # Add pooling layer
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
            
            # Add one linear layer
            dense_model = models.Dense(
                in_features=pooling_model.get_sentence_embedding_dimension(),
                out_features=1,
                activation_function=nn.Sigmoid()
            )
            
            self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
            self.model.to(self.device)
        else:
            # Initial as None, load model in the function load()
            self.model = None 
        
    # We create in-batch negative samples in this function!!!!
    def concatenate_text_graph_in_batch(self, texts, graphs):
        '''
        This function is used to concatenate the text and the linearized graph to fit cross-encoder training.
        We create in-batch negative samples in this function!!!!
        We concatenate the strings in the form of "TEXT [SEP] GRAPH" for each language"
        
        For example:
        texts = [
            string1_en, string1_zh, string1_fr, string1_ar, string1_es, string1_ru,
            string2_en, string2_zh, string2_fr, string2_ar, string2_es, string2_ru,
            ...
            stringN_en, stringN_zh, stringN_fr, stringN_ar, stringN_es, stringN_ru,
        ]
        where string{n}_en is the source graph verbalisation which is in English,
        string{n}_{lang} is the translation in different language corresponding to string{n}_en.
        
        
        graphs = [
            graph1, graph2, ..., graphN
        ]
        where graph{n} is a linearized graph in the form of 
        "[S] subject1 [P] predicate1 [O] object1 [S] ... [O] objectM"
        
        Concatenated_string = [
            string1_en [SEP] graph1, string1_en [SEP] graph2, ..., string1_en [SEP] graphN,
            string1_zh [SEP] graph1, string1_zh [SEP] graph2, ..., string1_zh [SEP] graphN,
            ...
            string1_ru [SEP] graph1, string1_ru [SEP] graph2, ..., string1_ru [SEP] graphN,
            string2_en [SEP] graph1, string2_en [SEP] graph2, ..., string2_en [SEP] graphN,
            string2_zh [SEP] graph1, string2_zh [SEP] graph2, ..., string2_zh [SEP] graphN,
            ...
            string2_ru [SEP] graph1, string2_ru [SEP] graph2, ..., string2_ru [SEP] graphN,
            ...
            stringN_ru [SEP] graph1, stringN_ru [SEP] graph2, ..., stringN_ru [SEP] graphN,
        ]
        
        
        In summary,
        Input:
            data: a dictionary includes the texts and the graphs
        Output:
            cross_text: a list of string (length: N x N x num_lang)
        '''
        
        # Ensure that the length of texts is divisible by the length of graphs
        assert len(texts) % len(graphs) == 0, f"Error: The length of graphs ({len(graphs)}) is not divisible by the length of texts ({len(texts)})."

        cross_text = []  # This will store the concatenated strings

        # Iterate through each text and concatenate it with each graph
        for i in range(len(texts)):
            text = texts[i]
            for graph in graphs:
                # Concatenate each text with the corresponding graph
                concatenated_string = f"{text} [SEP] {graph}"
                cross_text.append(concatenated_string)

        return cross_text
    
    def concatenate_text_graph_positive(self, texts, graphs):
        '''
        This function is used to concatenate the text and the linearized graph to fit cross-encoder training.
        There are all positive samples!!!
        We concatenate the strings in the form of "TEXT [SEP] GRAPH" for each language"
        
        For example:
        texts = [
            string1_en, string1_zh, string1_fr, string1_ar, string1_es, string1_ru,
            string2_en, string2_zh, string2_fr, string2_ar, string2_es, string2_ru,
            ...
            stringN_en, stringN_zh, stringN_fr, stringN_ar, stringN_es, stringN_ru,
        ]
        where string{n}_en is the source graph verbalisation which is in English,
        string{n}_{lang} is the translation in different language corresponding to string{n}_en.
        
        
        graphs = [
            graph1, graph2, ..., graphN
        ]
        where graph{n} is a linearized graph in the form of 
        "[S] subject1 [P] predicate1 [O] object1 [S] ... [O] objectM"
        
        Concatenated_string = [
            string1_en [SEP] graph1, string1_zh [SEP] graph1, ..., string1_ru [SEP] graph1,
            string2_en [SEP] graph2, string2_zh [SEP] graph2, ..., string2_ru [SEP] graph2,
            ...
            stringN_en [SEP] graphN, stringN_zh [SEP] graphN, ..., stringN_ru [SEP] graphN,
        ]
        
        
        In summary,
        Input:
            data: a dictionary includes the texts and the graphs
        Output:
            cross_text: a list of string (length: N x num_lang)
        '''
        
        # Ensure that the length of texts is divisible by the length of graphs
        assert len(texts) % len(graphs) == 0, f"Error: The length of graphs ({len(graphs)}) is not divisible by the length of texts ({len(texts)})."

        num_lang = len(texts) // len(graphs)  # Calculate the number of languages
        cross_text = []  # This will store the concatenated strings

        # Iterate through each graph and concatenate it with its corresponding multilingual texts
        for i in range(len(graphs)):
            graph = graphs[i]
            for j in range(num_lang):
                # Concatenate each text with the corresponding graph
                text = texts[i * num_lang + j]
                concatenated_string = f"{text} [SEP] {graph}"
                cross_text.append(concatenated_string)

        return cross_text

    
    def forward(self, texts, graphs, train=True):
        """
        texts: list of texts in different languages of length (batch_size x num_langs)
        graphs: list of graphs of length batch_size
        The goal is to get outputs of shape (batch_size * num_langs,)
        """
        # Encode texts [SEP] linearized_graph
        if texts != []:
            # Concatenate the text and the linearized graph
            if train:
                cross_data = self.concatenate_text_graph_in_batch(texts, graphs) # (length: N x N x num_lang), where N is the graph number and also the batch size
            else:
                self.model.eval()
                # Concatenate the text and the linearized graph
                cross_data = self.concatenate_text_graph_pairs(texts, graphs)
            
            encoded_texts = self.model.tokenize(cross_data)
            encoded_texts = {k: v.to(self.device) for k, v in encoded_texts.items()}
            # text_embeds = self.model(encoded_texts)["sentence_embedding"]
            
            # Apply the linear layer to get scores for each t[SEP]g combination
            # logits = self.linear(text_embeds).squeeze(-1)  # Shape: (N * N * num_lang,) or (N * num_lang,)
            model_output = self.model(encoded_texts)
            # 获取 Dense 层的输出，即 logits
            logits = model_output['sentence_embedding'].squeeze(-1)
            return logits
        else:
            return None
        
    def concatenate_text_graph_pairs(self, texts, graphs):
        """
        This function concatenates each text with its corresponding graph.
        It assumes that texts and graphs are lists of the same length.
        Output:
            cross_text: a list of strings (length: N)
        """
        assert len(texts) == len(graphs), "The length of texts and graphs must be equal."
        cross_text = [f"{text} [SEP] {graph}" for text, graph in zip(texts, graphs)]
        return cross_text

    def predict(self, texts, graphs):
        self.model.eval()
        with torch.no_grad():
            logits = self.forward(texts, graphs, train=False)      
        return logits
    
    def save_pretrained(self, path):
        self.model.save_pretrained(path)

    def upload_model(self, repo_id):
        self.model.push_to_hub(repo_id=repo_id)
        print(f"Model uploaded successfully!")
        
    @staticmethod
    def load(path) -> 'CrossEncoder':
        instance = CrossEncoder(encoder_model=None)
        instance.model = SentenceTransformer(path)
        instance.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        instance.model.to(instance.device)
        return instance
    
        
    def __str__(self):
        return f"CrossEncoder using device: {self.device}"
