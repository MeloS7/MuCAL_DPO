import re
import json
import random

class DataLoader:
    def __init__(self, train_path, dev_path, eval_path, lang_set=['en', 'zh', 'fr', 'ar', 'es', 'ru']):
        super(DataLoader, self).__init__()
        self.lang_set = lang_set
        
        # 设置采样比例，保留1/100的数据
        self.fraction = 1

        # Read data
        self.train_raw_data = self.read_json_file(train_path)
        self.dev_raw_data = self.read_json_file(dev_path)
        self.eval_raw_data = self.read_json_file(eval_path)
        
        # Pre-process loaded data
        self.train_data = self.preprocess_data(self.train_raw_data, 1, fraction=self.fraction)
        self.dev_data = self.preprocess_data(self.dev_raw_data, 1, fraction=self.fraction)
        self.eval_data = self.preprocess_data(self.eval_raw_data, 1, fraction=self.fraction)
        
    def read_json_file(self, input_path):
        '''
        This function is used to read input json file,
        '''
        with open(input_path, 'r') as f:
            data = json.load(f)
        return data
    
    def from_camel(self, s):
        return re.sub('([A-Z])', r' \1', s).lower()

    def to_camel(self, s):
        s = re.sub(r"(_|-)+", " ", s).title().replace(" ", "")
        
        # Join the string, ensuring the first letter is lowercase
        return ''.join([s[0].lower(), s[1:]])
    
    def linearize_graph(self, graph):
        linearized_graph = ""
        for triple in graph:
            # Process triple
            sub = triple["subject"].replace("_", " ").strip('\"')
            pred = self.to_camel(self.from_camel(triple["property"]))
            obj = triple["object"].replace("_", " ").strip('\"')
            # Combine triples
            linearized_graph += " [S] " + sub + " [P] " + pred + " [O] " + obj
            
        return linearized_graph.strip()
        
    
    def preprocess_data(self, input_data, num_lexs=3, fraction=1.0):
        '''
        This function is used to continuously read lexicalisation in five languages.
        The output dictionary consists of two lists (texts and graphs) of string.
        
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
        "[S] subject1 [P] predicate1 [O] object1 [S] ... [O] objectN"
        
        In summary,
        Input:
            data: loaded json data (a list)
        Output:
            A dictionary includes:
                texts: a list of string (length: N x num_lang)
                graphs: a list of string (length: N)
        '''
        entries = input_data["entries"]
        
        # Sample a fraction of entries
        if fraction < 1.0:
            num_entries = max(1, int(len(entries) * fraction))
            entries = random.sample(entries, num_entries)
        
        texts = []
        graphs = []
        # We'd like to use all translations!!!
        # Some graphs have at most three lexs.
        for num_trans in range(num_lexs):
            for entry in entries:
                lexs = entry["lexicalisations"]
                if len(lexs["en"]) > num_trans:
                   # Extract graphs
                    graph = entry["modifiedtripleset"]
                    graphs.append(self.linearize_graph(graph))
                    # Extract texts
                    for lang in self.lang_set:
                        assert len(lexs[lang]) > num_trans
                        texts.append(lexs[lang][num_trans]["lex"]) # Take the num_trans -th lexs
                        
        return {
            "texts": texts,
            "graphs": graphs,
        }
    
    
    def get_train_data(self):
        return self.train_data
    
    def get_dev_data(self):
        return self.dev_data
    
    def get_eval_data(self):
        return self.eval_data
    
    def get_graphset_texts(self):
        return self.preprocess_data(self.eval_raw_data, num_lexs=1)
    
    # def get_graphset_texts(self):
    #     return self.preprocess_data(self.eval_raw_data, num_lexs=1, fraction=self.fraction)
    
    def __str__(self):
        lines = [
            f"The length of the training text data: {len(self.train_data['texts'])}.",
            f"The length of the training graph data: {len(self.train_data['graphs'])}.",
            f"The length of the dev text data: {len(self.dev_data['texts'])}.",
            f"The length of the dev graph data: {len(self.dev_data['graphs'])}.",
            f"The length of the test text data: {len(self.eval_data['texts'])}.",
            f"The length of the test graph data: {len(self.eval_data['graphs'])}.",
        ]
        return "\n".join(lines)
    
    def get_graphset_dev(self):
        """
        获取用于计算MRR的开发集数据
        """
        return self.preprocess_data(self.dev_raw_data, num_lexs=1)
        