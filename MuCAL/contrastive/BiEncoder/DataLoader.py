import re
import os
import json
import copy
import random
from tqdm import tqdm

class DataLoader:
    def __init__(self, train_path, dev_path, eval_path, lang_set=['en', 'zh', 'fr', 'ar', 'es', 'ru']):
        super(DataLoader, self).__init__()
        self.lang_set = lang_set

        # Read data
        self.train_raw_data = self.read_json_file(train_path)
        self.dev_raw_data = self.read_json_file(dev_path)
        self.eval_raw_data = self.read_json_file(eval_path)
        
        # Pre-process loaded data
        self.train_data = self.preprocess_data(self.train_raw_data, 3)
        self.dev_data = self.preprocess_data(self.dev_raw_data, 3)
        self.eval_data = self.preprocess_data(self.eval_raw_data, 1)

        # Set up symmetrical predicates
        self.symmetrical_predicates = [
            "taxon synonym", "partner in business or sport", "opposite of",
            "partially coincident with", "physically interacts with", "partner",
            "relative", "related category", "connects with", "twinned administrative body",
            "different from", "said to be the same as", "sibling", "adjacent station",
            "shares border with"
        ]
        # Do to_camel case for symmetrical predicates
        self.symmetrical_predicates = set([self.to_camel(predicate) for predicate in self.symmetrical_predicates])  
        
        # Store corrupted graphs
        self.swapped_graphs = None
        self.property_similarity_dict = None
        self.hard_negatives = None
        
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

    def parse_linearized_graph(self, linearized_graph):
        """
        Convert linearized graph to triples
        """
        triples = []
        triple_parts = linearized_graph.split("[S]")[1:]  # Split each triple
        for triple in triple_parts:
            parts = triple.strip().split("[P]")
            subject = parts[0].strip()
            predicate, obj = parts[1].strip().split("[O]")
            triples.append({
                "subject": subject,
                "property": predicate.strip(),
                "object": obj.strip()
            })
        return triples
        
    
    def preprocess_data(self, input_data, num_lexs=3):
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

    def generate_swap_hard_negatives_dict(self, graphs, save_path='data/others/swap_hard_negatives.json'):
        """
        Generate a dictionary where each graph has one corrupted graph.
        
        Args:
            graphs (list[str]): Linearized graphs in the current batch.
        
        Returns:
            dict: A dictionary mapping graph indices to lists of corrupted graphs (or None for uncorrupted cases).
                Example: {0: ["corrupted_graph_0"], 1: [None], ...}
        """
        # Check if the dictionary file exists
        if os.path.exists(save_path):
            print(f"Loading existing swapped hard negatives dictionary from {save_path}.")
            with open(save_path, 'r', encoding='utf-8') as f:
                swapped_graphs = json.load(f)
            self.swapped_graphs = swapped_graphs
            print(f"Swapped hard negatives dictionary loaded from {save_path}.")
            return swapped_graphs

        print(f"No existing dictionary found. Creating a new one and saving to {save_path}.")

        # Generate the swapped hard negatives dictionary
        corrupted_graphs = []

        for graph_idx, graph in enumerate(graphs):
            # Parse the linearized graph into triples
            triples = self.parse_linearized_graph(graph)

            # Create a dictionary to store corrupted triples with length of triples
            corrupted_triples = {}

            # Create corrupted triples for all triples
            for triple_idx, triple in enumerate(triples):
                # Skip if the predicate is symmetric
                if triple['property'] in self.symmetrical_predicates:
                    corrupted_triples[triple_idx] = None
                    continue

                # Swap subject and object
                swapped_triple = {
                    "subject": triple["object"],
                    "property": triple["property"],
                    "object": triple["subject"]
                }

                # Replace the original triple with the swapped triple
                triples_copy = triples.copy()
                triples_copy[triple_idx] = swapped_triple

                # Add the corrupted triple to the dictionary
                corrupted_triples[triple_idx] = self.linearize_graph(triples_copy)

            # Add the dictionary to the list
            corrupted_graphs.append(corrupted_triples)

        # for testing if all graphs have at least one corrupted graph
        num_no_corrupted = 0
        for graph_idx, corrupted_triples in enumerate(corrupted_graphs):
            if len(corrupted_triples) == 1 and corrupted_triples[0] is None:
                num_no_corrupted += 1

        print(f"Number of graphs with no corrupted graphs: {num_no_corrupted}")

        # Save the corrupted graphs dictionary to a file
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(corrupted_graphs, f, ensure_ascii=False, indent=4)

        print(f"Swapped hard negatives dictionary saved to {save_path}.")

        self.swapped_graphs = corrupted_graphs
        return True

    def create_property_space(self, graphs):
        """
        Create a property space for each graph
        """
        property_space = {}
        for graph in graphs:
            triples = self.parse_linearized_graph(graph)
            for triple in triples:
                property_space[triple["property"]] = True

        return list(property_space.keys())

    def select_top_k_similar_properties(self, property_space, k=10, threshold=0.4, save_path='data/others/top_10_similar_property.json'):
        """
        Select 'top k' most similar properties to each property in the property space.
        ===== Important =====
        To avoid the property in top k is actually the same meaning as the property itself,
        we FILTER OUT the property whose similarity score is larger than 'threshold' with the property itself.

        Args:
        property_space (list): List of properties as strings.
        k (int): Number of most similar properties to select for each property. Default is 10.
        threshold (float): Threshold for similarity score. Default is 0.4.

        Returns:
        dict: A dictionary where the key is a property, and the value is a list of k most similar properties.
        """
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        import torch

        # Check if the dictionary file exists
        if os.path.exists(save_path):
            print(f"Loading existing property similarity dictionary from {save_path}.")
            with open(save_path, 'r', encoding='utf-8') as f:
                property_similarity_dict = json.load(f)
            self.property_similarity_dict = property_similarity_dict
            print(f"Property similarity dictionary loaded from {save_path}.")
            return True

        print(f"No existing dictionary found. Creating a new one and saving to {save_path}.")

        try:
            # Load model and create embeddings
            model = SentenceTransformer('all-mpnet-base-v2', device="cuda" if torch.cuda.is_available() else "cpu")
            property_embeddings = model.encode(property_space, convert_to_tensor=True)

            # Compute cosine similarity matrix
            similarity_matrix = cosine_similarity(property_embeddings.cpu().numpy())

            # Create a dictionary to store top k similar properties
            property_similarity_dict = {}

            for idx, property_name in enumerate(property_space):
                # Get similarity scores for the current property
                similarity_scores = list(enumerate(similarity_matrix[idx]))
                # Exclude the property itself and sort by similarity
                similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
                # Filter out the property whose similarity score is larger than 'threshold' with the property itself
                filtered_similarity_scores = [i for i in similarity_scores if i[0] != idx and i[1] < threshold]
                top_k_indices = [i[0] for i in filtered_similarity_scores][:k]
                top_k_properties = [property_space[i] for i in top_k_indices]
                # Add to the dictionary
                property_similarity_dict[property_name] = top_k_properties

            assert len(property_similarity_dict) == len(property_space)

            # Save the dictionary to the specified path
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(property_similarity_dict, f, ensure_ascii=False, indent=4)

            print(f"Property similarity dictionary saved to {save_path}.")

            self.property_similarity_dict = property_similarity_dict

        finally:
            # Move the model and embeddings off the GPU
            model.to('cpu')
            del model
            del property_embeddings
            torch.cuda.empty_cache()

        return True

    def generate_hard_negatives(self, num_hard_negatives=6):
        """
        Generate hard negatives for all graphs
        We first generate the property space for each graph, then select the top k similar properties.
        Then we create the hard negatives by swapping the subject and object for each triple in the graph.
        Finally, we combine the swapped graphs with the property-changed graphs.

        ===== Important =====
        To fit the training process, we need to make sure that the number of hard negatives is the same in each batch.
        It means that for each graph, if there is no enough swapped graphs, we need to select the property-changed graphs.

        Args:
            num_hard_negatives (int): Number of hard negatives to generate for each graph. Default is 1.

        Returns:
            list[list[str]]: A list of lists of hard negatives for all graphs.
        """        
        # Generate property space for each graph
        property_space = self.create_property_space(self.train_data["graphs"])

        # Select top k similar properties for each property
        if not self.select_top_k_similar_properties(property_space, k=10):
            print("Failed to create property similarity dictionary.")
            exit()

        # Generate swapped hard negatives
        if not self.generate_swap_hard_negatives_dict(self.train_data["graphs"]):
            print("Failed to generate hard negatives.")
            exit()

        def no_swapped_hard_negatives_check(graph_idx):
            # Iterate through the graph to check if there is no swapped hard negatives
            for triple in self.swapped_graphs[graph_idx]:
                if triple is not None:
                    return False
            return True

        # Initialize the list to store hard negatives
        hard_negatives = [[] for _ in range(len(self.train_data["graphs"]))]
        # Select hard negatives for each graph
        for graph_idx, graph in tqdm(enumerate(self.train_data["graphs"]), desc="Generating hard negatives"):
            # Prioritize the swapped hard negatives if there is any
            if not no_swapped_hard_negatives_check(graph_idx):
                # No enough swapped hard negatives
                # We will all not-None swapped hard negatives
                if len(self.swapped_graphs[graph_idx]) < num_hard_negatives:
                    for swap_idx, swapped_graph in self.swapped_graphs[graph_idx].items():
                        if swapped_graph is not None:
                            hard_negatives[graph_idx].append(swapped_graph)
                else:
                    # There is enough swapped hard negatives
                    # We randomly select one until the number of hard negatives reaches num_hard_negatives
                    patience = 5
                    while len(hard_negatives[graph_idx]) < num_hard_negatives and patience > 0:
                        random_triple_idx = random.randint(0, len(self.swapped_graphs[graph_idx]) - 1)
                        selected_graph = self.swapped_graphs[graph_idx][str(random_triple_idx)]
                        # the random triple should not be None and not repeat
                        if selected_graph is not None and \
                            selected_graph not in hard_negatives[graph_idx]:
                            hard_negatives[graph_idx].append(selected_graph)
                        else:
                            patience -= 1

                    assert len(hard_negatives[graph_idx]) <= num_hard_negatives

                    if len(hard_negatives[graph_idx]) == num_hard_negatives:
                        continue # to the next graph
            
            # If there is no enough swapped hard negatives, we select the property-changed graphs
            # print(f"Length of hard negatives for graph {graph_idx}: {len(hard_negatives[graph_idx])}")
            # print(f"Number of hard negatives needed: {num_hard_negatives}")
            assert len(hard_negatives[graph_idx]) < num_hard_negatives

            # Unlinearize the graph 
            current_graph = self.parse_linearized_graph(graph)

            # Make property changes to the triples
            num_property_changed = 0
            while len(hard_negatives[graph_idx]) < num_hard_negatives:
                # Deep copy the graph to avoid modifying the original graph
                current_graph_copy = copy.deepcopy(current_graph)

                assert num_property_changed < num_hard_negatives

                # Change the property of the triples
                random_property_idx = random.randint(0, len(current_graph_copy) - 1)
                current_property = current_graph_copy[random_property_idx]["property"]

                # Select a random property from the property similarity dictionary
                selected_property = self.property_similarity_dict[current_property][num_property_changed]
                current_graph_copy[random_property_idx]["property"] = selected_property

                # Linearize the graph
                hard_negatives[graph_idx].append(self.linearize_graph(current_graph_copy))

                num_property_changed += 1

            assert len(hard_negatives[graph_idx]) == num_hard_negatives

        # Save the hard negatives
        self.hard_negatives = hard_negatives

        print("Hard negatives generation done.")
        return True


    def select_hard_negatives_in_batch(self, batched_indices):
        """
        Select hard negatives for each graph in the current batch
        """
        return [self.hard_negatives[idx] for idx in batched_indices]
    
    def get_train_data(self):
        return self.train_data
    
    def get_dev_data(self):
        return self.dev_data
    
    def get_eval_data(self):
        return self.eval_data
    
    def get_graphset_texts(self):
        return self.preprocess_data(self.eval_raw_data, num_lexs=1)

    def get_graphset_dev(self):
        return self.preprocess_data(self.dev_raw_data, num_lexs=1)

    def shuffle_train_data(self):
        """Shuffle the training data while preserving alignment between texts and graphs."""
        assert len(self.train_data["texts"]) == len(self.train_data["graphs"]) * len(self.lang_set), \
            "Mismatch between texts and graphs length"
        
        # Group texts by graphs
        num_graphs = len(self.train_data["graphs"])
        grouped_texts = [
            self.train_data["texts"][i * len(self.lang_set):(i + 1) * len(self.lang_set)]
            for i in range(num_graphs)
        ]
        
        # Combine grouped texts and graphs
        combined = list(zip(grouped_texts, self.train_data["graphs"]))
        
        # Shuffle combined list
        random.shuffle(combined)
        
        # Unpack shuffled data
        shuffled_grouped_texts, shuffled_graphs = zip(*combined)
        self.train_data["graphs"] = list(shuffled_graphs)
        self.train_data["texts"] = [text for group in shuffled_grouped_texts for text in group]


    
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
    
        
    
    
            
            
            
                          
            
            
        
        
    
    