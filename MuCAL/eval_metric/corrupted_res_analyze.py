import re
import os
import json
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_json_file(json_file):
    with open(json_file, 'r') as f:
        json_dict = json.load(f)
    return json_dict

def analyze_corrupted_res(data, lang='en'):
    # Analyze the score of corrupted results in each type
    # Initialize a dictionary to store the scores for each type
    res_dict = {}
    for type in ["good", "replace_pred", "replace_obj", "added", "removed", "swapped"]:
        res_dict[type] = []

    # Iterate through the data and add the scores to the corresponding type
    for item in data:
        type_name = item["type"]
        index = item["index"]
        score = item["score_"+lang]
        res_dict[type_name].append((index, score))

    return res_dict

def show_mean_score(res_dict):
    # Show the mean score of each type
    for type, score_tuples in res_dict.items():
        scores = [score for _, score in score_tuples]
        mean_score = sum(scores) / len(scores)
        print(f"{type}: {mean_score:.4f}")

def compute_top_n_good(res_dict, n=1000):
    # Create a list to store all scores along with their type
    all_scores = []
    
    # Iterate through the res_dict and combine all scores with their type
    for type_name, scores_tuples in res_dict.items():
        scores = [score for _, score in scores_tuples]
        index = [index for index, _ in scores_tuples]
        for idx, score in enumerate(scores):
            all_scores.append((score, type_name, index[idx]))  # Store (score, type, index)
    
    # Sort the combined list based on scores (descending order)
    sorted_scores = sorted(all_scores, key=lambda x: x[0], reverse=True)
    
    # Get the top n scores
    top_n_scores = sorted_scores[:n]
    
    # Count the occurrence of each type in the top n
    type_count = {}
    for _, type_name, _ in top_n_scores:
        if type_name not in type_count:
            type_count[type_name] = 0
        type_count[type_name] += 1
    
    # Calculate the proportion for each type
    type_proportion = {type_name: count / n for type_name, count in type_count.items()}
    
    return type_proportion

def plot_score_distribution(score_instances, file_name, encoder, epoch, bs, lang):
    '''
    Plot the proportional score distribution for all instances in one plot (stacked bar plot),
    sorted by 'good' type scores, with sorted instances aligned on the X axis.
    '''
    instance_ids = list(score_instances.keys())
    type_names = ['good', 'removed', 'added', 'replace_obj', 'replace_pred', 'swapped']

    # Initialize lists for stacked bars
    type_scores = {type_name: [] for type_name in type_names}

    # Collect proportional scores for each type across all instances
    for instance_id in instance_ids:
        # Get the scores for the current instance
        instance_scores = dict(score_instances[instance_id])
        total_score = sum(instance_scores.values())  # Total score for this instance

        # Calculate the proportional score for each type
        for type_name in type_names:
            score = instance_scores.get(type_name, 0)
            proportion = score / total_score if total_score != 0 else 0
            type_scores[type_name].append(proportion)

    # Sort instances by 'good' type scores
    # Extract sorted indices based on 'good' scores
    sorted_indices = sorted(range(len(instance_ids)), key=lambda idx: type_scores['good'][idx], reverse=True)

    # Sort each type's scores according to sorted_indices
    sorted_type_scores = {type_name: [type_scores[type_name][i] for i in sorted_indices] for type_name in type_names}

    # Create a plot
    plt.figure(figsize=(12, 8))

    # Bottom array for stacking
    bottom = [0] * len(sorted_indices)

    # Use a high-contrast colormap, such as Paired
    colors = plt.get_cmap('Paired').colors

    # Plot each type's proportional scores as stacked bars, using sorted instance ids as X axis values
    for i, type_name in enumerate(type_names):
        plt.bar(range(len(sorted_indices)), sorted_type_scores[type_name], bottom=bottom, label=type_name, color=colors[i], width=0.8)
        bottom = [i + j for i, j in zip(bottom, sorted_type_scores[type_name])]  # Update bottom for next stacked bar

    # Add titles and labels
    plt.title(f"Proportional Score Distribution Across Instances \n(Encoder {encoder}, Epoch {epoch}, Batch Size {bs}, Lang:{lang})", fontsize=12)
    plt.xlabel("Instances (sorted by 'good' score)", fontsize=12)
    plt.ylabel("Proportions", fontsize=12)

    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Display legend outside the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Save the plot
    plt.savefig(file_name, bbox_inches='tight')


def compare_type_in_instance(res_dict, save_file_name, encoder, epoch, bs, lang):
    '''
    Compare the results of different types in each instance
    We want to know which type is more likely to appear in each instance and the order of the types
    Index Range for each type:
        - good: 0~10000
        - removed: 20000~30000
        - added: 30000~40000
        - replace_obj: 40000~50000
        - replace_pred: 50000~60000
        - swapped: 60000~70000
    '''
    index_range = {"good": (0, 10000), "removed": (20000, 30000), "added": (30000, 40000), "replace_obj": (40000, 50000), "replace_pred": (50000, 60000), "swapped": (60000, 70000)}
    # Get all index
    index_good = [index for index, _ in res_dict["good"]]
    print(f"Number of instances: {len(index_good)}")

    # Compare the results of different types in each instance
    score_instances = {}
    for index in index_good:
        for type in ["good", "removed", "added", "replace_obj", "replace_pred", "swapped"]:
            # Get the score of the type
            index_type = index + index_range[type][0]
            
            if index_type in [index for index, _ in res_dict[type]]:
                # Find the score of the type
                for score_tuple in res_dict[type]:
                    if score_tuple[0] == index_type:
                        score = score_tuple[1]
                        break

                # Add the score to the score_instances
                if index not in score_instances:
                    score_instances[index] = []
                score_instances[index].append((type, score))

        # Sort the score_instances by score
        score_instances[index].sort(key=lambda x: x[1], reverse=True)

    # Plot the score distribution for each instance
    plot_score_distribution(score_instances, save_file_name, encoder, epoch, bs, lang)
    
    
    

def main():
    # Argument parser
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--file_path", help="Path to processed json file", default="data/test/corrupt/crossencoder_score/ep10_bs4/corrupt_score_cr_ep10_bs4_en.json")
    argParser.add_argument("-l", "--lang", help="Language", default="en")
    argParser.add_argument("-s", "--save_path", help="Path to save the figure", default="fig/score_dist/")
    argParser.add_argument("-e", "--encoder", help="Encoder", default="CrossEncoder")
    argParser.add_argument("-ep", "--epoch", help="Epoch", default=10)
    argParser.add_argument("-bs", "--batch_size", help="Batch size", default=4)

    args = argParser.parse_args()
    file_path = args.file_path
    lang = args.lang
    save_path = args.save_path
    encoder = args.encoder
    epoch = args.epoch
    bs = args.batch_size
    save_file_name = save_path + f"score_distribution_{encoder}_ep{epoch}_bs{bs}_lang_{lang}.png"

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return
    
    # Load data
    data = load_json_file(file_path)

    # Analyze the score of corrupted results in each type
    res_dict = analyze_corrupted_res(data, lang)

    # Show the mean score of each type
    show_mean_score(res_dict)

    # Compute how many good samples are in top #Good
    num_good = 1000
    type_proportion = compute_top_n_good(res_dict, num_good)

    # Print the results
    for type_name, proportion in type_proportion.items():
        print(f"{type_name}: {proportion*100:.2f}%")

    # Compare the results of different types in each instance
    # compare_type_in_instance(res_dict, save_file_name, encoder, epoch, bs, lang)



if __name__ == "__main__":
    main()