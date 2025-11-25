import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import spacy
from nltk.tree import ParentedTree, Tree
from collections import defaultdict
from math import sqrt, exp
from typing import Set, List, Dict, Tuple
from tqdm import tqdm


class SPICalculator:
    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Downloading spaCy model...")
            from spacy.cli import download
            download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')

    def get_constituents(self, tree: ParentedTree) -> Set[str]:
        constituents = set()
        for subtree in tree.subtrees():
            constituent_str = ' '.join(str(subtree).split())
            constituents.add(constituent_str)
        return constituents

    def get_production_rules(self, tree: ParentedTree) -> List[str]:
        productions = []
        for subtree in tree.subtrees():
            if len(subtree) > 0:
                rule = f"{subtree.label()} -> {' '.join(n.label() if isinstance(n, Tree) else n for n in subtree)}"
                productions.append(rule)
        return productions

    def parse_sentence(self, sentence: str) -> ParentedTree:
        def build_tree(token):
            node_label = f"{token.pos_}_{token.dep_}"
            children = list(token.children)
            
            if not children:
                return ParentedTree(node_label, [token.text])
            
            child_trees = [build_tree(child) for child in children]
            
            if token.pos_ not in ['DET', 'ADP', 'PART', 'CCONJ', 'SCONJ']:
                child_trees.append(ParentedTree(f"WORD_{token.pos_}", [token.text]))
            
            return ParentedTree(node_label, child_trees)

        doc = self.nlp(sentence)
        root_token = next(token for token in doc if token.dep_ == "ROOT")
        tree = build_tree(root_token)
        final_tree = ParentedTree('S', [tree])
        return final_tree

    def calculate_tree_kernel(self, tree1: ParentedTree, tree2: ParentedTree) -> float:
        def normalize_tree(tree: ParentedTree) -> Dict:
            normalized = defaultdict(int)
            constituents = self.get_constituents(tree)
            productions = self.get_production_rules(tree)
            
            for constituent in constituents:
                normalized[constituent] += 1
            for production in productions:
                normalized[production] += 1
                
            return normalized

        norm1 = normalize_tree(tree1)
        norm2 = normalize_tree(tree2)
        
        common_features = set(norm1.keys()) & set(norm2.keys())
        kernel_value = sum(norm1[f] * norm2[f] for f in common_features)
        
        size1 = sum(v * v for v in norm1.values())
        size2 = sum(v * v for v in norm2.values())
        
        if size1 == 0 or size2 == 0:
            return 0.0
            
        return kernel_value / sqrt(size1 * size2)

    def calculate_priming_effect(self, pp_sentence: str, np_sentence: str, ps_sentence: str) -> Tuple[float, Dict]:
        try:
            pp_tree = self.parse_sentence(pp_sentence)
            np_tree = self.parse_sentence(np_sentence)
            ps_tree = self.parse_sentence(ps_sentence)
        except Exception as e:
            raise ValueError(f"Error parsing sentences: {str(e)}")
        
        Dp = self.calculate_tree_kernel(pp_tree, ps_tree)
        Dn = self.calculate_tree_kernel(np_tree, ps_tree)
        
        diff = Dp - Dn
        exp_term = exp(self.gamma * diff)
        spi = (exp_term - 1) / (exp_term + 1)
        
        details = {
            'Dp': Dp,
            'Dn': Dn,
            'diff': diff,
            'spi': spi
        }
        
        return spi, details


def process_caption_file(caption_csv: str, output_csv: str, gamma: float = 3.0):
    print(f"Reading {caption_csv}...")
    caption_df = pd.read_csv(caption_csv)
    
    calculator = SPICalculator(gamma=gamma)
    spi_scores = []
    
    print("Calculating SPI scores...")
    for idx, row in tqdm(caption_df.iterrows(), total=len(caption_df)):
        try:
            spi_score, _ = calculator.calculate_priming_effect(
                str(row['current_positive_prime']),
                str(row['current_negative_prime']),
                str(row['generated_caption'])
            )
            spi_scores.append(spi_score)
        except Exception as e:
            print(f"Error at row {idx}: {e}")
            spi_scores.append(None)
    
    caption_df['SPI_score'] = spi_scores
    caption_df.to_csv(output_csv, index=False)
    
    valid_scores = [s for s in spi_scores if s is not None]
    print(f"Mean SPI: {np.mean(valid_scores):.4f}")
    return caption_df


def analyze_spi_scores(file_path):
    df = pd.read_csv(file_path)
    
    group_names = [
        'Simple Active',                      # label 0
        'Simple Passive',                     # label 1
        'Prepositional Active',               # label 2
        'Prepositional Passive',              # label 3
        'Embedded Active',                    # label 4
        'Embedded Passive',                   # label 5
        'Mediopassive Like Active',           # label 6
        'Mediopassive',                       # label 7
        'Simple Double Object',               # label 8
        'Simple Prepositional Object',        # label 9
        'Complex Double Object',              # label 10
        'Complex Prepostional Object',        # label 11
        'Double Object with Clause',          # label 12
        'Prepositional Object with Clause',   # label 13
        'S-Genitive',                         # label 14
        'Of-Genitive'                         # label 15
    ]
    
    results = []
    for label in range(16):
        group_data = df[df['label'] == label]
        
        spi_scores = group_data['SPI_score'].dropna()
        
        if len(spi_scores) == 0:
            continue
        
        mean_score = spi_scores.mean()
        positive_ratio = (spi_scores > 0).mean()
        
        results.append({
            'Group': group_names[label],
            'Mean_SPI': round(mean_score, 4),
            'Positive_Ratio': f"{round(positive_ratio * 100, 2)}%",
            'Sample_Size': len(spi_scores)
        })
    
    return pd.DataFrame(results)

def plot_main_results(results_df, save_path=None):
    label_names = results_df['Group'].tolist()
    values = results_df['Mean_SPI'].tolist()
    
    plt.figure(figsize=(10, 4))
    
    y_pos = np.arange(len(label_names))
    
    colors = []
    color1 = '#2b506e'
    color2 = '#8cabc5'
    for i in range(len(label_names)):
        if i % 2 == 0:
            colors.append(color1)
        else:
            colors.append(color2)
    
    plt.barh(y_pos, values, height=0.6, color=colors)
    
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.yticks(y_pos, label_names)
    plt.xlabel('Mean SPI Value')
    
    for i in range(1, len(label_names), 2):
        plt.axhline(y=y_pos[i] + 0.5, color='gray', linestyle='-', alpha=0.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Main plot saved to {save_path}")
    
    plt.show()

#plot
def plot_comparison(with_prime_df, without_prime_df, save_path=None):
    plt.rcParams.update({'font.size': 12})
    
    group_mapping = {
        'Simple Active': 'active_passive',
        'Simple Passive': 'active_passive',
        'Prepositional Active': 'active_passive',
        'Prepositional Passive': 'active_passive',
        'Embedded Active': 'active_passive',
        'Embedded Passive': 'active_passive',
        'Mediopassive Like Active': 'active_passive',
        'Mediopassive': 'active_passive',
        'Simple Double Object': 'po_do',
        'Simple Prepositional Object': 'po_do',
        'Complex Double Object': 'po_do',
        'Complex Prepostional Object': 'po_do',
        'Double Object with Clause': 'po_do',
        'Prepositional Object with Clause': 'po_do',
        'S-Genitive': 'genitive',
        'Of-Genitive': 'genitive'
    }
    
    label_mapping = {
        'Simple Double Object': 'Simple DO',
        'Simple Prepositional Object': 'Simple PO',
        'Complex Double Object': 'Complex DO',
        'Complex Prepostional Object': 'Complex PO',
        'Double Object with Clause': 'DO with Clause',
        'Prepositional Object with Clause': 'PO with Clause'
    }
    
    data_pairs = []
    for idx, row in with_prime_df.iterrows():
        label = row['Group']
        display_label = label_mapping.get(label, label)
        with_prime = row['Mean_SPI']
        without_prime = without_prime_df.iloc[idx]['Mean_SPI']
        group_type = group_mapping[label]
        data_pairs.append((display_label, with_prime, without_prime, group_type))
    
    color_schemes = {
        'active_passive': ('#2b506e', '#8cabc5'),
        'po_do': ('#2e8540', '#92c69a'),
        'genitive': ('#e57373', '#ffb6c1')
    }
    
    current_y_pos = 0
    y_positions = []
    labels = []
    values_with_prime = []
    values_without_prime = []
    colors_with_prime = []
    colors_without_prime = []
    group_boundaries = []
    
    for group_type in ['active_passive', 'po_do', 'genitive']:
        group_data = [item for item in data_pairs if item[3] == group_type]
        for item in group_data:
            labels.append(item[0])
            values_with_prime.append(item[1])
            values_without_prime.append(item[2])
            colors_with_prime.append(color_schemes[group_type][0])
            colors_without_prime.append(color_schemes[group_type][1])
            y_positions.append(current_y_pos)
            current_y_pos += 1
        group_boundaries.append(current_y_pos - 0.5)
    
    plt.figure(figsize=(12, 5))
    
    bar_height = 0.3
    y_with_prime = [y + bar_height/2 for y in y_positions]
    y_without_prime = [y - bar_height/2 for y in y_positions]
    
    group_labels = {
        'active_passive': 'Active-Passive Types',
        'po_do': 'PO-DO Types',
        'genitive': 'Genitive Types'
    }
    
    for group_type in ['active_passive', 'po_do', 'genitive']:
        mask = [color == color_schemes[group_type][0] for color in colors_with_prime]
        if any(mask):
            indices = [j for j, m in enumerate(mask) if m]
            
            plt.barh([y_with_prime[j] for j in indices],
                    [values_with_prime[j] for j in indices],
                    height=bar_height, color=color_schemes[group_type][0],
                    label=f'{group_labels[group_type]} With Prime')
            
            plt.barh([y_without_prime[j] for j in indices],
                    [values_without_prime[j] for j in indices],
                    height=bar_height, color=color_schemes[group_type][1],
                    label=f'{group_labels[group_type]} Without Prime')
    
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.yticks(y_positions, labels, fontsize=12)
    plt.xticks(fontsize=14)
    plt.xlabel('Mean SPI Value', fontsize=18, labelpad=10)
    plt.title('Blip-2: With vs Without Prime Sentence', fontsize=20, pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=11)
    
    for boundary in group_boundaries[:-1]:
        plt.axhline(y=boundary, color='gray', linestyle='-', alpha=0.2, linewidth=2)
    
    plt.gca().invert_yaxis()
    
    # 自动计算合适的x轴范围
    all_values = values_with_prime + values_without_prime
    x_min = min(all_values)
    x_max = max(all_values)
    x_range = x_max - x_min
    x_margin = x_range * 0.15  # 留出15%的边距
    
    plt.xlim(x_min - x_margin, x_max + x_margin)
    
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.3f}'.format(x)))
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    caption_with_prime = 'caption_generation_results_blip.csv'
    caption_without_prime = 'caption_blip_noprime.csv'
    

    process_caption_file(caption_with_prime, 'SPI_with_prime.csv')
    analysis_with = analyze_spi_scores('SPI_with_prime.csv')
    print("\nWith Prime Results:")
    print(analysis_with.to_string(index=False))
    
    plot_main_results(analysis_with, 'blip_main_plot.png')

    process_caption_file(caption_without_prime, 'SPI_without_prime.csv')
    analysis_without = analyze_spi_scores('SPI_without_prime.csv')
    print("\nWithout Prime Results:")
    print(analysis_without.to_string(index=False))
    
    plot_comparison(analysis_with, analysis_without, 'blip_comparison.png')
    
    print("\n" + "="*60)
    print("Complete!")
    print("="*60)