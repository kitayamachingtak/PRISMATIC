import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
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

    def calculate_spi(self, prime_sentence: str, target_sentence: str) -> float:
        prime_tree = self.parse_sentence(prime_sentence)
        target_tree = self.parse_sentence(target_sentence)
        
        kernel_similarity = self.calculate_tree_kernel(prime_tree, target_tree)
        
        exp_term = exp(self.gamma * kernel_similarity)
        spi = (exp_term - 1) / (exp_term + 1)
        
        return spi
    
    def calculate_priming_effect(self, positive_prime: str, negative_prime: str, target: str) -> Tuple[float, Dict]:
        pp_tree = self.parse_sentence(positive_prime)
        np_tree = self.parse_sentence(negative_prime)
        ps_tree = self.parse_sentence(target)
        
        Dp = self.calculate_tree_kernel(pp_tree, ps_tree)
        Dn = self.calculate_tree_kernel(np_tree, ps_tree)
        
        diff = Dp - Dn
        exp_term = exp(self.gamma * diff)
        pe = (exp_term - 1) / (exp_term + 1)
        
        details = {
            'Dp': Dp,
            'Dn': Dn,
            'diff': diff,
            'pe': pe
        }
        
        return pe, details


def clean_caption(text):
    match = re.search(r'Assistant:\s*(.*?)$', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def process_csv(input_file, output_file):
    df = pd.read_csv(input_file)
    df['generated_caption'] = df['generated_caption'].apply(lambda x: clean_caption(str(x)))
    df.to_csv(output_file, index=False)
    print(df['generated_caption'].head())
    return df


def batch_calculate_spi(calculator, df, prime_col, target_col):
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Calculating SPI"):
        try:
            prime = str(row[prime_col])
            target = str(row[target_col])
            spi = calculator.calculate_spi(prime, target)
            results.append(spi)
        except Exception as e:
            print(f"Error at row {idx}: {e}")
            results.append(None)
    
    df['spi'] = results
    return df


def batch_calculate_priming(calculator, df, positive_col, negative_col, target_col):
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Calculating Priming Effect"):
        try:
            positive = str(row[positive_col])
            negative = str(row[negative_col])
            target = str(row[target_col])
            pe, details = calculator.calculate_priming_effect(positive, negative, target)
            results.append({
                'pe': pe,
                'Dp': details['Dp'],
                'Dn': details['Dn'],
                'diff': details['diff']
            })
        except Exception as e:
            print(f"Error at row {idx}: {e}")
            results.append({'pe': None, 'Dp': None, 'Dn': None, 'diff': None})
    
    results_df = pd.DataFrame(results)
    df = pd.concat([df, results_df], axis=1)
    return df


def calculate_pe(gamma, diff):
    exp_term = np.exp(gamma * diff)
    return (exp_term - 1) / (exp_term + 1)


def plot_spi_vs_gamma(save_path=None):
    gamma_values = np.linspace(0.1, 10, 200) 
    
    differences = [-0.8, -0.5, -0.2, 0.2, 0.5, 0.8]
    colors = ['#d62728', '#ff7f0e', '#1f77b4', '#1f77b4', '#ff7f0e', '#d62728'] 
    linestyles = ['--', '--', '--', '-', '-', '-'] 
    labels = ['Large Negative (-0.8)', 'Medium Negative (-0.5)', 'Small Negative (-0.2)', 
             'Small Positive (0.2)', 'Medium Positive (0.5)', 'Large Positive (0.8)']
    
    plt.figure(figsize=(7, 5))
    
    for diff, color, label, ls in zip(differences, colors, labels, linestyles):
        pe_values = [calculate_pe(gamma, diff) for gamma in gamma_values]
        plt.plot(gamma_values, pe_values, color=color, label=label, linewidth=2, linestyle=ls)
    
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    plt.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(y=-1, color='gray', linestyle=':', alpha=0.5)
    
    for gamma in [1, 2, 3, 5]:
        plt.axvline(x=gamma, color='gray', linestyle=':', alpha=0.3)
        plt.text(gamma, -1.15, f'γ={gamma}', ha='center', va='center', fontsize=10)
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Gamma (γ)', fontsize=16)
    plt.ylabel('Syntactic Preservation Index (SPI)', fontsize=18)
    plt.title('SPI vs Gamma for Kernel Difference Values', fontsize=18, pad=20)
    
    plt.legend(bbox_to_anchor=(0.7, 0.5), loc='center left', fontsize=14, 
              borderaxespad=0., frameon=True, fancybox=True, shadow=True)
    
    plt.xlim(0, 10)
    plt.ylim(-1.2, 1.2)
    
    plt.subplots_adjust(right=0.8)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    
    print("="*60)
    print("Example 1: Calculate SPI between two sentences")
    print("="*60)
    
    calculator = SPICalculator(gamma=3.0)
    
    sentence1 = "The cat sits on the mat"
    sentence2 = "The dog lies on the carpet"
    
    spi = calculator.calculate_spi(sentence1, sentence2)
    print(f"\nSentence 1: {sentence1}")
    print(f"Sentence 2: {sentence2}")
    print(f"SPI: {spi:.4f}")
    
    print("\n" + "="*60)
    print("Example 2: Calculate Priming Effect")
    print("="*60)
    
    positive_prime = "The talented artist performs street art to the audience"
    negative_prime = "The talented artist performs the audience street art"
    target = "The skilled painter produced an amazing artwork for the gallery"
    
    pe, details = calculator.calculate_priming_effect(positive_prime, negative_prime, target)
    print(f"\nPositive Prime: {positive_prime}")
    print(f"Negative Prime: {negative_prime}")
    print(f"Target: {target}")
    print(f"Priming Effect: {pe:.4f}")
    print(f"Dp: {details['Dp']:.4f}, Dn: {details['Dn']:.4f}, Diff: {details['diff']:.4f}")
    
    print("\n" + "="*60)
    print("Example 3: Visualization")
    print("="*60)
    
    plot_spi_vs_gamma()