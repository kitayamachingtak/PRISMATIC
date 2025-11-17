import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10


def load_positive_rates(analysis_csv):
    df = pd.read_csv(analysis_csv)
    positive_rates = []
    for _, row in df.iterrows():
        rate_str = row['Positive_Ratio']
        rate_value = float(rate_str.strip('%'))
        positive_rates.append(rate_value)
    return positive_rates


def plot_multi_model_comparison(model_files, model_names, colors, markers, save_path=None):
    categories = [
        'Simple Active', 'Simple Passive', 'Prepositional Active', 'Prepositional Passive',
        'Embedded Active', 'Embedded Passive', 'Mediopassive Like Active', 'Mediopassive',
        'Simple Double Object', 'Simple Prepositional Object', 'Complex Double Object',
        'Complex Prepositional Object', 'Double Object with Clause',
        'Prepositional Object with Clause', 'S-Genitive', 'Of-Genitive'
    ]
    
    all_rates = []
    for model_file in model_files:
        rates = load_positive_rates(model_file)
        all_rates.append(rates)
    
    x = np.arange(len(categories))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6))
    
    for i in range(2):
        rates = all_rates[i]
        ax1.plot(x, rates, f'{markers[i]}-', label=model_names[i], 
                linewidth=2, color=colors[i], markersize=5)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.set_ylabel('Positive Rate (%)')
    ax1.set_ylim(15, 90)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=True)
    ax1.set_title(f'{model_names[0]} vs {model_names[1]} Correct Priming Rates Comparison', pad=15)
    
    for i in range(2, 4):
        rates = all_rates[i]
        ax2.plot(x, rates, f'{markers[i]}-', label=model_names[i], 
                linewidth=2, color=colors[i], markersize=5)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    ax2.set_ylabel('Positive Rate (%)')
    ax2.set_ylim(15, 90)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=True)
    ax2.set_title(f'{model_names[2]} vs {model_names[3]} Correct Priming Rates Comparison', pad=15)
    
    plt.tight_layout()
    
    if save_path:
        pdf_path = save_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"Figures saved!")
        print(f"- {pdf_path}")
        print(f"- {save_path}")
    
    plt.show()


if __name__ == "__main__":
    model_files = [
        'analysis_with_prime_blip2.csv',
        'analysis_with_prime_llava.csv',
        'analysis_with_prime_model1.csv',
        'analysis_with_prime_model2.csv'
    ]
    
    model_names = [
        'Blip2',
        'Llava',
        'Model 1 - Dual',
        'Model 2 - Fusion'
    ]
    
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
    markers = ['o', 's', '^', 'D']
    
    print("="*60)
    print("Multi-Model Comparison")
    print("="*60)
    
    plot_multi_model_comparison(
        model_files, 
        model_names, 
        colors, 
        markers, 
        'priming_rates_comparison.png'
    )
    
    print("\n" + "="*60)
    print("Complete!")
    print("="*60)