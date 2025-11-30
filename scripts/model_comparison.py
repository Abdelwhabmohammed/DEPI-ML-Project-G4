import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import *
from src.utils import calculate_bleu_score


def compare_models():
    """
    Compare different models on image captioning task.
    
    Note: This is a template script. You'll need to add actual model loading
    and prediction code for each model you want to compare.
    """
    print("\n" + "="*60)
    print("Model Comparison for Image Captioning")
    print("="*60 + "\n")
    
    # Example BLEU scores
    model_scores = {
        "Model": [
            "Transformer",
            "BLIP",
            "ViT-GPT2",
            "BLIP2",
            "GIT-Large"
        ],
        "BLEU Score": [
            7.85,  
            22.50,  
            18.30,
            28.70,
            25.40
        ]
    }
    
    df = pd.DataFrame(model_scores)
    
    # Display results
    print("BLEU Score Comparison:\n")
    print(df.to_string(index=False))
    print()
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    colors = ['#FF6B6B' if model == "Transformer" else '#4ECDC4'
              for model in df['Model']]
    
    plt.barh(df['Model'], df['BLEU Score'], color=colors)
    plt.xlabel('BLEU Score', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.title('Image Captioning Model Comparison\\n(MS COCO Dataset)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(df['BLEU Score']):
        plt.text(v + 0.5, i, f'{v:.2f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('./plots', exist_ok=True)
    plt.savefig('./plots/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Comparison plot saved to ./plots/model_comparison.png")
    
    # Analysis
    print(f"\n{'='*60}")
    print("Analysis:")
    print(f"{'='*60}\n")
    
    best_model = df.loc[df['BLEU Score'].idxmax(), 'Model']
    best_score = df['BLEU Score'].max()
    our_score = df.loc[df['Model'] == "Transformer", 'BLEU Score'].values[0]
    
    print(f"Best performing model: {best_model} ({best_score:.2f})")
    print(f"our model score: {our_score:.2f}")
    print(f"Gap to best model: {best_score - our_score:.2f} BLEU points")
    print()
    
    print("Possible improvements:")
    print("  1. Use larger pretrained vision encoders (e.g., ViT-Large)")
    print("  2. Increase model capacity (more layers, larger hidden dimensions)")
    print("  3. Train for more epochs with better learning rate scheduling")
    print("  4. Use beam search during inference instead of greedy decoding")
    print("  5. Apply data augmentation and better preprocessing")
    print("  6. Fine-tune vision encoder instead of using frozen features")
    print()


def load_and_compare_our_models():
    """
    Compare different versions of our own models.
    """
    print("\n" + "="*60)
    print("Comparing Our Model Variants")
    print("="*60 + "\n")
    
    # Example comparison between our model variants
    our_models = {
        "Architecture": [
            "Hybrid LSTM-Transformer",
            "Full Transformer",
            "Simple Transformer"
        ],
        "Parameters (M)": [
            12.5,   
            18.3,
            8.7
        ],
        "BLEU Score": [
            7.85,  
            8.2,   
            7.1   
        ],
        "Training Time (min)": [
            120,  
            180,
            90
        ]
    }
    
    df = pd.DataFrame(our_models)
    
    print("Model Variants:\n")
    print(df.to_string(index=False))
    print()
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # BLEU Score comparison
    axes[0].bar(df['Architecture'], df['BLEU Score'], color='#FF6B6B')
    axes[0].set_ylabel('BLEU Score', fontsize=11)
    axes[0].set_title('BLEU Score by Architecture', fontsize=12, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=15)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Parameters vs BLEU
    axes[1].scatter(df['Parameters (M)'], df['BLEU Score'], s=200, c='#4ECDC4', alpha=0.6, edgecolors='black')
    for i, txt in enumerate(df['Architecture']):
        axes[1].annotate(txt, (df['Parameters (M)'].iloc[i], df['BLEU Score'].iloc[i]),
                        fontsize=8, ha='center', va='bottom')
    axes[1].set_xlabel('Parameters (M)', fontsize=11)
    axes[1].set_ylabel('BLEU Score', fontsize=11)
    axes[1].set_title('Model Complexity vs Performance', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./plots/models_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Models comparison plot saved to ./plots/models_comparison.png")


def main():
    """Main function."""
    # Compare with other models
    compare_models()
    
    # Compare own model variants
    load_and_compare_our_models()
    
    print(f"\n{'='*60}")
    print("Model comparison completed!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
