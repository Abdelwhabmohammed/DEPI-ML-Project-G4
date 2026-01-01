import os
import sys
import pickle
import numpy as np
from tensorflow.keras.models import model_from_json

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import *
from src.data import load_features_and_captions
from src.utils import calculate_bleu_score, generate_caption, decode_sequence


def load_trained_model(model_path: str):
    """Load a trained model from pickle file."""
    with open(model_path, 'rb') as f:
        model_info = pickle.load(f)
    
    model = model_from_json(model_info['model_config'])
    model.set_weights(model_info['model_weights'])
    return model


def evaluate_model(
    model,
    test_features,
    test_sequences,
    vocab,
    idx_to_word,
    max_length,
    num_samples=1000
):
    """
    Evaluate model using BLEU score.
    
    Args:
        model: Trained model
        test_features: Test image features
        test_sequences: Test caption sequences
        vocab: Word to index mapping
        idx_to_word: Index to word mapping
        max_length: Maximum caption length
        num_samples: Number of samples to evaluate
        
    Returns:
        BLEU score
    """
    references = []
    hypotheses = []
    
    # Limit evaluation to num_samples
    num_samples = min(num_samples, len(test_features))
    
    print(f"Evaluating on {num_samples} samples...")
    
    for i in range(num_samples):
        if i % 100 == 0:
            print(f"  Progress: {i}/{num_samples}")
        
        # Generate caption
        generated_caption = generate_caption(
            model, test_features[i], vocab, idx_to_word, max_length
        )
        
        # Get reference caption
        reference_caption = decode_sequence(test_sequences[i], idx_to_word)
        
        # Tokenize for BLEU calculation
        references.append([reference_caption.split()])
        hypotheses.append(generated_caption.split())
    
    # Calculate BLEU score
    bleu_score = calculate_bleu_score(references, hypotheses)
    
    return bleu_score, references, hypotheses


def main():
    """Main evaluation function."""
    # Configuration
    experiment_name = os.environ.get('EXPERIMENT_NAME', ACTIVE_EXPERIMENT_NAME)
    model_path = os.path.join(SAVE_PATH, f'{experiment_name}_inception_v3_model.pkl')
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {experiment_name}")
    print(f"{'='*60}\n")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using scripts/train.py")
        return
    
    # Load model
    print("Loading trained model...")
    model = load_trained_model(model_path)
    print("Model loaded successfully.\n")
    
    # Load data
    print("Loading test data...")
    (_, test_features,
     _, test_sequences,
     metadata, vocab) = load_features_and_captions(
        feature_type='inception_v3',
        image_features_path=INCEPTION_V3_TRAIN_FEATURES,
        val_image_features_path=INCEPTION_V3_VAL_FEATURES,
        padded_sequences_path=PADDED_SEQUENCES_PATH,
        vocab_path=VOCAB_PATH,
        metadata_path=METADATA_PATH
    )
    
    # Create reverse vocabulary
    idx_to_word = {v: k for k, v in vocab.items()}
    max_length = metadata.get('max_length', MAX_LENGTH)
    
    print(f"Test samples: {len(test_features):,}\n")
    
    # Evaluate
    bleu_score, references, hypotheses = evaluate_model(
        model, test_features, test_sequences, vocab, idx_to_word, max_length
    )
    
    print(f"\n{'='*60}")
    print(f"BLEU Score: {bleu_score:.2f}")
    print(f"{'='*60}\n")
    
    # Show some examples
    print("Sample Captions:\n")
    for i in range(min(5, len(hypotheses))):
        print(f"Example {i+1}:")
        print(f"  Reference: {' '.join(references[i][0])}")
        print(f"  Generated: {' '.join(hypotheses[i])}")
        print()
    
    # Save results
    results_path = os.path.join(RECORDS_DIR, f'{experiment_name}_evaluation.txt')
    os.makedirs(RECORDS_DIR, exist_ok=True)
    
    with open(results_path, 'w') as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"BLEU Score: {bleu_score:.2f}\n\n")
        f.write("Sample Captions:\n\n")
        for i in range(min(10, len(hypotheses))):
            f.write(f"Example {i+1}:\n")
            f.write(f"  Reference: {' '.join(references[i][0])}\n")
            f.write(f"  Generated: {' '.join(hypotheses[i])}\n\n")
    
    print(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()