import os
import random
import numpy as np
import tensorflow as tf
import logging
from typing import Dict, Any


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")


def setup_logging(log_file: str = None, level: int = logging.INFO):
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file (if None, logs only to console)
        level: Logging level
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers
    )
    
    logging.info(f"Logging initialized{' to ' + log_file if log_file else ''}")


def load_experiment_config() -> Dict[str, Any]:
    """
    Load experiment configuration from environment or config file.
    
    Returns:
        Dictionary containing experiment configuration
    """
    from . import config
    
    experiment_name = os.environ.get('EXPERIMENT_NAME', config.ACTIVE_EXPERIMENT_NAME)
    return config.get_experiment_config(experiment_name)


def calculate_bleu_score(references, hypotheses, max_n=4):
    """
    Calculate BLEU score for generated captions.
    
    Args:
        references: List of reference captions
        hypotheses: List of generated captions
        max_n: Maximum n-gram order
        
    Returns:
        BLEU score
    """
    from nltk.translate.bleu_score import corpus_bleu
    
    # Ensure references is a list of lists
    if references and not isinstance(references[0], list):
        references = [[ref] for ref in references]
    
    score = corpus_bleu(references, hypotheses)
    return score * 100  # Convert to percentage


def decode_sequence(sequence, idx_to_word):
    """
    Decode a tokenized sequence back to text.
    
    Args:
        sequence: Tokenized sequence (list of integers)
        idx_to_word: Dictionary mapping token IDs to words
        
    Returns:
        Decoded string
    """
    words = []
    for idx in sequence:
        if idx == 0:
            continue
        if idx in idx_to_word:
            word = idx_to_word[idx]
            if word in ['<start>', '<end>']:
                continue
            words.append(word)
    
    return ' '.join(words)


def generate_caption(model, image_feature, word_to_idx, idx_to_word, max_length=59):
    """
    Generate a caption for an image using the trained model.
    
    Args:
        model: Trained captioning model
        image_feature: Pre-extracted image features
        word_to_idx: Dictionary mapping words to token IDs
        idx_to_word: Dictionary mapping token IDs to words
        max_length: Maximum caption length
        
    Returns:
        Generated caption string
    """
    # Start with the <start> token
    caption = [word_to_idx.get('<start>', 1)]
    
    for _ in range(max_length):
        # Prepare input
        sequence = np.array([caption + [0] * (max_length - 1 - len(caption))])
        sequence = sequence[:, :max_length - 1]
        
        # Predict next word
        predictions = model.predict([image_feature.reshape(1, -1), sequence], verbose=0)
        
        # Get the word with highest probability
        next_word_idx = np.argmax(predictions[0, len(caption) - 1, :])
        
        # Stop if we predict <end> or reach max length
        if idx_to_word.get(next_word_idx) == '<end>':
            break
        
        caption.append(next_word_idx)
    
    # Decode the caption
    return decode_sequence(caption, idx_to_word)


def create_directories():
    """Create necessary directories for the project."""
    from . import config
    
    dirs = [config.SAVE_PATH, config.RECORDS_DIR, config.PLOTS_DIR]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory created/verified: {directory}")
