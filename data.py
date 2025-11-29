import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any


def load_features_and_captions(
    feature_type: str,
    image_features_path: str,
    val_image_features_path: str = None,
    padded_sequences_path: str = None,
    vocab_path: str = None,
    metadata_path: str = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], Dict[str, int]]:
    """
    Load pre-extracted features and caption sequences.
    
    Args:
        feature_type: 'inception_v3' or 'vgg16'
        image_features_path: Path to train features
        val_image_features_path: Path to validation features (for InceptionV3)
        padded_sequences_path: Path to padded caption sequences
        vocab_path: Path to vocabulary
        metadata_path: Path to metadata
        
    Returns:
        Tuple of (train_features, val_features, train_sequences, val_sequences, metadata, vocab)
    """
    print(f"Loading {feature_type} features...")
    
    # Load metadata
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
    else:
        metadata = {}
    
    # Load vocabulary
    if vocab_path and os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
    else:
        vocab = {}
    
    # Load padded sequences
    if padded_sequences_path and os.path.exists(padded_sequences_path):
        padded_sequences = np.load(padded_sequences_path)
    else:
        raise FileNotFoundError(f"Padded sequences not found at {padded_sequences_path}")
    
    # Load features based on feature_type
    if feature_type == 'inception_v3':
        # For InceptionV3, load train and val features separately
        with open(image_features_path, 'rb') as f:
            train_image_features = pickle.load(f)
        
        # If it's a dict, convert values to a numpy array
        if isinstance(train_image_features, dict):
            train_image_features = np.array(list(train_image_features.values()))
        
        if val_image_features_path:
            with open(val_image_features_path, 'rb') as f:
                val_image_features = pickle.load(f)
            
            if isinstance(val_image_features, dict):
                val_image_features = np.array(list(val_image_features.values()))
        else:
            val_image_features = None
        
        print(f"Loaded InceptionV3 features. Train shape: {train_image_features.shape}")
        if val_image_features is not None:
            print(f"Val shape: {val_image_features.shape}")
        
        # Split padded_sequences to match the image feature split
        padded_sequences_train, padded_sequences_temp = train_test_split(
            padded_sequences, test_size=0.2, random_state=42
        )
        padded_sequences_val, _ = train_test_split(
            padded_sequences_temp, test_size=0.5, random_state=42
        )
        
    elif feature_type == 'vgg16':
        # For VGG16, load all features and split
        all_image_features = np.load(image_features_path)
        print(f"Loaded VGG16 features. Shape: {all_image_features.shape}")
        
        # Split image features and padded sequences together
        (train_image_features, X_temp_img,
         padded_sequences_train, padded_sequences_temp) = train_test_split(
            all_image_features, padded_sequences, test_size=0.2, random_state=42
        )
        (val_image_features, _,
         padded_sequences_val, _) = train_test_split(
            X_temp_img, padded_sequences_temp, test_size=0.5, random_state=42
        )
    else:
        raise ValueError(f"Invalid feature_type: {feature_type}. Must be 'inception_v3' or 'vgg16'")
    
    return (
        train_image_features,
        val_image_features,
        padded_sequences_train,
        padded_sequences_val,
        metadata,
        vocab
    )


def load_vocabulary(vocab_path: str) -> Dict[str, int]:
    """Load vocabulary from pickle file."""
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab


def load_metadata(metadata_path: str) -> Dict[str, Any]:
    """Load metadata from pickle file."""
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    return metadata
