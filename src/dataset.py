import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from typing import Tuple


class CaptionDataGenerator(Sequence):
    """
    Data generator for batching image features and caption sequences.
    
    Handles:
    - Image features (pre-extracted)
    - Caption sequences (tokenized and padded)
    - Creating input/target pairs for training
    """
    
    def __init__(
        self,
        image_features: np.ndarray,
        sequences: np.ndarray,
        vocab_size: int,
        max_length: int,
        batch_size: int = 32
    ):
        """
        Initialize the data generator.
        
        Args:
            image_features: Pre-extracted image features, shape (N, feature_dim)
            sequences: Padded caption sequences, shape (N, max_length)
            vocab_size: Size of the vocabulary
            max_length: Maximum caption length
            batch_size: Batch size for training
        """
        self.image_features = image_features
        self.sequences = sequences
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Ensure we have matching lengths
        min_len = min(len(self.image_features), len(self.sequences))
        self.indices = np.arange(min_len)
    
    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return int(np.ceil(len(self.image_features) / self.batch_size))
    
    def __getitem__(self, idx: int) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Get a batch of data.
        
        Returns:
            Tuple of (inputs, targets) where:
            - inputs: (batch_images, batch_sequences)
            - targets: one-hot encoded next tokens
        """
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = self.image_features[batch_indices]
        batch_sequences = self.sequences[batch_indices]
        
        # Model inputs: image + sequence (without last token)
        X_batch = [batch_images, batch_sequences[:, :-1]]
        
        # Model targets: sequence shifted (without first token)
        y_batch = tf.keras.utils.to_categorical(
            batch_sequences[:, 1:],
            num_classes=self.vocab_size
        )
        return tuple(X_batch), y_batch
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch."""
        np.random.shuffle(self.indices)
    
    def get_batch_info(self) -> dict:
        """Get information about the data generator."""
        return {
            'total_samples': len(self.image_features),
            'num_batches': len(self),
            'batch_size': self.batch_size,
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'feature_dim': self.image_features.shape[1]
        }
