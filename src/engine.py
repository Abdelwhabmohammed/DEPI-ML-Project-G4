import os
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model
from typing import Tuple, Optional
import numpy as np


class AccuracyImprovementCallback(tf.keras.callbacks.Callback):
    """Callback to log validation accuracy improvements."""
    
    def __init__(self):
        super().__init__()
        self.best_val_accuracy = -1
    
    def on_epoch_end(self, epoch, logs=None):
        current_val_accuracy = logs.get('val_accuracy')
        if current_val_accuracy is not None:
            if current_val_accuracy > self.best_val_accuracy:
                print(f"\nEpoch {epoch+1}: Validation accuracy improved from {self.best_val_accuracy:.4f} to {current_val_accuracy:.4f}. Saving model.")
                self.best_val_accuracy = current_val_accuracy
            else:
                print(f"\nEpoch {epoch+1}: Validation accuracy did not improve. Best so far: {self.best_val_accuracy:.4f}.")


def train_model(
    model: Model,
    train_generator,
    val_generator,
    epochs: int = 5,
    model_save_dir: str = './models',
    model_name_prefix: str = 'caption_model',
    feature_type: str = 'inception_v3'
) -> Tuple[Model, tf.keras.callbacks.History]:
    """
    Train the captioning model.
    
    Args:
        model: Compiled Keras model
        train_generator: Training data generator
        val_generator: Validation data generator
        epochs: Number of training epochs
        model_save_dir: Directory to save models
        model_name_prefix: Prefix for saved model files
        feature_type: Type of features being used
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    # Ensure save directory exists
    os.makedirs(model_save_dir, exist_ok=True)
    
    print(f"Training model with {feature_type} features for {epochs} epochs...")
    
    # Callbacks for saving the best model and early stopping
    checkpoint_filepath = f'{model_save_dir}/{model_name_prefix}_{feature_type}_best_model.keras'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min',
        verbose=1
    )
    
    accuracy_callback = AccuracyImprovementCallback()
    
    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[model_checkpoint_callback, early_stopping_callback, accuracy_callback]
    )
    
    print(f"Training complete for {feature_type} features.")
    
    # Save model architecture and weights separately
    model_info = {
        "model_config": model.to_json(),
        "model_weights": model.get_weights()  
    }
    
    with open(f'{model_save_dir}/{model_name_prefix}_{feature_type}_model.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    
    # Save the training history separately
    with open(f'{model_save_dir}/{model_name_prefix}_{feature_type}_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    
    print(f"Model and history saved to {model_save_dir}/")
    
    return model, history


def plot_training_history(
    history: tf.keras.callbacks.History,
    output_dir: str = './plots',
    filename: str = 'training_history.png'
):
    """
    Plot training and validation metrics.
    
    Args:
        history: Keras training history object
        output_dir: Directory to save the plot
        filename: Filename for the saved plot
    """
    import matplotlib.pyplot as plt
    
    if not hasattr(history, "history"):
        print("No history data found.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = history.history.keys()
    print("Available metrics:", list(metrics))
    
    plt.figure(figsize=(10, 4))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    if 'accuracy' in metrics:
        plt.plot(history.history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in metrics:
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.title("Accuracy")
    
    # Loss
    plt.subplot(1, 2, 2)
    if 'loss' in metrics:
        plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in metrics:
        plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title("Loss")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}", bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to {output_dir}/{filename}")
