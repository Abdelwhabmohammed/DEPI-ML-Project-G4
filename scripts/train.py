import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import *
from src.config import *
from src.model import build_hybrid_model, build_transformer_decoder_model, build_simple_transformer_model
from src.data import load_features_and_captions
from src.dataset import CaptionDataGenerator
from src.engine import train_model, plot_training_history
from src.utils import set_seed, setup_logging, create_directories


def main():
    """Main training function."""
    # Setup
    setup_logging()
    set_seed(SEED)
    create_directories()
    
    # Load experiment configuration
    experiment_name = os.environ.get('EXPERIMENT_NAME', ACTIVE_EXPERIMENT_NAME)
    exp_config = get_experiment_config(experiment_name)
    
    print(f"\n{'='*60}")
    print(f"Training Experiment: {experiment_name}")
    print(f"{'='*60}\n")
    
    # Extract configuration
    model_type = exp_config['model_type']
    feature_type = exp_config['feature_type']
    epochs = exp_config['epochs']
    batch_size = exp_config['batch_size']
    learning_rate = exp_config['learning_rate']
    
    print(f"Model Type: {model_type}")
    print(f"Feature Type: {feature_type}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}\n")
    
    # Load data
    print("Loading features and captions...")
    (train_features, val_features,
     train_sequences, val_sequences,
     metadata, vocab) = load_features_and_captions(
        feature_type=feature_type,
        image_features_path=INCEPTION_V3_TRAIN_FEATURES,
        val_image_features_path=INCEPTION_V3_VAL_FEATURES,
        padded_sequences_path=PADDED_SEQUENCES_PATH,
        vocab_path=VOCAB_PATH,
        metadata_path=METADATA_PATH
    )
    
    # Get dimensions
    vocab_size = metadata.get('vocab_size', VOCAB_SIZE)
    max_length = metadata.get('max_length', MAX_LENGTH)
    feature_dim = train_features.shape[1]
    
    print(f"\nData loaded:")
    print(f"  Train samples: {len(train_features):,}")
    print(f"  Val samples: {len(val_features):,}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Max length: {max_length}")
    print(f"  Feature dim: {feature_dim}\n")
    
    # Create data generators
    train_generator = CaptionDataGenerator(
        train_features, train_sequences, vocab_size, max_length, batch_size
    )
    val_generator = CaptionDataGenerator(
        val_features, val_sequences, vocab_size, max_length, batch_size
    )
    
    print("Data generators created.\n")
    
    # Build model
    print(f"Building {model_type} model...")
    if model_type == 'hybrid':
        model = build_hybrid_model(
            vocab_size, max_length, feature_dim,
            lstm_units=HYBRID_LSTM_UNITS,
            d_model=HYBRID_D_MODEL,
            num_heads=HYBRID_NUM_HEADS,
            dropout_rate=HYBRID_DROPOUT_RATE
        )
    elif model_type == 'transformer':
        model = build_transformer_decoder_model(
            vocab_size, max_length, feature_dim,
            d_model=TRANSFORMER_D_MODEL,
            num_heads=TRANSFORMER_NUM_HEADS,
            num_layers=TRANSFORMER_NUM_LAYERS,
            dff=TRANSFORMER_DFF,
            dropout_rate=TRANSFORMER_DROPOUT_RATE
        )
    elif model_type == 'simple_transformer':
        model = build_simple_transformer_model(
            vocab_size, max_length, feature_dim,
            d_model=SIMPLE_TRANSFORMER_D_MODEL,
            num_heads=SIMPLE_TRANSFORMER_NUM_HEADS,
            num_layers=SIMPLE_TRANSFORMER_NUM_LAYERS,
            dropout_rate=SIMPLE_TRANSFORMER_DROPOUT_RATE
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print("Model built successfully.\n")
    model.summary()
    
    # Train model
    print(f"\nStarting training for {epochs} epochs...")
    trained_model, history = train_model(
        model=model,
        train_generator=train_generator,
        val_generator=val_generator,
        epochs=epochs,
        model_save_dir=SAVE_PATH,
        model_name_prefix=experiment_name,
        feature_type=feature_type
    )
    
    # Plot training history
    plot_training_history(
        history,
        output_dir=PLOTS_DIR,
        filename=f'{experiment_name}_history.png'
    )
    
    print(f"\n{'='*60}")
    print(f"Training completed for {experiment_name}!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
