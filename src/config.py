import os
import tensorflow as tf

# --- Environment & Paths ---
COCO_DATASET_DIR = '/kaggle/input/coco-2017-dataset/coco2017'
COCO_ANNOTATIONS_DIR = os.path.join(COCO_DATASET_DIR, 'annotations')
COCO_TRAIN_DIR = os.path.join(COCO_DATASET_DIR, 'train2017')
COCO_VAL_DIR = os.path.join(COCO_DATASET_DIR, 'val2017')

# Pre-extracted features from your notebooks
MILESTONE_1_DIR = '/kaggle/input/mile-stone-1'  # InceptionV3 features
IMG_CAP_V1_DIR = '/kaggle/input/img-cap-v1'     # VGG16 features and preprocessed data

# Output paths
SAVE_PATH = "./models"
RECORDS_DIR = "./records"
PLOTS_DIR = "./plots"

# Feature paths
# InceptionV3 features (from milestone-1)
INCEPTION_V3_TRAIN_FEATURES = os.path.join(MILESTONE_1_DIR, 'train_features.pkl')
INCEPTION_V3_VAL_FEATURES = os.path.join(MILESTONE_1_DIR, 'val_features.pkl')

# VGG16 features (from img-cap-v1)
VGG16_FEATURES_PATH = os.path.join(IMG_CAP_V1_DIR, 'image_features.npy')

# Preprocessed caption data (from img-cap-v1)
PADDED_SEQUENCES_PATH = os.path.join(IMG_CAP_V1_DIR, 'padded_sequences.npy')
VOCAB_PATH = os.path.join(IMG_CAP_V1_DIR, 'idx_to_word.pkl')
METADATA_PATH = os.path.join(IMG_CAP_V1_DIR, 'metadata.pkl')

# --- Global Parameters ---
DEVICE = "GPU:0" if len(tf.config.list_physical_devices('GPU')) > 0 else "CPU:0"
SEED = 42

# --- Dataset Parameters ---
VOCAB_SIZE = 29854 
MAX_LENGTH = 59 
FEATURE_DIM_INCEPTION_V3 = 2048  
FEATURE_DIM_VGG16 = 512  

# --- Model Hyperparameters ---

# Hybrid LSTM-Transformer
HYBRID_LSTM_UNITS = 256
HYBRID_D_MODEL = 256
HYBRID_NUM_HEADS = 4
HYBRID_DROPOUT_RATE = 0.2

# Full Transformer
TRANSFORMER_D_MODEL = 512
TRANSFORMER_NUM_HEADS = 8
TRANSFORMER_NUM_LAYERS = 4
TRANSFORMER_DFF = 2048
TRANSFORMER_DROPOUT_RATE = 0.1

# Simple Transformer
SIMPLE_TRANSFORMER_D_MODEL = 256
SIMPLE_TRANSFORMER_NUM_HEADS = 4
SIMPLE_TRANSFORMER_NUM_LAYERS = 2
SIMPLE_TRANSFORMER_DROPOUT_RATE = 0.1

# --- Training Configuration ---
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 5
DEFAULT_LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 5

# --- Experiment Matrix ---
ACTIVE_EXPERIMENT_NAME = "hybrid_inception_v3"

EXPERIMENTS = {
    "hybrid_inception_v3": {
        "model_type": "hybrid",
        "feature_type": "inception_v3",
        "epochs": 5,
        "batch_size": 32,
        "learning_rate": 0.001,
    },
    "transformer_inception_v3": {
        "model_type": "transformer",
        "feature_type": "inception_v3",
        "epochs": 5,
        "batch_size": 32,
        "learning_rate": 0.0001,
    },
    "simple_transformer_inception_v3": {
        "model_type": "simple_transformer",
        "feature_type": "inception_v3",
        "epochs": 5,
        "batch_size": 32,
        "learning_rate": 0.0001,
    },
}

def get_experiment_config(experiment_name: str = None) -> dict:
    """Get configuration for specified experiment."""
    if experiment_name is None:
        experiment_name = ACTIVE_EXPERIMENT_NAME
    
    if experiment_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    
    return EXPERIMENTS[experiment_name]
