import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Embedding, Dropout, LayerNormalization,
    MultiHeadAttention, LSTM, Add, Reshape
)
import numpy as np


def build_hybrid_model(
    vocab_size: int,
    max_length: int,
    feature_dim: int,
    lstm_units: int = 256,
    d_model: int = 256,
    num_heads: int = 4,
    dropout_rate: float = 0.2
) -> Model:
    """
    Build a Hybrid LSTM-Transformer model for image captioning.
    
    Combines LSTM for sequential processing with Transformer cross-attention
    for better attention to image features.
    
    Args:
        vocab_size: Size of the vocabulary
        max_length: Maximum caption length
        feature_dim: Dimension of image features
        lstm_units: Number of LSTM units
        d_model: Dimensionality of embeddings
        num_heads: Number of attention heads
       dropout_rate: Dropout rate
        
    Returns:
        Compiled Keras model
    """
    # Image input
    image_input = Input(shape=(feature_dim,), name='image_input')
    image_proj = Dense(d_model,activation='relu')(image_input)
    image_proj = Dropout(dropout_rate)(image_proj)
    
    # Text input
    caption_input = Input(shape=(max_length - 1,), name='caption_input')
    
    # Text embedding
    text_embedding = Embedding(vocab_size, d_model, name='text_embedding')(caption_input)
    text_embedding = Dropout(dropout_rate)(text_embedding)
    
    # LSTM layer for sequence processing
    lstm_output = LSTM(
        lstm_units,
        return_sequences=True,
        name='lstm'
    )(text_embedding)
    
    # Prepare image features for cross-attention
    image_features = Reshape((1, d_model))(image_proj)
    
    # Transformer cross-attention
    attended_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout_rate,
        name='cross_attention'
    )(lstm_output, image_features)
    
    # Combine LSTM and attention outputs
    combined = Add()([lstm_output, attended_output])
    combined = LayerNormalization()(combined)
    combined = Dropout(dropout_rate)(combined)
    
    # Output layer
    output = Dense(vocab_size, activation='softmax')(combined)
    
    model = Model(inputs=[image_input, caption_input], outputs=output)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_transformer_decoder_model(
    vocab_size: int,
    max_length: int,
    feature_dim: int,
    d_model: int = 512,
    num_heads: int = 8,
    num_layers: int = 4,
    dff: int = 2048,
    dropout_rate: float = 0.1
) -> Model:
    """
    Build a complete Transformer decoder model for image captioning.
    
    Args:
        vocab_size: Size of the vocabulary
        max_length: Maximum caption length
        feature_dim: Dimension of image features
        d_model: Dimensionality of embeddings
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dff: Dimensionality of feed-forward network
        dropout_rate: Dropout rate
        
    Returns:
        Compiled Keras model
    """
    # Image Input
    image_input = Input(shape=(feature_dim,), name='image_input')
    image_projection = Dense(d_model, activation='linear', name='image_projection')(image_input)
    image_projection = Dropout(dropout_rate)(image_projection)
    
    # Text Input (caption sequences)
    caption_input = Input(shape=(max_length - 1,), name='caption_input')
    
    # Text Embedding
    text_embedding = Embedding(vocab_size, d_model, name='text_embedding')(caption_input)
    
    # Add positional encoding
    def positional_encoding(seq_len, d_model):
        pos = np.arange(seq_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates
        
        # Apply sin to even indices, cos to odd indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:,1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    pos_encoding = positional_encoding(max_length - 1, d_model)
    text_embedding = text_embedding + pos_encoding
    text_embedding = Dropout(dropout_rate)(text_embedding)
    
    # Prepare image features for attention  
    image_features = Reshape((1, d_model))(image_projection)
    
    # Transformer Decoder Layers
    x = text_embedding
    
    for i in range(num_layers):
        # Self-attention
        attn_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
            name=f'decoder_self_attention_{i}'
        )(x, x)
        
        # Add & Norm
        x = LayerNormalization(name=f'decoder_ln1_{i}')(x + attn_output)
        
        # Cross-attention with image features
        cross_attn_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
            name=f'decoder_cross_attention_{i}'
        )(x, image_features)
        
        # Add & Norm
        x = LayerNormalization(name=f'decoder_ln2_{i}')(x + cross_attn_output)
        
        # Feed-forward network
        ff_output = Dense(dff, activation='relu', name=f'decoder_ff1_{i}')(x)
        ff_output = Dropout(dropout_rate)(ff_output)
        ff_output = Dense(d_model, name=f'decoder_ff2_{i}')(ff_output)
        ff_output = Dropout(dropout_rate)(ff_output)
        
        # Add & Norm
        x = LayerNormalization(name=f'decoder_ln3_{i}')(x + ff_output)
    
    # Final output layer
    output = Dense(vocab_size, activation='softmax', name='output')(x)
    
    # Build model
    model = Model(inputs=[image_input, caption_input], outputs=output)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-9, clipvalue=1.0),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_simple_transformer_model(
    vocab_size: int,
    max_length: int,
    feature_dim: int,
    d_model: int = 256,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout_rate: float = 0.1
) -> Model:
    """
    Build a simplified Transformer decoder model for faster training.
    
    Args:
        vocab_size: Size of the vocabulary
        max_length: Maximum caption length
        feature_dim: Dimension of image features
        d_model: Dimensionality of embeddings
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout_rate: Dropout rate
        
    Returns:
        Compiled Keras model
    """
    # Image input
    image_input = Input(shape=(feature_dim,), name='image_input')
    image_dense = Dense(d_model, activation='relu')(image_input)
    image_dropout = Dropout(dropout_rate)(image_dense)
    
    # Text input
    caption_input = Input(shape=(max_length - 1,), name='caption_input')
    
    # Text embedding
    text_embedding = Embedding(vocab_size, d_model)(caption_input)
    text_embedding = Dropout(dropout_rate)(text_embedding)
    
    # Prepare image features for attention
    image_features = Reshape((1, d_model))(image_dropout)
    
    # Transformer layers
    x = text_embedding
    
    for i in range(num_layers):
        # Multi-head attention
        attn_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads
        )(x, x)
        
        x = LayerNormalization()(x + attn_output)
        
        # Cross-attention with image
        cross_attn = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads
        )(x, image_features)
        
        x = LayerNormalization()(x + cross_attn)
        
        # Feed forward
        ff = Dense(d_model * 2, activation='relu')(x)
        ff = Dense(d_model)(ff)
        x = LayerNormalization()(x + ff)
    
    # Output layer
    output = Dense(vocab_size, activation='softmax')(x)
    
    model = Model(inputs=[image_input, caption_input], outputs=output)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=1.0),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model