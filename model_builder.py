"""
Model Builder for 3DES S-Box Prediction

Implements ASCAD-inspired CNN architecture with strong regularization
to promote learning of general SCA patterns rather than memorization.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from typing import Tuple


class ModelBuilder:
    """Build CNN models for S-Box prediction with anti-overfitting measures."""
    
    def __init__(self, trace_length: int = 131124):
        """
        Initialize model builder.
        
        Args:
            trace_length: Length of input power traces
        """
        self.trace_length = trace_length
        self.num_classes = 16  # 4-bit S-Box output
        self.num_sboxes = 8
    
    def build_sbox_model(
        self,
        sbox_idx: int,
        filters_1: int = 64,
        filters_2: int = 128,
        kernel_size: int = 11,
        dense_units: int = 256,
        l2_weight: float = 1e-2,
        dropout_rate: float = 0.5
    ) -> keras.Model:
        """
        Build CNN model for single S-Box prediction.
        
        Architecture based on ANSSI-FR/ASCAD with strong regularization
        to prevent overfitting and promote generalization.
        
        Args:
            sbox_idx: S-Box index (0-7) for naming
            filters_1: Number of filters in first conv layer
            filters_2: Number of filters in second conv layer
            kernel_size: Kernel size for conv layers
            dense_units: Units in dense layer
            l2_weight: L2 regularization weight
            dropout_rate: Dropout rate
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(self.trace_length, 1), name=f'sbox_{sbox_idx}_input'),
            
            # First convolutional block
            layers.Conv1D(
                filters=filters_1,
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                name=f'sbox_{sbox_idx}_conv1'
            ),
            layers.BatchNormalization(name=f'sbox_{sbox_idx}_bn1'),
            layers.AveragePooling1D(pool_size=2, name=f'sbox_{sbox_idx}_pool1'),
            
            # Second convolutional block
            layers.Conv1D(
                filters=filters_2,
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                name=f'sbox_{sbox_idx}_conv2'
            ),
            layers.BatchNormalization(name=f'sbox_{sbox_idx}_bn2'),
            layers.AveragePooling1D(pool_size=2, name=f'sbox_{sbox_idx}_pool2'),
            
            # Flatten
            layers.Flatten(name=f'sbox_{sbox_idx}_flatten'),
            
            # Dense layer with L2 regularization (CRITICAL for generalization)
            layers.Dense(
                units=dense_units,
                activation='relu',
                kernel_regularizer=regularizers.l2(l2_weight),
                name=f'sbox_{sbox_idx}_dense'
            ),
            
            # Dropout (CRITICAL for preventing memorization)
            layers.Dropout(rate=dropout_rate, name=f'sbox_{sbox_idx}_dropout'),
            
            # Output layer (16 classes for 4-bit S-Box output)
            layers.Dense(
                units=self.num_classes,
                activation='softmax',
                name=f'sbox_{sbox_idx}_output'
            )
        ], name=f'sbox_{sbox_idx}_model')
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_acc')]
        )
        
        return model
    
    
    @staticmethod
    def get_callbacks(
        name: str,
        patience: int = 10,
        min_delta: float = 0.001
    ) -> list:
        """
        Get training callbacks for a single S-Box model.
        
        Args:
            name: Unique name for the model (e.g. "kenc_sbox_0")
            patience: Early stopping patience
            min_delta: Minimum change to qualify as improvement
            
        Returns:
            List of Keras callbacks
        """
        callbacks = [
            # Early stopping to prevent overfitting
            # CRITICAL: restore_best_weights=True ensures the model object 
            # holds the best epoch's weights when training finishes.
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                min_delta=min_delta,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        return callbacks


if __name__ == "__main__":
    # Test model builder
    print("=" * 60)
    print("Testing Model Builder")
    print("=" * 60)
    
    builder = ModelBuilder(trace_length=131124)
    
    # Build all models
    models = builder.build_all_models()
    
    # Test with dummy data
    print("\n" + "=" * 60)
    print("Testing Model Prediction")
    print("=" * 60)
    
    import numpy as np
    
    # Create dummy input (batch of 2 traces)
    dummy_input = np.random.randn(2, 131124, 1).astype(np.float32)
    
    print(f"\nDummy input shape: {dummy_input.shape}")
    
    # Test prediction on first model
    print("\nTesting S-Box 0 model...")
    predictions = models[0].predict(dummy_input, verbose=0)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions sum to 1: {np.allclose(predictions.sum(axis=1), 1.0)}")
    print(f"Sample prediction: {predictions[0]}")
    print(f"Predicted class: {np.argmax(predictions[0])}")
    
    print("\n" + "=" * 60)
    print("Model Building Complete!")
    print("=" * 60)
