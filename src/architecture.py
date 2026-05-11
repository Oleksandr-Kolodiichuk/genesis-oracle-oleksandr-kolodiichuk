import numpy as np
import keras
from keras import layers

# 1. Sequence Windowing
def create_sequences(data, window_size=50):
    """Slices a 1D signal into overlapping windows of fixed size."""
    sequences = []
    for i in range(len(data) - window_size + 1):
        sequences.append(data[i : i + window_size])
    return np.array(sequences)

def prepare_data(data, split_period_index):
    """Splits raw data into training (normal) and testing (anomaly) sets."""
    train_data = data[:split_period_index]
    test_data = data[split_period_index:]
    return create_sequences(train_data), create_sequences(test_data)

# 2. Updated Conv1D Encoder
class SignalCompression(layers.Layer):
    def __init__(self, latent_dim=8, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim

    def build(self, input_shape):
        # Add channel dimension: (50) -> (50, 1)
        self.reshape = layers.Reshape((input_shape[-1], 1))
        # Downsample temporal dimension: (50, 1) -> (25, 16)
        self.conv1 = layers.Conv1D(16, kernel_size=3, strides=2, padding='same', activation='relu')
        # Further downsampling: (25, 16) -> (5, 32)
        self.conv2 = layers.Conv1D(32, kernel_size=3, strides=5, padding='same', activation='relu')
        # Flatten temporal and filter dimensions (5 * 32 = 160)
        self.flatten = layers.Flatten()
        # Final bottleneck compression to exactly 8 units
        self.dense = layers.Dense(self.latent_dim, activation='relu')

    def call(self, inputs):
        x = self.reshape(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        return self.dense(x)

# 3. Updated Conv1D Decoder
class SignalExpansion(layers.Layer):
    def __init__(self, original_dim=50, **kwargs):
        super().__init__(**kwargs)
        self.original_dim = original_dim

    def build(self, input_shape):
        # Project latent vector (8) back to convolutional shape (5 * 32 = 160)
        self.dense = layers.Dense(5 * 32, activation='relu')
        self.reshape = layers.Reshape((5, 32))
        # Upsample temporal dimension: (5, 32) -> (25, 16)
        self.conv_t1 = layers.Conv1DTranspose(16, kernel_size=3, strides=5, padding='same', activation='relu')
        # Final reconstruction to original length: (25, 16) -> (50, 1)
        self.conv_t2 = layers.Conv1DTranspose(1, kernel_size=3, strides=2, padding='same', activation='linear')
        # Flatten back to match original input shape (batch_size, 50)
        self.flatten = layers.Flatten()

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.reshape(x)
        x = self.conv_t1(x)
        x = self.conv_t2(x)
        return self.flatten(x)

class PhysicsAutoencoder(keras.Model):
    def __init__(self, latent_dim=8, original_dim=50, **kwargs):
        super().__init__(**kwargs)
        self.encoder = SignalCompression(latent_dim=latent_dim)
        self.decoder = SignalExpansion(original_dim=original_dim)

    def call(self, inputs):
        # End-to-end forward pass: Input -> Latent Space -> Reconstruction
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded