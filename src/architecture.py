import numpy as np
import keras
from keras import layers

# 1. Sequence Windowing
def create_sequences(data, window_size=50):
    """Slices a 1D signal into overlapping windows."""
    sequences = []
    # Slide over the data to create overlapping windows
    for i in range(len(data) - window_size + 1):
        sequences.append(data[i : i + window_size])
    return np.array(sequences)

def prepare_data(data, split_period_index):
    """Splits data into train (normal) and test (anomaly)."""
    # Training set gets the clean data before period 60
    train_data = data[:split_period_index]
    # Test set gets the rest of the timeline
    test_data = data[split_period_index:]
    x_train = create_sequences(train_data)
    x_test = create_sequences(test_data)
    return x_train, x_test

# 2. The Subclassed Layer
class SignalCompression(layers.Layer):
    def __init__(self, latent_dim=8, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim

    def build(self, input_shape):
        # The bottleneck: compressing 50 timesteps to 8 dimensions
        self.dense = layers.Dense(self.latent_dim, activation='relu')

    def call(self, inputs):
        # Forward pass
        return self.dense(inputs)


# 3. The Subclassed Model
class SignalExpansion(layers.Layer):
    def __init__(self, original_dim=50, **kwargs):
        super().__init__(**kwargs)
        self.original_dim = original_dim

    def build(self, input_shape):
        # Expanding back from 8 to 50 dimensions
        self.dense = layers.Dense(self.original_dim, activation='linear')

    def call(self, inputs):
        return self.dense(inputs)

class PhysicsAutoencoder(keras.Model):
    def __init__(self, latent_dim=8, original_dim=50, **kwargs):
        super().__init__(**kwargs)
        # Instantiate our custom sub-layers
        self.encoder = SignalCompression(latent_dim=latent_dim)
        self.decoder = SignalExpansion(original_dim=original_dim)

    def call(self, inputs):
        # Chain the input through the bottleneck and back out
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded