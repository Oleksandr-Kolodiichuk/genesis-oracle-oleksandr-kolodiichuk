import os
# Set the Keras backend to JAX
os.environ["KERAS_BACKEND"] = "jax"

import keras
# Verify the active Keras backend
print(f"Active Keras Backend: {keras.backend.backend()}")

# Create a random tensor using Keras
random_tensor = keras.random.normal(shape=(3, 3))

# Print the random tensor and its type
print(f"Random Tensor Type: {type(random_tensor)}")