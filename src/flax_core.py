import jax
import jax.numpy as jnp
from flax import linen as nn

class MLP(nn.Module):
    """A simple Multi-Layer Perceptron (MLP) defined using flax.linen."""
    hidden_dim: int = 64
    out_dim: int = 10

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.out_dim)(x)
        return x

if __name__ == "__main__":
    # --- DEMONSTRATION OF FLAX'S STATELESS NATURE ---
    
    # 1. Instantiate the module.
    # In Keras, initializing a layer or model typically allocates its weights (implicit state).
    # In Flax, the module is just a dataclass defining the architecture. It holds no weights.
    model = MLP(hidden_dim=128, out_dim=10)

    # Create dummy input data to infer shapes during initialization
    # Batch size 1, feature size 32
    dummy_input = jnp.ones((1, 32))  

    # 2. Explicit Model Initialization (model.init)
    # Flax requires a PRNGKey to pseudo-randomly initialize weights.
    # We explicitly pass the key and the dummy input to `model.init()`.
    # This returns a frozen dictionary containing the actual initialized parameters.
    key = jax.random.PRNGKey(42)
    variables = model.init(key, dummy_input)
    
    # The 'variables' dictionary contains the state (e.g., 'params'). 
    # The `model` instance itself remains completely stateless and pure.

    # 3. Explicit Forward Pass (model.apply)
    # To run a forward pass, we must explicitly pass the state (variables) 
    # back into the model alongside the input data.
    # In Keras: `model(data)`
    # In Flax: `model.apply(variables, data)`
    output = model.apply(variables, dummy_input)

    print("Output shape:", output.shape)
    print("Flax explicitly separates model definition, initialization, and application!")
