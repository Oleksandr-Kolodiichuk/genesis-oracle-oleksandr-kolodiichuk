import jax
import jax.numpy as jnp
import flax.linen as nn

class HeatSurrogate(nn.Module):
    @nn.compact
    def __call__(self, x):
        # x is expected to be of shape (..., 2) representing (space, time)
        
        # 4 hidden layers with 32 neurons each
        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        
        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        
        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        
        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        
        # Output layer to predict 1D scalar (temperature u)
        x = nn.Dense(1)(x)
        return x

if __name__ == "__main__":
    # Execution block demonstrating how to initialize the model's weights
    
    # 1. Initialize the model
    model = HeatSurrogate()
    
    # 2. Create a PRNGKey for reproducibility
    key = jax.random.PRNGKey(42)
    
    # 3. Create dummy input data of shape (batch_size, 2)
    # where 2 corresponds to space x and time t
    dummy_input = jnp.ones((1, 2))
    
    # 4. Initialize the model's weights explicitly
    variables = model.init(key, dummy_input)
    
    print("Model initialized successfully!")
    
    # Extract params to verify initialization
    params = variables['params']
    print(f"Initialized parameter keys: {list(params.keys())}")
    
    # Do a forward pass to verify output shape
    output = model.apply(variables, dummy_input)
    print(f"Output shape (expected (1, 1)): {output.shape}")
