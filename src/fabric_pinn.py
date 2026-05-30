import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

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

    # ==============================================================================
    # EXERCISE 3: THE DIFFERENTIABLE FABRIC (PHYSICS LOSS & OPTAX)
    # ==============================================================================

    # 1. The Math Engine: Define thermal diffusivity parameter
    alpha = 0.01

    # Define a scalar forward function for a single (x, t) point
    def predict_u(params, x, t):
        # Combine scalars into a 1D array [x, t] because your model expects shape (2,)
        inputs = jnp.array([x, t])
        # apply the model and extract the scalar temperature
        return model.apply({'params': params}, inputs)[0]

    # 2. Analytical Derivatives using jax.grad
    # First derivative w.r.t time (t is argument index 2)
    dudt = jax.grad(predict_u, argnums=2)

    # First derivative w.r.t space (x is argument index 1)
    dudx = jax.grad(predict_u, argnums=1)

    # Second derivative w.r.t space (nested grad)
    d2udx2 = jax.grad(dudx, argnums=1)

    # Compute the PDE residual: u_t - alpha * u_xx
    def pde_residual(params, x, t, alpha):
        return dudt(params, x, t) - alpha * d2udx2(params, x, t)

    # Vectorize the residual function to handle batches of thousands of points
    vmap_residual = jax.vmap(pde_residual, in_axes=(None, 0, 0, None))

    def physics_loss(params, x_colloc, t_colloc, alpha):
        # Flatten inputs from shape (N, 1) to (N,) for the scalar vmap
        res = vmap_residual(params, x_colloc.flatten(), t_colloc.flatten(), alpha)
        return jnp.mean(res ** 2)

    # --- Data Loss (For IC and BC constraints) ---
    vmap_predict = jax.vmap(predict_u, in_axes=(None, 0, 0))

    def data_loss(params, x_batch, t_batch, u_true):
        u_pred = vmap_predict(params, x_batch.flatten(), t_batch.flatten())
        return jnp.mean((u_pred - u_true.flatten()) ** 2)

    # 3. The Unified Gradient
    def total_loss(params, colloc_data, ic_data, bc_data, alpha=0.01):
        # Unpack the datasets
        x_colloc, t_colloc = colloc_data
        x_ic, t_ic, u_ic = ic_data
        x_bc, t_bc, u_bc = bc_data
        
        # Calculate individual loss components
        L_phy = physics_loss(params, x_colloc, t_colloc, alpha)
        L_ic = data_loss(params, x_ic, t_ic, u_ic)
        L_bc = data_loss(params, x_bc, t_bc, u_bc)
        
        # Sum all components
        return L_phy + L_ic + L_bc

    # 4. Optax Training Loop Setup
    learning_rate = 1e-3
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    print("Physics Loss, Unified Gradient, and Optax Optimizer successfully configured!")