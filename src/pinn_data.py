import jax
import jax.numpy as jnp

def generate_pinn_data(num_colloc=5000, num_ic=500, num_bc=500, seed=42):
    """
    Generates mesh-free datasets for PINN training.
    Manages JAX PRNGKeys explicitly for deterministic randomness.
    """
    # 1. Explicit PRNGKey management
    key = jax.random.PRNGKey(seed)
    # Split the main key into multiple subkeys for independent random sampling
    k_colloc_x, k_colloc_t, k_ic_x, k_bc_t, k_bc_side = jax.random.split(key, 5)
    # --- Collocation Points (PDE Interior) ---
    # 5,000 random points where physics rules apply (x in [-1, 1], t in [0, 1])
    x_colloc = jax.random.uniform(k_colloc_x, shape=(num_colloc, 1), minval=-1.0, maxval=1.0)
    t_colloc = jax.random.uniform(k_colloc_t, shape=(num_colloc, 1), minval=0.0, maxval=1.0)
    # --- Initial Condition (IC) ---
    # 500 points exactly at t = 0
    x_ic = jax.random.uniform(k_ic_x, shape=(num_ic, 1), minval=-1.0, maxval=1.0)
    t_ic = jnp.zeros_like(x_ic)
    # Starting temperature is a negative sine wave
    u_ic = -jnp.sin(jnp.pi * x_ic)
    # --- Boundary Conditions (BC) ---
    # 500 points randomly spread over time, but strictly on the edges
    t_bc = jax.random.uniform(k_bc_t, shape=(num_bc, 1), minval=0.0, maxval=1.0)
    # Randomly assign each BC point to either the left (-1) or right (1) edge
    random_sides = jax.random.choice(k_bc_side, jnp.array([-1.0, 1.0]), shape=(num_bc, 1))
    x_bc = random_sides
    # The edges are kept on ice
    u_bc = jnp.zeros_like(t_bc)
    return (x_colloc, t_colloc), (x_ic, t_ic, u_ic), (x_bc, t_bc, u_bc)

# Testing block to verify shapes if run locally
if __name__ == "__main__":
    colloc, ic, bc = generate_pinn_data()
    print("PINN mesh-free datasets generated successfully!")
    print(f"Collocation (x, t): {colloc[0].shape}, {colloc[1].shape}")
    print(f"Initial Condition (x, t, u): {ic[0].shape}, {ic[1].shape}, {ic[2].shape}")
    print(f"Boundary Condition (x, t, u): {bc[0].shape}, {bc[1].shape}, {bc[2].shape}")