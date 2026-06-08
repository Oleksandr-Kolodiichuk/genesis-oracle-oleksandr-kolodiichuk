import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import time

# 2. Write a mathematically pure JAX function
def simulate_path(key):
    # Split the key to isolate three distinct random samples
    key_D, key_C, key_R = jax.random.split(key, 3)
    
    # Market Demand (D): Normal distribution N(1000, 150^2)
    D = 1000.0 + 150.0 * jax.random.normal(key_D)
    
    # Production Asset Cost (C): Log-Normal distribution ln(C) ~ N(5.5, 0.3^2)
    # jax.random.normal gives standard normal, we scale and exponentiate
    C = jnp.exp(5.5 + 0.3 * jax.random.normal(key_C))
    
    # Regulatory Penalty Rate (R): Uniform distribution U(0.05, 0.25)
    R = jax.random.uniform(key_R, minval=0.05, maxval=0.25)
    
    # The deterministic economic engine
    revenue = (D * 150.0) - C * (1.0 - R)
    return revenue

def main():
    print("Initializing JAX Monte Carlo Engine...")
    start_time = time.perf_counter()

    # 3. Initialize master key
    master_key = jax.random.PRNGKey(42)
    
    # 4. Generate 1,000,000 unique subkeys (No Python loop)
    N = 1_000_000
    subkeys = jax.random.split(master_key, N)
    
    # 5. Apply jax.vmap to evaluate all keys in parallel
    print(f"Executing {N} parallel stochastic paths via jax.vmap...")
    vmap_simulate = jax.vmap(simulate_path)
    
    # Execute the simulation
    revenues = vmap_simulate(subkeys)
    
    # Force computation to finish before stopping timer (JAX uses async dispatch)
    revenues.block_until_ready()
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    
    # 6. Calculate Expected Value and Value-at-Risk (VaR_95%)
    expected_revenue = jnp.mean(revenues)
    var_95 = jnp.percentile(revenues, 5) # 5th percentile for 95% Confidence VaR
    
    print(f"Simulation completed in {execution_time:.4f} seconds")
    print(f"Expected Revenue E[R]: ${expected_revenue:,.2f}")
    print(f"Value-at-Risk (VaR 95%): ${var_95:,.2f}")
    
    # Deliverable: Generate a clean distribution histogram
    print("Generating revenue distribution histogram...")
    plt.figure(figsize=(10, 6))
    
    # Convert to standard numpy for matplotlib
    rev_np = np.array(revenues) if 'np' in globals() else revenues
    
    plt.hist(rev_np, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    
    # Draw vertical lines for Expected Value and VaR
    plt.axvline(expected_revenue, color='black', linestyle='solid', linewidth=2, 
                label=f'Expected Revenue: ${expected_revenue:,.0f}')
    plt.axvline(var_95, color='red', linestyle='dashed', linewidth=2, 
                label=f'VaR (95%): ${var_95:,.0f}')
    
    plt.title('Monte Carlo Business Simulation: Annual Net Revenue Distribution', fontsize=14)
    plt.xlabel('Net Revenue ($)', fontsize=12)
    plt.ylabel('Frequency (Number of Paths)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Save the resulting graphic
    os.makedirs('data', exist_ok=True)
    save_path = 'data/revenue_dist.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Histogram successfully saved to {save_path}")

if __name__ == "__main__":
    main()