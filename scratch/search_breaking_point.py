import jax
import jax.numpy as jnp
import numpy as np

def simulate_path(key, sigma):
    key_D, key_C, key_R = jax.random.split(key, 3)
    D = 1000.0 + 150.0 * jax.random.normal(key_D)
    C = jnp.exp(5.5 + sigma * jax.random.normal(key_C))
    R = jax.random.uniform(key_R, minval=0.05, maxval=0.25)
    revenue = (D * 150.0) - C * (1.0 - R)
    return revenue

def get_var_95(sigma):
    master_key = jax.random.PRNGKey(42)
    N = 1_000_000
    subkeys = jax.random.split(master_key, N)
    run_sim = jax.jit(jax.vmap(simulate_path, in_axes=(0, None)))
    revenues = run_sim(subkeys, sigma)
    # block_until_ready to make sure JAX computation is finished
    revenues.block_until_ready()
    expected_revenue = jnp.mean(revenues)
    var_95 = jnp.percentile(revenues, 5)
    return float(expected_revenue), float(var_95)

def main():
    print("Starting binary search for breaking point...")
    
    # Let's first verify baseline at sigma = 0.3
    exp_rev, var_95 = get_var_95(0.3)
    print(f"Baseline sigma = 0.3: Expected Revenue = ${exp_rev:,.2f}, VaR 95% = ${var_95:,.2f}")
    
    # We want to find sigma where VaR 95% == 0
    # Let's search between low = 0.3 and high = 2.0 (or check if high needs to be larger)
    low = 0.3
    high = 5.0
    
    # Verify that at high, VaR 95% is negative
    _, var_high = get_var_95(high)
    print(f"High sigma = {high}: VaR 95% = ${var_high:,.2f}")
    if var_high > 0:
        print("Error: VaR 95% at high is still positive. Need to increase high.")
        return
        
    tolerance = 1e-7
    steps = 0
    while high - low > tolerance:
        mid = (low + high) / 2.0
        exp_rev, var_mid = get_var_95(mid)
        print(f"Step {steps:02d}: sigma = {mid:.8f}, Expected Revenue = ${exp_rev:,.2f}, VaR 95% = ${var_mid:,.4f}")
        
        # Since VaR 95% decreases as sigma increases:
        # If var_mid > 0, we need to increase sigma to drop VaR, so low = mid
        # If var_mid < 0, we need to decrease sigma to raise VaR, so high = mid
        if var_mid > 0:
            low = mid
        else:
            high = mid
        steps += 1

    critical_sigma = (low + high) / 2.0
    final_exp_rev, final_var_95 = get_var_95(critical_sigma)
    critical_variance = critical_sigma ** 2
    
    print("\n--- SEARCH COMPLETED ---")
    print(f"Critical standard deviation (sigma): {critical_sigma:.6f}")
    print(f"Critical variance (sigma^2): {critical_variance:.6f}")
    print(f"Expected Revenue at breaking point: ${final_exp_rev:,.2f}")
    print(f"VaR 95% at breaking point: ${final_var_95:,.6f}")

if __name__ == "__main__":
    main()
