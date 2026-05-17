import jax
import jax.numpy as jnp
import time

def oscillator_step(x, v, w):
    """
    Pure function to calculate the new state of a SINGLE oscillator 
    for exactly one time step.
    """
    dt = 0.01
    damping = 0.1
    # Calculate acceleration for a single pendulum
    # a = -omega^2 * x - c * v
    a = -(w**2) * x - damping * v
    # Update position and velocity (Explicit Euler)
    x_new = x + v * dt
    v_new = v + a * dt
    # Return the new state without modifying any external variables (Functional Purity)
    return x_new, v_new

vmap_step = jax.vmap(oscillator_step)

# Hardware Fusion (jit)
@jax.jit
def simulate_jax(x, v, w, steps):
    """
    Executes the full simulation loop compiled via XLA.
    Uses jax.lax.fori_loop to prevent loop unrolling and speed up compilation.
    """
    def body_fun(i, state):
        x_curr, v_curr = state
        # Call the vectorized batch-processing engine
        x_next, v_next = vmap_step(x_curr, v_curr, w)
        return x_next, v_next
    # Execute the loop from 0 to 'steps', passing the initial state (x, v)
    final_x, final_v = jax.lax.fori_loop(0, steps, body_fun, (x, v))
    return final_x, final_v

if __name__ == "__main__":
    N = 100000
    steps = 1000
    # Initialization of JAX arrays (Using PRNGKey for pure random generation)
    key = jax.random.PRNGKey(42)
    w = jax.random.uniform(key, shape=(N,), minval=0.5, maxval=2.0)
    x = jnp.ones(N)
    v = jnp.zeros(N)
    
    print("Run 1: JIT Compilation (Warm-up / Tracing phase)...")
    start_run1 = time.time()
    res_x, res_v = simulate_jax(x, v, w, steps)
    res_x.block_until_ready() 
    end_run1 = time.time()
    print(f"JAX Run 1 Time (Compilation + Execution): {end_run1 - start_run1:.4f} seconds")
    
    print("\nRun 2: Compiled Execution...")
    start_run2 = time.time()
    res_x2, res_v2 = simulate_jax(x, v, w, steps)
    res_x2.block_until_ready()
    end_run2 = time.time()
    jax_final_time = end_run2 - start_run2
    print(f"JAX Run 2 Time (Pure Execution): {jax_final_time:.6f} seconds")