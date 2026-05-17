import jax
import jax.numpy as jnp

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