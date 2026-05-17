import jax
import jax.numpy as jnp

# 1. The Differentiable Target
def projectile_loss(v_initial):
    """
    Pure JAX function that calculates the Mean Squared Error (MSE) between
    the simulated projectile distance after 5 seconds and a target of 150.0 meters.
    """
    flight_time = 5.0
    target_distance = 150.0
    # Simulate final distance covered using linear kinematics
    simulated_distance = v_initial * flight_time
    # Compute the Mean Squared Error (MSE) loss
    loss = jnp.mean((simulated_distance - target_distance) ** 2)
    return loss

# 2. Pulling the Derivative
grad_projectile_loss = jax.grad(projectile_loss)

# 3. The Optimization Loop & 4. Deliverables
if __name__ == "__main__":
    # Start with a random guess as instructed by the oracle
    v_initial = 10.0
    learning_rate = 0.01
    iterations = 20
    print(f"Starting gradient descent optimization from v_initial = {v_initial}")
    print("-" * 65)
    for i in range(iterations):
        current_loss = projectile_loss(v_initial)
        # Calculate the exact analytical gradient using jax.grad
        gradient = grad_projectile_loss(v_initial)
        # Parameter update formula: v = v - alpha * grad
        v_initial = v_initial - learning_rate * gradient
        print(f"Iteration {i+1:02d}: v_initial = {v_initial:.4f} | Loss = {current_loss:.4f} | Gradient = {gradient:.4f}")
    print("-" * 65)
    print(f"Final Optimized v_initial: {v_initial:.4f}")