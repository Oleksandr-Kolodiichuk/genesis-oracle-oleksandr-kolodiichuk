import numpy as np
import time

def simulate_legacy_swarm():
    # Simulation parameters
    N = 100000        # Number of independent oscillators
    steps = 1000      # Number of discrete time steps
    dt = 0.01         # Time step size
    damping = 0.1     # Damping coefficient

    # 1. Array Initialization
    # Set initial positions to 1.0 and velocities to 0.0
    x = np.ones(N)    
    v = np.zeros(N)   
    
    # Initialize an array of 100,000 random natural frequencies (omega)
    # Frequencies are uniformly distributed between 0.5 and 2.0
    omega = np.random.uniform(0.5, 2.0, N)

    print(f"Starting sequential simulation of {N} pendulums for {steps} steps...")
    
    # 2. Start the stopwatch
    start_time = time.time()

    # 3. Sequential loop iterating over the discrete time steps
    for _ in range(steps):
        # Calculate acceleration for the entire array
        # Equation: a = -omega^2 * x - c * v
        a = -(omega**2) * x - damping * v
        
        # Update positions and velocities using Explicit Euler integration
        x = x + v * dt
        v = v + a * dt

    # 4. Stop the stopwatch
    end_time = time.time()

    # Calculate and print total execution time
    execution_time = end_time - start_time
    print("Simulation finished.")
    print(f"Total execution time (Legacy NumPy): {execution_time:.4f} seconds")

if __name__ == "__main__":
    simulate_legacy_swarm()