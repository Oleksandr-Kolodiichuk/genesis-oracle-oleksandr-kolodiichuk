import numpy as np
import time
import matplotlib.pyplot as plt
import os

def main():
    # 1. & 2. Generate 5000000 random (x, y) points
    N = 5_000_000
    print(f"Generating {N} random points using classical NumPy...")
    
    # Start the timer
    start_time = time.perf_counter()
    
    # Uniformly distribute points between 0 and 1
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)
    
    # 3. Compute Euclidean distance squared from the origin
    distances_squared = x**2 + y**2
    
    # Count how many points fall inside the unit circle boundary (distance <= 1)
    inside_circle = distances_squared <= 1.0
    N_inside = np.sum(inside_circle)
    
    # 4. Calculate empirical estimation of Pi
    pi_estimate = 4 * N_inside / N
    
    # Stop the timer
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    
    print(f"Empirical estimation of Pi: {pi_estimate}")
    print(f"Execution time: {execution_time:.4f} seconds")
    
    # 5. Extract a random subset of 10000 points for visualization
    print("Generating scatter plot for a 10,000 point subset...")
    subset_size = 10000
    
    # Randomly select 10,000 indices
    subset_indices = np.random.choice(N, subset_size, replace=False)
    
    x_subset = x[subset_indices]
    y_subset = y[subset_indices]
    inside_subset = inside_circle[subset_indices]
    
    # Setup the plot
    plt.figure(figsize=(8, 8))
    
    # Plot points: Blue for inside, Red for outside
    plt.scatter(x_subset[inside_subset], y_subset[inside_subset], color='blue', s=2, label='Inside Circle', alpha=0.6)
    plt.scatter(x_subset[~inside_subset], y_subset[~inside_subset], color='red', s=2, label='Outside Circle', alpha=0.6)
    
    # Draw the quarter-circle boundary line
    theta = np.linspace(0, np.pi/2, 100)
    plt.plot(np.cos(theta), np.sin(theta), color='black', linewidth=2, label='Boundary Line')
    
    # Formatting
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f'Monte Carlo Pi Estimation (Subset of {subset_size} points)\nEstimated Pi: {pi_estimate:.5f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Create data directory if it doesn't exist and save the plot
    os.makedirs('data', exist_ok=True)
    save_path = 'data/classical_pi_disp.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to {save_path}")

if __name__ == "__main__":
    main()