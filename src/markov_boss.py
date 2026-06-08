import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os

def main():
    print("Initiating Module Alpha: The Matrix Carrier...")
    
    # Baseline Transition Matrix (P)
    # State 0: Bull Market, State 1: Stagnation, State 2: Catastrophic Recession
    P_base = jnp.array([
        [0.85, 0.12, 0.03],
        [0.10, 0.75, 0.15],
        [0.05, 0.20, 0.75]
    ])
    
    # The Sabotage Matrix (Black Swan Shock)
    # Shift 80% of the transition probability mass from State 0 and 1 directly into State 2.
    # We multiply the first two columns by 0.2 and add 0.8 to the third column.
    P_shock = jnp.array([
        [0.17, 0.024, 0.806],  # 0.85*0.2, 0.12*0.2, 0.03+0.80
        [0.02, 0.150, 0.830],  # 0.10*0.2, 0.75*0.2, 0.15+0.80
        [0.05, 0.200, 0.750]   # Row 2 remains unchanged
    ])
    
    # Define the pure functional step for jax.lax.scan
    def transition_step(state_vector, day):
        # Determine if we are in the 10-day Black Swan crisis window
        is_shock = jnp.logical_and(day >= 180, day < 190)
        
        # Dynamically select the matrix using JIT-compatible jnp.where
        P_active = jnp.where(is_shock, P_shock, P_base)
        
        # Calculate the macro-economic state for the next day
        next_state = jnp.dot(state_vector, P_active)
        
        # Return the state to carry forward, and the output to save in history
        return next_state, state_vector
        
    # Initial condition: Assume we start in a 100% Bull Market at Day 0
    initial_state = jnp.array([1.0, 0.0, 0.0])
    timeline_days = jnp.arange(365)
    
    # Execute the highly optimized functional loop
    final_state, state_history = jax.lax.scan(transition_step, initial_state, timeline_days)
    
    print("Simulation complete. Generating historical timeline...")
    
    # Plotting the percentage distribution
    plt.figure(figsize=(12, 6))
    
    # Multiply by 100 to convert probabilities to percentages
    bull_history = state_history[:, 0] * 100
    stag_history = state_history[:, 1] * 100
    rec_history = state_history[:, 2] * 100
    
    plt.plot(timeline_days, bull_history, label='State 0: Bull Market', color='forestgreen', linewidth=2.5)
    plt.plot(timeline_days, stag_history, label='State 1: Stagnation', color='goldenrod', linewidth=2.5)
    plt.plot(timeline_days, rec_history, label='State 2: Catastrophic Recession', color='firebrick', linewidth=2.5)
    
    # Highlight the anomaly zone
    plt.axvspan(180, 190, color='black', alpha=0.15, label='Black Swan Sabotage (Days 180-190)')
    
    plt.title('Macro-Economic Markov Chain: The Black Swan Event', fontsize=15, fontweight='bold')
    plt.xlabel('Simulated Days', fontsize=12)
    plt.ylabel('Probability Mass (%)', fontsize=12)
    plt.xlim(0, 365)
    plt.ylim(0, 100)
    plt.legend(loc='center right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the output image
    os.makedirs('data', exist_ok=True)
    save_path = 'data/markov_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"Graph successfully saved to {save_path}")

if __name__ == "__main__":
    main()