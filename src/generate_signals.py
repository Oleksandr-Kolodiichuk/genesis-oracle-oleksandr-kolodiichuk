import os
import numpy as np
import matplotlib.pyplot as plt
import random

def main():
    # Ensure the target directory exists
    os.makedirs("data", exist_ok=True)

    # 1. Generate a smooth dynamic wave signal (baseline)
    t = np.linspace(0, 10, 1000) # 10 seconds, 1000 data points
    signal = np.sin(2 * np.pi * 0.5 * t) + 0.3 * np.sin(2 * np.pi * 2 * t)

    # 2. Introduce the secret malfunction (high-frequency clipping artifact)
    # Pick a random starting point for the anomaly
    start_idx = random.randint(100, 850)
    end_idx = start_idx + 80 # The glitch lasts for 80 data points
    
    # Inject heavy noise
    noise = np.random.normal(0, 2.0, end_idx - start_idx)
    signal[start_idx:end_idx] += noise
    
    # Simulate amplitude saturation (clipping the extremes)
    signal = np.clip(signal, -1.2, 1.2)

    # 3. Plot the signal
    plt.figure(figsize=(10, 4))
    plt.plot(t, signal, color='blue', linewidth=1.2)
    plt.title("Telemetry Data: Dynamic Wave Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle='--', alpha=0.6)

    # 4. Save the output plot silently (no terminal print of the timestamp)
    output_path = "data/audit_target.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()