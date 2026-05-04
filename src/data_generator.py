import numpy as np
import matplotlib.pyplot as plt
import os

# --- 0. STUDENT PARAMETERS ---
# Matrikelnummer: 5015970
T = 70                  # Last 2 digits of ID
C_microF = 1970         # 1000 + Last 3 digits of ID (970)

# --- 1. The Fourier Datastream ---
omega_0 = 2 * np.pi / T

# Time setup: 100 periods, 1000 points per period for smooth resolution
periods = 100
points_per_period = 1000
t = np.linspace(0, periods * T, periods * points_per_period)

# Generate continuous Fourier series of a square wave (first 9 odd harmonics)
harmonics = [1, 3, 5, 7, 9, 11, 13, 15, 17]
original_signal = np.zeros_like(t)

for n in harmonics:
    amplitude = 4 / (np.pi * n)
    original_signal += amplitude * np.sin(n * omega_0 * t)

# --- 2. The Numerical RC Filter ---
R = 500                    # 0.5 kOhm = 500 Ohm
C = C_microF * 1e-6        # Convert microFarads to Farads

filtered_signal = np.zeros_like(t)

for n in harmonics:
    omega_n = n * omega_0
    
    # Calculate analytical effect of RC low-pass filter: H(w) = 1 / (1 + j*w*R*C)
    H_magnitude = 1 / np.sqrt(1 + (omega_n * R * C)**2)
    H_phase = -np.arctan(omega_n * R * C)
    
    # Apply filter to each harmonic
    amplitude = 4 / (np.pi * n)
    filtered_signal += amplitude * H_magnitude * np.sin(n * omega_0 * t + H_phase)

# --- 3. Noise & Sabotage (Anomaly Injection) ---
# Add random Gaussian noise
noise_std = 0.1
noise = np.random.normal(0, noise_std, size=t.shape)
noisy_signal = filtered_signal + noise

# Inject a massive high-frequency voltage spike between period 70 and 75
anomaly_mask = (t >= 70 * T) & (t <= 75 * T)
high_freq = 50 * omega_0
spike = 5.0 * np.sin(high_freq * t[anomaly_mask])

final_signal = np.copy(noisy_signal)
final_signal[anomaly_mask] += spike

# Save 1D continuous array locally inside /data folder
os.makedirs('data', exist_ok=True)
np.save('data/datastream.npy', final_signal)
print("✅ Data successfully generated and saved to 'data/datastream.npy'")

# --- 4. Plotting & Deliverables ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot normal noisy signal (Periods 10 to 15)
normal_mask = (t >= 10 * T) & (t <= 15 * T)
ax1.plot(t[normal_mask], final_signal[normal_mask], color='blue')
ax1.set_title(f'Normal Noisy Signal (Periods 10-15, T={T})')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Voltage (V)')
ax1.grid(True)

# Plot anomaly spike (Periods 68 to 77 for contrast context)
spike_window_mask = (t >= 68 * T) & (t <= 77 * T)
ax2.plot(t[spike_window_mask], final_signal[spike_window_mask], color='red')
ax2.set_title('Anomaly Spike (Periods 70-75)')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Voltage (V)')
ax2.grid(True)

plt.tight_layout()
plt.savefig('data_feed.png')
print("✅ Plot successfully saved as 'data_feed.png'")