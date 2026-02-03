# Run_SallenKey_LPF.py
# Enhanced version: User-configurable cutoff frequency and Q
# Automatically calculates and rounds components to nearest E96 values
# Allows R1 ≠ R2 for better accuracy after rounding
# Single nominal simulation (Monte-Carlo later)

from PyLTSpice import SimCommander
import ltspice
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import sys
from scipy import signal
import math

# Switch to interactive backend
matplotlib.use('TkAgg')

# -------------------------- User Configuration --------------------------
TARGET_FC = 50000         # Desired 3 dB cutoff frequency in Hz
TARGET_Q  = 2.0         # Desired Q factor (0.707 ≈ Butterworth)
PREFERRED_R_AVG = 10000   # Target average resistor value ~ this (Ohms) - guides scaling
# -----------------------------------------------------------------------

# Configuration (no need to edit below unless changing paths)
LTSPICE_PATH = "C:/Program Files/ADI/LTspice/LTspice.exe"  # Update if needed
ASC_FILE = "SallenKey_LPF.asc"
RAW_FILE = "SallenKey_LPF_1.raw"
SCHEMATIC_IMG = "SallenKey_LPF.jpg"

if not os.path.exists(ASC_FILE):
    print(f"Error: {ASC_FILE} not found!")
    sys.exit(1)

# E96 series multipliers (standard 1% values)
E96_MULTIPLIERS = np.array([
    1.00, 1.02, 1.05, 1.07, 1.10, 1.13, 1.15, 1.18, 1.21, 1.24,
    1.27, 1.30, 1.33, 1.37, 1.40, 1.43, 1.47, 1.50, 1.54, 1.58,
    1.62, 1.65, 1.69, 1.74, 1.78, 1.82, 1.87, 1.91, 1.96, 2.00,
    2.05, 2.10, 2.15, 2.21, 2.26, 2.32, 2.37, 2.43, 2.49, 2.55,
    2.61, 2.67, 2.74, 2.80, 2.87, 2.94, 3.01, 3.09, 3.16, 3.24,
    3.32, 3.40, 3.48, 3.57, 3.65, 3.74, 3.83, 3.92, 4.02, 4.12,
    4.22, 4.32, 4.42, 4.53, 4.64, 4.75, 4.87, 4.99, 5.11, 5.23,
    5.36, 5.49, 5.62, 5.76, 5.90, 6.04, 6.19, 6.34, 6.49, 6.65,
    6.81, 6.98, 7.15, 7.32, 7.50, 7.68, 7.87, 8.06, 8.25, 8.45,
    8.66, 8.87, 9.09, 9.31, 9.53, 9.76
])

def nearest_e96(value):
    if value <= 0:
        raise ValueError("Value must be positive for E96 rounding")
    log_val = math.log10(value)
    decade = math.floor(log_val)
    base = 10 ** (log_val - decade)
    idx = np.argmin(np.abs(E96_MULTIPLIERS - base))
    closest_base = E96_MULTIPLIERS[idx]
    return closest_base * (10 ** decade)

def format_for_ltspice(value, is_cap=False):
    """Format number for LTSpice .param (e.g., 10000 → '10k', 220e-12 → '220p')"""
    abs_val = abs(value)
    if is_cap:
        if abs_val >= 1e-6:
            return f"{value * 1e6:.6g}u"
        elif abs_val >= 1e-9:
            return f"{value * 1e9:.6g}n"
        else:
            return f"{value * 1e12:.6g}p"
    else:  # resistor
        if abs_val >= 1e6:
            return f"{value / 1e6:.6g}Meg"
        elif abs_val >= 1e3:
            return f"{value / 1e3:.6g}k"
        else:
            return f"{value:.6g}"

# -------------------------- Component Calculation --------------------------
print(f"Designing for fc = {TARGET_FC:.0f} Hz, Q = {TARGET_Q:.3f}")

# Step 1: Start with equal-R assumption to get ideal caps scaled to preferred R
R_ideal = PREFERRED_R_AVG
Cground_ideal = 1 / (2 * np.pi * TARGET_FC * 2 * TARGET_Q * R_ideal)
Cfb_ideal = (2 * TARGET_Q)**2 * Cground_ideal

# Step 2: Round caps to nearest E96
Cground = nearest_e96(Cground_ideal)
Cfb = nearest_e96(Cfb_ideal)

print(f"Rounded caps: Cground = {Cground*1e12:.0f} pF, Cfb = {Cfb*1e12:.0f} pF")

# Step 3: Solve for exact R1, R2 to hit target fc & Q with these rounded caps
A = 2 * np.pi * TARGET_FC
P = 1 / (A**2 * Cfb * Cground)               # R1 * R2
S = 1 / (A * TARGET_Q * Cground)             # R1 + R2

discriminant = S**2 - 4 * P
if discriminant < 0:
    print("Warning: Cannot achieve exact Q with these caps - falling back to equal R approximation")
    R1_ideal = R2_ideal = PREFERRED_R_AVG
else:
    sqrt_disc = np.sqrt(discriminant)
    R1_ideal = (S + sqrt_disc) / 2
    R2_ideal = (S - sqrt_disc) / 2
    if R2_ideal < 100 or R1_ideal > 1e6:  # sanity check
        print("R values out of reasonable range - using equal R fallback")
        R1_ideal = R2_ideal = PREFERRED_R_AVG

# Step 4: Round resistors to nearest E96
R1val = nearest_e96(R1_ideal)
R2val = nearest_e96(R2_ideal)

print(f"Using: R1 = {R1val:.0f} Ω, R2 = {R2val:.0f} Ω")

# Format for LTSpice
R1_str = format_for_ltspice(R1val, is_cap=False)
R2_str = format_for_ltspice(R2val, is_cap=False)
Cground_str = format_for_ltspice(Cground, is_cap=True)
Cfb_str = format_for_ltspice(Cfb, is_cap=True)

# -------------------------- Run Simulation --------------------------
print("Starting LTSpice simulation...")
lt = SimCommander(ASC_FILE)

# Set parameters
lt.set_parameters(R1val=R1_str, R2val=R2_str, Cground=Cground_str, Cfb=Cfb_str)

# Optional: nicer AC resolution
lt.add_instruction(".ac dec 200 10 1Meg")

lt.run()
if not lt.wait_completion(timeout=60):
    print("Simulation timed out!")
    sys.exit(1)

print("Simulation complete.")

# -------------------------- Post-Processing --------------------------
l = ltspice.Ltspice(RAW_FILE)
l.parse()

freq = l.get_frequency()
vout = l.get_data('V(vout)')

mag_sim = 20 * np.log10(np.abs(vout))
phase_sim = np.angle(vout, deg=True)

# Actual fc and Q from used component values
actual_fc = 1 / (2 * np.pi * np.sqrt(R1val * R2val * Cfb * Cground))
actual_Q = np.sqrt(R1val * R2val * Cfb * Cground) / ((R1val + R2val) * Cground)

print(f"Actual (from components): fc = {actual_fc:.1f} Hz, Q = {actual_Q:.3f}")

# Ideal analytical (target)
sys_ideal = signal.TransferFunction([ (2*np.pi*TARGET_FC)**2 ], 
                                   [1, (2*np.pi*TARGET_FC)/TARGET_Q, (2*np.pi*TARGET_FC)**2])
w = 2 * np.pi * freq
_, mag_ideal, phase_ideal = signal.bode(sys_ideal, w=w)

# -------------------------- Plotting --------------------------
fig = plt.figure(figsize=(12, 12))
gs = fig.add_gridspec(3, 1, height_ratios=[2, 2, 1])

ax1 = fig.add_subplot(gs[0, 0])
ax1.semilogx(freq, mag_sim, label='Simulated', color='blue')
ax1.semilogx(freq, mag_ideal, '--', label=f'Ideal (target fc={TARGET_FC/1000:.1f} kHz, Q={TARGET_Q:.3f})', color='red')
ax1.grid(True, which="both", ls="--")
ax1.set_ylabel('Magnitude (dB)')
ax1.set_title(f'Sallen-Key Low-Pass: Target fc={TARGET_FC/1000:.1f} kHz, Q={TARGET_Q:.3f}')
ax1.axvline(x=TARGET_FC, color='green', linestyle='--', label=f'Target -3 dB: {TARGET_FC/1000:.1f} kHz')
ax1.axvline(x=actual_fc, color='orange', linestyle=':', alpha=0.7, label=f'Actual -3 dB: {actual_fc/1000:.1f} kHz')
ax1.axhline(y=-3, color='gray', linestyle=':', alpha=0.7)
ax1.legend(loc='lower left')

ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax2.semilogx(freq, phase_sim, label='Simulated', color='purple')
ax2.semilogx(freq, phase_ideal, '--', label='Ideal (target)', color='orange')
ax2.grid(True, which="both", ls="--")
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Phase (deg)')
ax2.legend(loc='lower left')

ax3 = fig.add_subplot(gs[2, 0])
if os.path.exists(SCHEMATIC_IMG):
    ax3.imshow(plt.imread(SCHEMATIC_IMG))
ax3.axis('off')
ax3.set_title('Sallen-Key Schematic')

# Interactive cursor (same as before)
mag_dot, = ax1.plot([], [], 'ko', markersize=5)
mag_text = ax1.text(0.05, 0.95, '', transform=ax1.transAxes, ha='left', va='top',
                    bbox=dict(boxstyle="round", facecolor="wheat"))

phase_dot, = ax2.plot([], [], 'ko', markersize=5)
phase_text = ax2.text(0.05, 0.95, '', transform=ax2.transAxes, ha='left', va='top',
                      bbox=dict(boxstyle="round", facecolor="wheat"))

def on_move(event):
    if event.inaxes == ax1:
        x = event.xdata
        if x is None: return
        idx = np.argmin(np.abs(freq - x))
        mag_dot.set_data([freq[idx]], [mag_sim[idx]])
        mag_text.set_text(f'Freq: {freq[idx]:.2f} Hz\nMag: {mag_sim[idx]:.2f} dB')
    elif event.inaxes == ax2:
        x = event.xdata
        if x is None: return
        idx = np.argmin(np.abs(freq - x))
        phase_dot.set_data([freq[idx]], [phase_sim[idx]])
        phase_text.set_text(f'Freq: {freq[idx]:.2f} Hz\nPhase: {phase_sim[idx]:.2f}°')
    else:
        mag_dot.set_data([], [])
        mag_text.set_text('')
        phase_dot.set_data([], [])
        phase_text.set_text('')
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', on_move)

plt.tight_layout()
plt.show()