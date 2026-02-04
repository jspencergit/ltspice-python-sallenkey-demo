# Run_SallenKey_LPF.py
# Enhanced: User-configurable fc/Q, E96 rounding, unequal R1/R2
# Dual simulation: AC for Bode + Transient for step response
# Logs all key info to log.txt for debugging
# Fixed: Use UTF-8 encoding for log file to handle Ω symbol

from PyLTSpice import SimCommander
import ltspice
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import sys
from scipy import signal
import math
from datetime import datetime

# Switch to interactive backend
matplotlib.use('TkAgg')

# -------------------------- User Configuration --------------------------
TARGET_FC = 50000         # Desired 3 dB cutoff frequency in Hz
TARGET_Q  = 0.707         # Desired Q factor (0.707 ≈ Butterworth)
PREFERRED_R_AVG = 10000   # Target average resistor value ~ this (Ohms) - guides scaling
STEP_DELAY = 100e-6       # Initial delay before step (seconds) - now 100 µs
STEP_RISE_TIME = 1e-9     # Rise time of step (1 ns)
STEP_AMPLITUDE = 1.0      # Step height (0 V → 1 V)
# -----------------------------------------------------------------------

# Configuration (no need to edit below)
LTSPICE_PATH = "C:/Program Files/ADI/LTspice/LTspice.exe"  # Update if needed
ASC_FILE = "SallenKey_LPF.asc"
AC_NET_FILE = "SallenKey_LPF_ac.net"      # Controls raw name: SallenKey_LPF_ac.raw
TRAN_NET_FILE = "SallenKey_LPF_tran.net"  # Controls raw name: SallenKey_LPF_tran.raw
RAW_AC_FILE = "SallenKey_LPF_ac.raw"
RAW_TRAN_FILE = "SallenKey_LPF_tran.raw"
SCHEMATIC_IMG = "SallenKey_LPF.jpg"
LOG_FILE = "log.txt"

if not os.path.exists(ASC_FILE):
    print(f"Error: {ASC_FILE} not found!")
    sys.exit(1)

# E96 series (same as before)
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
        raise ValueError("Value must be positive")
    log_val = math.log10(value)
    decade = math.floor(log_val)
    base = 10 ** (log_val - decade)
    idx = np.argmin(np.abs(E96_MULTIPLIERS - base))
    return E96_MULTIPLIERS[idx] * (10 ** decade)

def format_for_ltspice(value, is_cap=False):
    abs_val = abs(value)
    if is_cap:
        if abs_val >= 1e-6: return f"{value * 1e6:.6g}u"
        elif abs_val >= 1e-9: return f"{value * 1e9:.6g}n"
        else: return f"{value * 1e12:.6g}p"
    else:
        if abs_val >= 1e6: return f"{value / 1e6:.6g}Meg"
        elif abs_val >= 1e3: return f"{value / 1e3:.6g}k"
        else: return f"{value:.6g}"

# -------------------------- Logging Setup --------------------------
log_lines = []
log_lines.append(f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
log_lines.append("=== User Configuration ===")
log_lines.append(f"Target fc: {TARGET_FC} Hz")
log_lines.append(f"Target Q: {TARGET_Q:.3f}")
log_lines.append(f"Preferred average R: {PREFERRED_R_AVG} Ω")
log_lines.append(f"Step delay: {STEP_DELAY*1e6:.0f} µs")
log_lines.append(f"Step rise time: {STEP_RISE_TIME*1e9:.0f} ns")
log_lines.append(f"Step amplitude: {STEP_AMPLITUDE} V\n")

# -------------------------- Component Calculation --------------------------
print(f"Designing for fc = {TARGET_FC:.0f} Hz, Q = {TARGET_Q:.3f}")
log_lines.append("=== Component Calculation ===")

R_ideal = PREFERRED_R_AVG
Cground_ideal = 1 / (2 * np.pi * TARGET_FC * 2 * TARGET_Q * R_ideal)
Cfb_ideal = (2 * TARGET_Q)**2 * Cground_ideal

Cground = nearest_e96(Cground_ideal)
Cfb = nearest_e96(Cfb_ideal)

log_lines.append(f"Ideal Cground: {Cground_ideal*1e12:.3f} pF → Rounded: {Cground*1e12:.1f} pF")
log_lines.append(f"Ideal Cfb: {Cfb_ideal*1e12:.3f} pF → Rounded: {Cfb*1e12:.1f} pF")

A = 2 * np.pi * TARGET_FC
P = 1 / (A**2 * Cfb * Cground)
S = 1 / (A * TARGET_Q * Cground)

discriminant = S**2 - 4 * P
if discriminant < 0:
    print("Warning: Using equal-R fallback")
    log_lines.append("Warning: Discriminant negative → equal-R fallback")
    R1_ideal = R2_ideal = PREFERRED_R_AVG
else:
    sqrt_disc = np.sqrt(discriminant)
    R1_ideal = (S + sqrt_disc) / 2
    R2_ideal = (S - sqrt_disc) / 2

R1val = nearest_e96(R1_ideal)
R2val = nearest_e96(R2_ideal)

log_lines.append(f"R1: {R1_ideal:.0f} Ω → Rounded: {R1val:.0f} Ω")
log_lines.append(f"R2: {R2_ideal:.0f} Ω → Rounded: {R2val:.0f} Ω")

actual_fc = 1 / (2 * np.pi * np.sqrt(R1val * R2val * Cfb * Cground))
actual_Q = np.sqrt(R1val * R2val * Cfb * Cground) / ((R1val + R2val) * Cground)

log_lines.append(f"\nActual fc: {actual_fc:.1f} Hz (error: {100*(actual_fc/TARGET_FC - 1):.3f} %)")
log_lines.append(f"Actual Q: {actual_Q:.3f}\n")

R1_str = format_for_ltspice(R1val)
R2_str = format_for_ltspice(R2val)
Cground_str = format_for_ltspice(Cground, True)
Cfb_str = format_for_ltspice(Cfb, True)

# -------------------------- Simulations --------------------------
# Common parameters
params = {'R1val': R1_str, 'R2val': R2_str, 'Cground': Cground_str, 'Cfb': Cfb_str}

# 1. AC Simulation
print("Running AC simulation...")
lt_ac = SimCommander(ASC_FILE)
lt_ac.set_parameters(**params)
lt_ac.add_instruction(".ac dec 200 10 1Meg")
lt_ac.run(run_filename=AC_NET_FILE)
if not lt_ac.wait_completion(timeout=60):
    print("AC simulation timed out!")
    sys.exit(1)

l_ac = ltspice.Ltspice(RAW_AC_FILE)
l_ac.parse()
freq = l_ac.get_frequency()
vout_ac = l_ac.get_data('V(vout)')
mag_sim = 20 * np.log10(np.abs(vout_ac))
phase_sim = np.angle(vout_ac, deg=True)

# Ideal Bode
sys_ideal = signal.TransferFunction([(2*np.pi*TARGET_FC)**2], 
                                   [1, (2*np.pi*TARGET_FC)/TARGET_Q, (2*np.pi*TARGET_FC)**2])
w = 2 * np.pi * freq
_, mag_ideal, phase_ideal = signal.bode(sys_ideal, w=w)

# 2. Transient Simulation
print("Running Transient simulation...")
t_max = STEP_DELAY + max(20e-6, 10 / TARGET_FC)  # At least 20 µs after step, or ~10 time constants
max_step = min(1e-9, (t_max - STEP_DELAY)/2000)   # Reasonable resolution

pwl_str = f"PWL(0 0 {STEP_DELAY} 0 {STEP_DELAY + STEP_RISE_TIME} {STEP_AMPLITUDE} {t_max} {STEP_AMPLITUDE})"

lt_tran = SimCommander(ASC_FILE)
lt_tran.set_parameters(**params)
lt_tran.set_element_model('V3', pwl_str)
lt_tran.add_instruction(f".tran 0 {t_max} 0 {max_step} startup")
lt_tran.run(run_filename=TRAN_NET_FILE)
if not lt_tran.wait_completion(timeout=60):
    print("Transient simulation timed out!")
    sys.exit(1)

l_tran = ltspice.Ltspice(RAW_TRAN_FILE)
l_tran.parse()
time = l_tran.get_time()
vin_tran = l_tran.get_data('V(vin)')
vout_tran = l_tran.get_data('V(vout)')

log_lines.append("=== Transient Simulation ===")
log_lines.append(f"Total sim time: {t_max*1e6:.1f} µs")
log_lines.append(f"Max timestep: {max_step*1e9:.2f} ns")
log_lines.append(f"Step: 0→{STEP_AMPLITUDE} V at t={STEP_DELAY*1e6:.0f} µs")

# -------------------------- Logging --------------------------
with open(LOG_FILE, 'w', encoding='utf-8') as f:  # Fixed: UTF-8 encoding
    f.write('\n'.join(log_lines))
print(f"Log saved to {LOG_FILE}")

# -------------------------- Plotting --------------------------
fig = plt.figure(figsize=(12, 14))
gs = fig.add_gridspec(3, 1, height_ratios=[3, 2, 1])

# Bode plot (mag + phase on twin axes)
ax_bode = fig.add_subplot(gs[0, 0])
ax_phase = ax_bode.twinx()

ax_bode.semilogx(freq, mag_sim, label='Simulated Mag', color='blue')
ax_bode.semilogx(freq, mag_ideal, '--', label='Ideal Mag', color='red')
ax_phase.semilogx(freq, phase_sim, label='Simulated Phase', color='purple')
ax_phase.semilogx(freq, phase_ideal, '--', label='Ideal Phase', color='orange')

ax_bode.grid(True, which="both", ls="--")
ax_bode.set_ylabel('Magnitude (dB)')
ax_phase.set_ylabel('Phase (deg)')
ax_bode.set_xlabel('Frequency (Hz)')
ax_bode.set_title(f'Sallen-Key Bode: Target fc={TARGET_FC/1000:.1f} kHz, Q={TARGET_Q:.3f}')
ax_bode.axvline(TARGET_FC, color='green', linestyle='--', label='Target fc')
ax_bode.axvline(actual_fc, color='orange', linestyle=':', label='Actual fc')

lines1, labels1 = ax_bode.get_legend_handles_labels()
lines2, labels2 = ax_phase.get_legend_handles_labels()
ax_bode.legend(lines1 + lines2, labels1 + labels2, loc='lower left')

# Step response
ax_step = fig.add_subplot(gs[1, 0])
ax_step.plot(time*1e6, vin_tran, label='Vin (step)', color='gray', linestyle='--')
ax_step.plot(time*1e6, vout_tran, label='Vout', color='blue')
ax_step.grid(True)
ax_step.set_xlabel('Time (µs)')
ax_step.set_ylabel('Voltage (V)')
ax_step.set_title('Step Response')
ax_step.legend()

# Schematic
ax_sch = fig.add_subplot(gs[2, 0])
if os.path.exists(SCHEMATIC_IMG):
    ax_sch.imshow(plt.imread(SCHEMATIC_IMG))
ax_sch.axis('off')
ax_sch.set_title('Schematic')

# Interactive title updates on hover
def on_move(event):
    if event.inaxes == ax_bode or event.inaxes == ax_phase:
        x = event.xdata
        if x is None: return
        idx = np.argmin(np.abs(freq - x))
        ax_bode.set_title(f'Bode @ {freq[idx]:.1f} Hz: Mag {mag_sim[idx]:.2f} dB, Phase {phase_sim[idx]:.1f}°')
    elif event.inaxes == ax_step:
        x = event.xdata
        if x is None: return
        idx = np.argmin(np.abs(time - x*1e-6))
        ax_step.set_title(f'Step @ {time[idx]*1e6:.3f} µs: Vin {vin_tran[idx]:.3f} V, Vout {vout_tran[idx]:.3f} V')
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', on_move)

plt.tight_layout()
plt.show()