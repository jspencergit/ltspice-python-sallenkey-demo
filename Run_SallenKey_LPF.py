# Run_SallenKey_LPF.py
# Enhanced: User-configurable fc/Q, E96 rounding, unequal R1/R2
# Dual simulation: AC for Bode + Transient for step response
# Added: Step response analysis - % overshoot, rise time (10-90%), settling time (±2%)
# Annotations on step plot + metrics text box with improved placement to avoid overlap

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
TARGET_FC = 1000000         # Desired 3 dB cutoff frequency in Hz
TARGET_Q  = 0.707         # Desired Q factor (0.707 ≈ Butterworth)
PREFERRED_R_AVG = 10000   # Target average resistor value ~ this (Ohms) - guides scaling
STEP_DELAY = 100e-6       # Initial delay before step (seconds) - 100 µs
STEP_RISE_TIME = 1e-9     # Rise time of step (1 ns)
STEP_AMPLITUDE = 1.0      # Step height (0 V → 1 V)
SETTLING_TOLERANCE = 0.02 # Settling band ±2% (change to 0.01 for ±1%, etc.)
# -----------------------------------------------------------------------

# Configuration (no need to edit below)
ASC_FILE = "SallenKey_LPF.asc"
AC_NET_FILE = "SallenKey_LPF_ac.net"
TRAN_NET_FILE = "SallenKey_LPF_tran.net"
RAW_AC_FILE = "SallenKey_LPF_ac.raw"
RAW_TRAN_FILE = "SallenKey_LPF_tran.raw"
SCHEMATIC_IMG = "SallenKey_LPF.jpg"
LOG_FILE = "log.txt"

if not os.path.exists(ASC_FILE):
    print(f"Error: {ASC_FILE} not found!")
    sys.exit(1)

# E96 series multipliers
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
log_lines.append(f"Settling tolerance: ±{SETTLING_TOLERANCE*100:.1f}%\n")

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
log_lines.append(f"Actual Q (from components): {actual_Q:.3f} (target: {TARGET_Q:.3f})\n")

R1_str = format_for_ltspice(R1val)
R2_str = format_for_ltspice(R2val)
Cground_str = format_for_ltspice(Cground, True)
Cfb_str = format_for_ltspice(Cfb, True)

# -------------------------- Simulations --------------------------
params = {'R1val': R1_str, 'R2val': R2_str, 'Cground': Cground_str, 'Cfb': Cfb_str}

# AC Simulation
print("Running AC simulation...")
lt_ac = SimCommander(ASC_FILE)
lt_ac.set_parameters(**params)
lt_ac.add_instruction(".ac dec 200 10 1Meg")
lt_ac.run(run_filename=AC_NET_FILE)
lt_ac.wait_completion(timeout=60)

l_ac = ltspice.Ltspice(RAW_AC_FILE)
l_ac.parse()
freq = l_ac.get_frequency()
vout_ac = l_ac.get_data('V(vout)')
mag_sim = 20 * np.log10(np.abs(vout_ac))
phase_sim = np.angle(vout_ac, deg=True)

sys_ideal = signal.TransferFunction([(2*np.pi*TARGET_FC)**2], 
                                   [1, (2*np.pi*TARGET_FC)/TARGET_Q, (2*np.pi*TARGET_FC)**2])
w = 2 * np.pi * freq
_, mag_ideal, phase_ideal = signal.bode(sys_ideal, w=w)

# Transient Simulation
print("Running Transient simulation...")
t_max = STEP_DELAY + max(50e-6, 20 / TARGET_FC)  # Ensure enough settling time
max_step = min(1e-9, (t_max - STEP_DELAY)/3000)

pwl_str = f"PWL(0 0 {STEP_DELAY} 0 {STEP_DELAY + STEP_RISE_TIME} {STEP_AMPLITUDE} {t_max} {STEP_AMPLITUDE})"

lt_tran = SimCommander(ASC_FILE)
lt_tran.set_parameters(**params)
lt_tran.set_element_model('V3', pwl_str)
lt_tran.add_instruction(f".tran 0 {t_max} 0 {max_step} startup")
lt_tran.run(run_filename=TRAN_NET_FILE)
lt_tran.wait_completion(timeout=60)

l_tran = ltspice.Ltspice(RAW_TRAN_FILE)
l_tran.parse()
time = l_tran.get_time()
vin_tran = l_tran.get_data('V(vin)')
vout_tran = l_tran.get_data('V(vout)')

# -------------------------- Step Response Analysis --------------------------
log_lines.append("=== Step Response Analysis ===")

# Find index where step arrives (±10 ns tolerance)
step_idx = np.argmin(np.abs(time - STEP_DELAY))

# Final value (average after last 20% of sim)
final_idx_start = int(len(time) * 0.8)
final_value = np.mean(vout_tran[final_idx_start:])

# Overshoot
peak_value = np.max(vout_tran[step_idx:])
overshoot_percent = 100 * (peak_value - final_value) / STEP_AMPLITUDE if peak_value > final_value else 0.0
peak_idx = np.argmax(vout_tran[step_idx:]) + step_idx

# Rise time 10-90%
low_thresh = final_value * 0.1
high_thresh = final_value * 0.9
rise_low_idx = step_idx + np.argmax(vout_tran[step_idx:] >= low_thresh)
rise_high_idx = step_idx + np.argmax(vout_tran[step_idx:] >= high_thresh)
rise_time_us = (time[rise_high_idx] - time[rise_low_idx]) * 1e6

# Settling time (± tolerance)
settle_band_high = final_value * (1 + SETTLING_TOLERANCE)
settle_band_low = final_value * (1 - SETTLING_TOLERANCE)
settled = np.logical_and(vout_tran >= settle_band_low, vout_tran <= settle_band_high)
settle_start_idx = step_idx
for i in range(len(vout_tran) - 1, step_idx, -1):
    if not settled[i]:
        settle_start_idx = i + 1
        break
settling_time_us = (time[settle_start_idx] - time[step_idx]) * 1e6 if settle_start_idx > step_idx else 0.0

log_lines.append(f"Overshoot: {overshoot_percent:.2f}%")
log_lines.append(f"Rise time (10-90%): {rise_time_us:.2f} µs")
log_lines.append(f"Settling time (±{SETTLING_TOLERANCE*100:.1f}%): {settling_time_us:.2f} µs")

# -------------------------- Logging --------------------------
with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))
print(f"Log saved to {LOG_FILE}")

# -------------------------- Plotting --------------------------
fig = plt.figure(figsize=(12, 16))
gs = fig.add_gridspec(3, 1, height_ratios=[3, 4, 1])  # Extra height for step plot annotations

# Bode plot (unchanged)
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
ax_bode.set_title(f'Sallen-Key Bode: Target fc={TARGET_FC/1000:.1f} kHz')
ax_bode.axvline(TARGET_FC, color='green', linestyle='--', label='Target fc')
ax_bode.axvline(actual_fc, color='orange', linestyle=':', label='Actual fc')
lines1, labels1 = ax_bode.get_legend_handles_labels()
lines2, labels2 = ax_phase.get_legend_handles_labels()
ax_bode.legend(lines1 + lines2, labels1 + labels2, loc='lower left')

# Step response with improved annotation placement
ax_step = fig.add_subplot(gs[1, 0])
ax_step.plot(time*1e6, vin_tran, label='Vin (step)', color='gray', linestyle='--', linewidth=1.5)
ax_step.plot(time*1e6, vout_tran, label='Vout', color='blue', linewidth=2)
ax_step.grid(True)
ax_step.set_xlabel('Time (µs)')
ax_step.set_ylabel('Voltage (V)')
ax_step.set_title(f'Step Response (Actual Q = {actual_Q:.3f}, Target Q = {TARGET_Q:.3f})')
ax_step.legend(loc='lower right')

# Auto ylim with padding for annotation space
y_min = min(vout_tran.min(), vin_tran.min()) - 0.15
y_max = max(vout_tran.max(), vin_tran.max()) + 0.15
ax_step.set_ylim(y_min, y_max)

# Metrics text box (moved slightly lower to avoid title overlap)
metrics_text = (
    f"Target fc: {TARGET_FC/1000:.1f} kHz\n"
    f"Actual fc: {actual_fc/1000:.1f} kHz\n"
    f"Target Q: {TARGET_Q:.3f}\n"
    f"Actual Q: {actual_Q:.3f}\n"
    f"Overshoot: {overshoot_percent:.2f}%\n"
    f"Rise Time (10-90%): {rise_time_us:.2f} µs\n"
    f"Settling Time (±{SETTLING_TOLERANCE*100:.1f}%): {settling_time_us:.2f} µs"
)
ax_step.text(0.7, 0.5, metrics_text, transform=ax_step.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

# Overshoot annotation - local to peak, text offset right with bbox
if overshoot_percent > 0.5:  # Only if meaningful
    ax_step.annotate('', xy=(time[peak_idx]*1e6, final_value), xytext=(time[peak_idx]*1e6, peak_value),
                     arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax_step.text(time[peak_idx]*1e6 + 8, (peak_value + final_value)/2, f'{overshoot_percent:.2f}%',
                 color='red', fontsize=11, fontweight='bold', ha='left', va='center',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='red'))

# Rise time annotation - vertical bracket on rising edge, text offset right
mid_rise_y = final_value * 0.5
ax_step.hlines([final_value*0.1, final_value*0.9], time[rise_low_idx]*1e6, time[rise_high_idx]*1e6,
               color='green', linestyle='--', lw=1.5)
ax_step.annotate('', xy=(time[rise_low_idx]*1e6, final_value*0.1), xytext=(time[rise_low_idx]*1e6, final_value*0.9),
                 arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax_step.text(time[rise_high_idx]*1e6 + 5, mid_rise_y, f'Rise: {rise_time_us:.2f} µs',
             color='green', fontsize=11, ha='left', va='center',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='green'))

# Settling time annotation - arrow at bottom, text above it on the right side
settle_y = y_min + 0.05 * (y_max - y_min)  # Near bottom with space
ax_step.annotate('', xy=(time[step_idx]*1e6, settle_y), xytext=(time[settle_start_idx]*1e6, settle_y),
                 arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
ax_step.text((time[step_idx] + time[settle_start_idx])*0.5e6, settle_y + 0.08*(y_max - y_min),
             f'Settling (±{SETTLING_TOLERANCE*100:.0f}%): {settling_time_us:.2f} µs',
             color='purple', fontsize=11, ha='center', va='bottom',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='purple'))

# Settling band (light shaded)
ax_step.axhspan(settle_band_low, settle_band_high, color='cyan', alpha=0.15)

# Schematic
ax_sch = fig.add_subplot(gs[2, 0])
if os.path.exists(SCHEMATIC_IMG):
    ax_sch.imshow(plt.imread(SCHEMATIC_IMG))
ax_sch.axis('off')
ax_sch.set_title('Schematic')

# Interactive hover
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