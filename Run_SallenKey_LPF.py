# Run_SallenKey_LPF.py
from PyLTSpice import SimCommander  # For running LTSpice
import ltspice                      # For reading .raw files
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import sys                          # For sys.exit()
from scipy import signal             # For ideal analytical Bode plot

# Switch to an interactive backend
matplotlib.use('TkAgg')

# Configuration
LTSPICE_PATH = "C:/Program Files/ADI/LTspice/LTspice.exe"  # Update if needed
ASC_FILE = "SallenKey_LPF.asc"                            # Your parametrized .asc file
RAW_FILE = "SallenKey_LPF_1.raw"                           # LTSpice output name (adjust if needed)
SCHEMATIC_IMG = "SallenKey_LPF.jpg"                        # Optional schematic screenshot

# Check if the .asc file exists
if not os.path.exists(ASC_FILE):
    print(f"Error: {ASC_FILE} not found in the current directory!")
    sys.exit(1)

# Optional: Modify .ac directive for better resolution around cutoff (e.g., more points, lower start)
# Comment out if you prefer the default in the .asc
with open(ASC_FILE, 'r') as file:
    asc_content = file.read()
asc_content = asc_content.replace('.ac oct 100 1 10Meg', '.ac dec 200 10 1Meg')
with open(ASC_FILE, 'w') as file:
    file.write(asc_content)

# Run the simulation
print("Starting LTSpice simulation...")
lt = SimCommander(ASC_FILE)
lt.run()
if not lt.wait_completion(timeout=60):  # Wait up to 60 seconds
    print("Simulation timed out!")
    sys.exit(1)
print("Simulation complete. Output saved as", RAW_FILE)

# Load the raw file
if not os.path.exists(RAW_FILE):
    print(f"Error: {RAW_FILE} not found! Simulation may have failed.")
    sys.exit(1)

l = ltspice.Ltspice(RAW_FILE)
l.parse()

# Extract frequency and output voltage (complex)
freq = l.get_frequency()
vout = l.get_data('V(vout)')  # Node is labeled "Vout" in the netlist

# Simulated magnitude and phase
mag_sim = 20 * np.log10(np.abs(vout))   # dB
phase_sim = np.angle(vout, deg=True)    # degrees

# Theoretical calculation from component values (hardcoded to match defaults)
R1val = 10e3
R2val = 10e3
Cground = 220e-12   # 220 pF (grounded cap)
Cfb = 470e-12       # 470 pF (feedback cap)

fc_theor = 1 / (2 * np.pi * np.sqrt(R1val * R2val * Cground * Cfb))
Q_theor = np.sqrt(R1val * R2val * Cground * Cfb) / ((R1val + R2val) * Cground)

print(f"Theoretical cutoff frequency: {fc_theor:.1f} Hz")
print(f"Theoretical Q factor: {Q_theor:.3f}")

# Ideal analytical Bode using SciPy (exact transfer function)
omega0 = 2 * np.pi * fc_theor
sys_ideal = signal.TransferFunction([omega0**2], [1, omega0 / Q_theor, omega0**2])
w = freq * 2 * np.pi  # Angular frequency vector matching simulation
_, mag_ideal, phase_ideal = signal.bode(sys_ideal, w=w)

# Create figure with three subplots
fig = plt.figure(figsize=(12, 12))
gs = fig.add_gridspec(3, 1, height_ratios=[2, 2, 1])

# Magnitude plot
ax1 = fig.add_subplot(gs[0, 0])
ax1.semilogx(freq, mag_sim, label='Simulated', color='blue')
ax1.semilogx(freq, mag_ideal, '--', label='Ideal Analytical', color='red')
ax1.grid(True, which="both", ls="--")
ax1.set_ylabel('Magnitude (dB)')
ax1.set_title('Sallen-Key 2nd Order Low-Pass Filter Bode Plot (~50 kHz cutoff)')
ax1.axvline(x=fc_theor, color='green', linestyle='--', label=f'-3 dB at {fc_theor:.0f} Hz')
ax1.axhline(y=-3, color='green', linestyle=':', alpha=0.7)
ax1.legend(loc='lower left')

# Phase plot
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax2.semilogx(freq, phase_sim, label='Simulated', color='purple')
ax2.semilogx(freq, phase_ideal, '--', label='Ideal Analytical', color='orange')
ax2.grid(True, which="both", ls="--")
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Phase (deg)')
ax2.axvline(x=fc_theor, color='green', linestyle='--', label=f'Cutoff {fc_theor:.0f} Hz')
ax2.legend(loc='lower left')

# Schematic image
ax3 = fig.add_subplot(gs[2, 0])
if not os.path.exists(SCHEMATIC_IMG):
    ax3.text(0.5, 0.5, "Schematic image not found\nPlace SallenKey_LPF.jpg in directory", 
             ha="center", va="center", fontsize=12)
    ax3.axis('off')
else:
    ax3.imshow(plt.imread(SCHEMATIC_IMG))
    ax3.axis('off')
ax3.set_title('Sallen-Key Low-Pass Filter Schematic')

# Interactive cursor readouts
mag_dot, = ax1.plot([], [], 'ko', markersize=5)
mag_text = ax1.text(0.05, 0.95, '', transform=ax1.transAxes, ha='left', va='top', fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="wheat"))

phase_dot, = ax2.plot([], [], 'ko', markersize=5)
phase_text = ax2.text(0.05, 0.95, '', transform=ax2.transAxes, ha='left', va='top', fontsize=10,
                      bbox=dict(boxstyle="round", facecolor="wheat"))

def on_move(event):
    if event.inaxes == ax1:
        x = event.xdata
        if x is None:
            return
        idx = np.argmin(np.abs(freq - x))
        mag_dot.set_data([freq[idx]], [mag_sim[idx]])
        mag_text.set_text(f'Freq: {freq[idx]:.1f} Hz\nMag: {mag_sim[idx]:.2f} dB')
    elif event.inaxes == ax2:
        x = event.xdata
        if x is None:
            return
        idx = np.argmin(np.abs(freq - x))
        phase_dot.set_data([freq[idx]], [phase_sim[idx]])
        phase_text.set_text(f'Freq: {freq[idx]:.1f} Hz\nPhase: {phase_sim[idx]:.2f}°')
    else:
        mag_dot.set_data([], [])
        mag_text.set_text('')
        phase_dot.set_data([], [])
        phase_text.set_text('')
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', on_move)

# Adjust and show
plt.tight_layout()
plt.show()

# Optional cleanup
if os.path.exists("LTSpice_Cleanup.py"):
    os.system("python LTSpice_Cleanup.py")