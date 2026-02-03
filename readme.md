# LTSpice + Python: Sallen-Key Low-Pass Filter Demo

This repo demonstrates calling LTSpice from Python to simulate a 2nd-order Sallen-Key low-pass filter, extract AC analysis data, and plot beautiful Bode plots with Matplotlib — including ideal analytical overlay and interactive cursor.

## Files
- `SallenKey_LPF.asc`: Parametrized LTSpice schematic (~50 kHz cutoff by default).
- `Run_SallenKey_LPF.py`: Python script that runs the simulation and generates plots.
- `SallenKey_LPF.jpg`: Schematic screenshot (displayed in the plot).

## Requirements
- LTSpice (from Analog Devices)
- Python packages: `PyLTSpice`, `ltspice`, `matplotlib`, `numpy`, `scipy`
  ```bash
  pip install PyLTSpice ltspice matplotlib numpy scipy