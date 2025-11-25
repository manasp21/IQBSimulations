# Quantum Mechanics Simulations

This repository contains Python scripts for simulating and visualizing quantum mechanical wavefunctions, primarily hydrogen orbitals and their superpositions.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- SciPy
- FFmpeg (for saving animations)

## Simulation Files

| File | Description |
|------|-------------|
| `1.py` - `9.py` | Various initial experiments and simulations of lower orbitals (n=1, n=2). |
| `10.py` | 2D Cross-section animation of a Stark effect or superposition. |
| `11.py` | 2D animation of orbital dynamics. |
| `12.py` | 1D/2D wavefunction visualization. |
| `13.py` | 3D animation of a superposition of 1s and 2s orbitals. |
| `14.py` | **[NEW]** 3D visualization of the Hydrogen 3d ($m=0$) orbital with camera rotation. |
| `15.py` | **[NEW]** 3D animation of the superposition of 3d ($m=0$) and 2p ($m=0$) orbitals, showing interference beating. |

## How to Run

To run a simulation and generate an animation:

```bash
python <filename>.py
```

For example:

```bash
python 14.py
```

This will display the animation window and save the output as an MP4 file (e.g., `3d_orbital_rotation.mp4`) in the current directory.

## Notes

- The simulations use atomic units where possible.
- `num_points` in the scripts controls the density of the point cloud. Higher values give better quality but are slower.
- `num_frames` controls the length of the animation.
