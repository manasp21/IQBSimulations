import numpy as np
import matplotlib.pyplot as plt

# Define constants (atomic units, a0 = 1)
a0 = 1.0

# Wavefunction for 1s (n=1, l=0)
def psi_1s(r):
    return (1 / np.sqrt(np.pi * a0**3)) * np.exp(-r / a0)

# Density for 1s
def density_1s(r):
    return np.abs(psi_1s(r))**2

# Wavefunction for 2s (n=2, l=0)
def psi_2s(r):
    return (1 / np.sqrt(32 * np.pi * a0**3)) * (2 - r / a0) * np.exp(-r / (2 * a0))

# Density for 2s
def density_2s(r):
    return np.abs(psi_2s(r))**2

# Create grid for 2D slice (x-y plane, z=0)
grid_size = 100  # Higher resolution for better plot
x = np.linspace(-10 * a0, 10 * a0, grid_size)
y = np.linspace(-10 * a0, 10 * a0, grid_size)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

# Compute densities on the 2D grid
dens_1s_2d = density_1s(R)
dens_2s_2d = density_2s(R)

# Plot 2D cross-section for 1s
fig1, ax1 = plt.subplots()
p1 = ax1.pcolormesh(X, Y, dens_1s_2d, shading='auto', cmap='viridis')
fig1.colorbar(p1, ax=ax1, label='Probability Density |ψ|^2 (a.u.)')
ax1.set_aspect('equal')
ax1.set_title('2D Cross-Section of 3D Atomic Density for 1s Orbital (z=0 plane)')
ax1.set_xlabel('x (a0)')
ax1.set_ylabel('y (a0)')

# Plot 2D cross-section for 2s
fig2, ax2 = plt.subplots()
p2 = ax2.pcolormesh(X, Y, dens_2s_2d, shading='auto', cmap='viridis')
fig2.colorbar(p2, ax=ax2, label='Probability Density |ψ|^2 (a.u.)')
ax2.set_aspect('equal')
ax2.set_title('2D Cross-Section of 3D Atomic Density for 2s Orbital (z=0 plane)')
ax2.set_xlabel('x (a0)')
ax2.set_ylabel('y (a0)')

# Also, 1D radial plots
r = np.linspace(0, 20 * a0, 500)
dens_1s_rad = density_1s(r)
dens_2s_rad = density_2s(r)

fig3, ax3 = plt.subplots()
ax3.plot(r, dens_1s_rad, label='1s')
ax3.plot(r, dens_2s_rad, label='2s')
ax3.set_title('Radial Atomic Density |ψ(r)|^2 vs r')
ax3.set_xlabel('r (a0)')
ax3.set_ylabel('|ψ|^2 (a.u.)')
ax3.legend()
ax3.grid(True)

# Display all plots
plt.show()