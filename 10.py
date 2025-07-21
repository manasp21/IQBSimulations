import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm

# Radial wavefunctions
def R20(r):
    return (1 / np.sqrt(2)) * (1 - 0.5 * r) * np.exp(-0.5 * r)

def R21(r):
    return (1 / np.sqrt(24)) * r * np.exp(-0.5 * r)

# Density function rho(r, theta, t)
def rho(r, theta, t, omega=1.0):
    alpha = omega * t / 2.0
    cos_a2 = np.cos(alpha)**2
    sin_a2 = np.sin(alpha)**2
    sin_wt = np.sin(omega * t)
    r20 = R20(r)
    r21 = R21(r)
    density = (1 / (4 * np.pi)) * (cos_a2 * r20**2 + sin_a2 * r21**2 + sin_wt * r20 * r21 * np.cos(theta))
    return np.clip(density, 1e-10, None)  # Safe for log

# Animation parameters
omega = 1.0
num_frames = 100
r_max = 20.0
grid_size = 2000
num_cycles = 3

# Precompute grid
x = np.linspace(-r_max, r_max, grid_size)
z = np.linspace(-r_max, r_max, grid_size)
X, Z = np.meshgrid(x, z)
R = np.sqrt(X**2 + Z**2)
Theta = np.arccos(Z / (R + 1e-10))
mask = R <= r_max

# Setup figure
fig, ax = plt.subplots(figsize=(8, 8))
density = np.full_like(X, 0.0)
density[mask] = rho(R[mask], Theta[mask], 0, omega)
pcm = ax.pcolormesh(X, Z, density, cmap='viridis', norm=LogNorm(vmin=1e-6, vmax=0.1), shading='auto')
cbar = fig.colorbar(pcm, ax=ax, label='Density ρ (a.u.)')
ax.set_xlabel('x (a_0)')
ax.set_ylabel('z (a_0)')
ax.set_title('2D Cross-Section (z-x plane) at t=0')
ax.set_aspect('equal')

def update(frame):
    t = frame * (2 * np.pi * num_cycles / num_frames)
    density = np.full_like(X, 0.0)
    density[mask] = rho(R[mask], Theta[mask], t, omega)
    pcm.set_array(density)
    ax.set_title(f'2D Cross-Section (z-x plane) at t={t:.2f} (arb. units)')
    return [pcm]

ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100, repeat=True, blit=True)

# Save animation
ani.save('stark_2d_animation2.mp4', writer='ffmpeg', fps=30, dpi=500)

plt.show()