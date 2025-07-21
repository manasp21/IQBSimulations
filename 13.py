import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import cumulative_trapezoid
import matplotlib.animation as animation

# Radial wavefunctions in atomic units
def R10(r):
    return 2 * np.exp(-r)

def R20(r):
    return (1 / np.sqrt(2)) * (1 - 0.5 * r) * np.exp(-0.5 * r)

# Time-dependent radial probability P(r,t) = 4 pi r^2 rho(r,t)
def sample_r(t, num_samples, max_r=20.0, omega=1.0):
    r_vals = np.linspace(0, max_r, 1000)
    r10 = R10(r_vals)
    r20 = R20(r_vals)
    rho_rad = 0.5 * r10**2 + 0.5 * r20**2 + np.cos(omega * t) * r10 * r20
    P = r_vals**2 * rho_rad  # Note: normalization factor absorbed in CDF
    cdf = cumulative_trapezoid(P, r_vals, initial=0)
    cdf /= cdf[-1]  # Normalize
    u = np.random.uniform(0, 1, num_samples)
    r_sampled = np.interp(u, cdf, r_vals)
    return r_sampled

# Generate 3D points at time t (isotropic)
def generate_points_t(t, num_points, max_r=20.0, omega=1.0):
    np.random.seed(42)  # Fixed for smooth animation
    r = sample_r(t, num_points, max_r, omega)
    cos_theta = np.random.uniform(-1, 1, num_points)
    theta = np.arccos(cos_theta)
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

# Animation setup
num_points = 50000  # Dense cloud
num_frames = 50
max_r = 20.0
plot_range = [-10, 10]
omega = 1.0  # Arbitrary; physical = 3/8

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter([], [], [], s=0.5, c='green', alpha=0.05)

def update(frame):
    t = frame * (2 * np.pi / num_frames)
    x, y, z = generate_points_t(t, num_points, max_r, omega)
    sc._offsets3d = (x, y, z)
    ax.set_title(f'3D Density at t = {t:.2f} (arb. units)')
    return sc,

ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False, repeat=True)

ax.set_xlabel('x ($a_0$)')
ax.set_ylabel('y ($a_0$)')
ax.set_zlabel('z ($a_0$)')
ax.set_xlim(plot_range)
ax.set_ylim(plot_range)
ax.set_zlim(plot_range)

ani.save('1s_2s_superposition_3d_animation.mp4', writer='ffmpeg', fps=30)

plt.show()