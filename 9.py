import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import cumulative_trapezoid
import matplotlib.animation as animation

# Radial wavefunctions in atomic units
def R20(r):
    return (1 / np.sqrt(2)) * (1 - 0.5 * r) * np.exp(-0.5 * r)

def R21(r):
    return (1 / np.sqrt(24)) * r * np.exp(-0.5 * r)

# Time-dependent radial density (average part)
def rho_radial_t(r, alpha):
    cos2 = np.cos(alpha)**2
    sin2 = np.sin(alpha)**2
    return cos2 * R20(r)**2 + sin2 * R21(r)**2

# Sample radial positions from the time-dependent probability
def sample_r(t, num_samples, max_r=20.0, omega=1.0):
    alpha = omega * t / 2.0
    r_vals = np.linspace(0, max_r, 1000)
    P = r_vals**2 * rho_radial_t(r_vals, alpha)  
    cdf = cumulative_trapezoid(P, r_vals, initial=0)
    if cdf[-1] == 0:
        cdf[-1] = 1  
    cdf /= cdf[-1]  
    u = np.random.uniform(0, 1, num_samples)
    r_sampled = np.interp(u, cdf, r_vals)
    return r_sampled

# Interference factor k(r, t)
def get_k(r, t, omega=1.0):
    alpha = omega * t / 2.0
    r20 = R20(r)
    r21 = R21(r)
    denom = np.cos(alpha)**2 * r20**2 + np.sin(alpha)**2 * r21**2
    if denom == 0:
        return 0.0
    k = np.sin(omega * t) * (r20 * r21 / denom)
    k = np.clip(k, -1, 1)  # Ensure stability
    return k

# Sample cos_theta from the angular distribution
def sample_cos_theta(k):
    if abs(k) < 1e-6:
        return np.random.uniform(-1, 1)
    a = k / 4.0
    b = 0.5
    cc = 0.5 - k / 4.0
    y = np.random.uniform(0, 1)
    disc = b**2 - 4 * a * (cc - y)
    if disc < 0:
        disc = 0
    sd = np.sqrt(disc)
    u1 = (-b + sd) / (2 * a) if abs(a) > 1e-10 else (2 * y - 1)
    u2 = (-b - sd) / (2 * a) if abs(a) > 1e-10 else (2 * y - 1)
    if -1 <= u1 <= 1:
        return u1
    elif -1 <= u2 <= 1:
        return u2
    else:
        return np.random.uniform(-1, 1)

# Generate 3D points for density at time t
def generate_points_t(t, num_points, max_r=20.0, omega=1.0):
    np.random.seed(42)  # Fixed seed for smooth animation
    r = sample_r(t, num_points, max_r, omega)
    cos_theta = np.array([sample_cos_theta(get_k(ri, t, omega)) for ri in r])
    theta = np.arccos(cos_theta)
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

# Animation parameters
num_points = 50000
num_frames = 50
max_r = 20.0
plot_range = [-10, 10]
omega = 1.0

# Setup figure
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter([], [], [], s=0.5, c='green', alpha=0.05)

def update(frame):
    t = frame * (2 * np.pi / num_frames)
    x, y, z = generate_points_t(t, num_points, max_r, omega)
    sc._offsets3d = (x, y, z)
    ax.set_title(f'Stark Superposition $\\rho(t)$ at t = {t:.2f} (arb. units)')
    return sc,

ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False, repeat=True)

ax.set_xlabel('x ($a_0$)')
ax.set_ylabel('y ($a_0$)')
ax.set_zlabel('z ($a_0$)')
ax.set_xlim(plot_range)
ax.set_ylim(plot_range)
ax.set_zlim(plot_range)

# Save as MP4 (requires ffmpeg) or GIF (requires Pillow)
ani.save('stark_superposition_animation.mp4', writer='ffmpeg', fps=30)  # Or 'pillow' for .gif

plt.show()  # Optional interactive display