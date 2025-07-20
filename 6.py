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

# Average radial density (time-independent part)
def rho_avg(r):
    return 0.5 * (R20(r)**2 / (4 * np.pi)) + 0.5 * (R21(r)**2 / (4 * np.pi))

# Sample radial positions from the average probability
def sample_r(num_samples, max_r=20.0):
    r_vals = np.linspace(0, max_r, 1000)
    P = 4 * np.pi * r_vals**2 * rho_avg(r_vals)
    cdf = cumulative_trapezoid(P, r_vals, initial=0)
    cdf /= cdf[-1]  # Normalize CDF
    u = np.random.uniform(0, 1, num_samples)
    r_sampled = np.interp(u, cdf, r_vals)
    return r_sampled

# Interference factor k(r, t)
def get_k(r, t, omega=1.0):
    r20 = R20(r)
    r21 = R21(r)
    denom = r20**2 + r21**2
    if denom == 0:
        return 0.0
    return -2.0 * (r20 * r21 / denom) * np.cos(omega * t)

# Sample cos_theta from the angular distribution (inverted CDF for p(cos_theta) = 0.5 (1 + k cos_theta))
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
        return np.random.uniform(-1, 1)  # Fallback

# Generate 3D points for density at time t
def generate_points_t(t, num_points, max_r=20.0, omega=1.0):
    np.random.seed(42)  # Fixed seed for coherent frames (no jiggling)
    r = sample_r(num_points, max_r)
    cos_theta = np.array([sample_cos_theta(get_k(ri, t, omega)) for ri in r])
    theta = np.arccos(cos_theta)
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

# Animation parameters
num_points = 50000  # For dense cloud; reduce to 10000 if slow
num_frames = 500  # Frames per cycle; higher for smoother animation
max_r = 20.0  # Truncation radius (captures >99% probability)
plot_range = [-10, 10]
omega = 1.0  # Arbitrary units (scale to physical Lamb frequency ~6.65e9 rad/s)

# Setup figure
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter([], [], [], s=0.5, c='green', alpha=0.05)  # Point cloud

def update(frame):
    t = frame * (2 * np.pi / num_frames)
    x, y, z = generate_points_t(t, num_points, max_r, omega)
    sc._offsets3d = (x, y, z)
    ax.set_title(f'Superposition $\\rho(t)$ at t = {t:.2f} (arb. units)')
    return sc,

ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False, repeat=True)

ax.set_xlabel('x ($a_0$)')
ax.set_ylabel('y ($a_0$)')
ax.set_zlabel('z ($a_0$)')
ax.set_xlim(plot_range)
ax.set_ylim(plot_range)
ax.set_zlim(plot_range)

# Save the animation as file (rendering happens here)
ani.save('superposition_animation2.mp4', writer='ffmpeg', fps=30)

# Optional: Display if running interactively
# plt.show()