import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import cumulative_trapezoid
import matplotlib.animation as animation

# Radial wavefunctions (in atomic units, a=1)
def R20(r):
    return (1 / np.sqrt(2)) * (1 - 0.5 * r) * np.exp(-0.5 * r)

def R21(r):
    return (1 / np.sqrt(24)) * r * np.exp(-0.5 * r)

# Average density for radial sampling (independent of t)
def rho_avg(r):
    return 0.5 * (R20(r)**2 / (4 * np.pi)) + 0.5 * (R21(r)**2 / (4 * np.pi))

# Sample r from P(r) = 4 pi r^2 rho_avg(r)
def sample_r(num_samples, max_r=20.0):
    r_vals = np.linspace(0, max_r, 1000)
    P = 4 * np.pi * r_vals**2 * rho_avg(r_vals)
    cdf = cumulative_trapezoid(P, r_vals, initial=0)
    cdf /= cdf[-1]  # Normalize
    u = np.random.uniform(0, 1, num_samples)
    r_sampled = np.interp(u, cdf, r_vals)
    return r_sampled

# Compute k(r, t)
def get_k(r, t, omega=1.0):
    r20 = R20(r)
    r21 = R21(r)
    denom = r20**2 + r21**2
    if denom == 0:
        return 0.0
    return -2.0 * (r20 * r21 / denom) * np.cos(omega * t)

# Sample cos_theta = u given k
def sample_cos_theta(k):
    if abs(k) < 1e-6:
        return np.random.uniform(-1, 1)
    a = k / 4.0
    b = 0.5
    cc = 0.5 - k / 4.0
    y = np.random.uniform(0, 1)
    disc = b**2 - 4 * a * (cc - y)
    if disc < 0:
        disc = 0  # Numerical stability
    sd = np.sqrt(disc)
    u1 = (-b + sd) / (2 * a) if abs(a) > 1e-10 else (2 * y - 1)
    u2 = (-b - sd) / (2 * a) if abs(a) > 1e-10 else (2 * y - 1)
    # Select the root in [-1, 1]
    if -1 <= u1 <= 1:
        return u1
    elif -1 <= u2 <= 1:
        return u2
    else:
        # Fallback to uniform if numerical issue (rare)
        return np.random.uniform(-1, 1)

# Generate points at time t
def generate_points_t(t, num_points, max_r=20.0, omega=1.0):
    r = sample_r(num_points, max_r)
    cos_theta = np.array([sample_cos_theta(get_k(ri, t, omega)) for ri in r])
    theta = np.arccos(cos_theta)
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

# Animation setup
num_points = 10000  # Increase to 20000 for denser cloud (slower)
num_frames = 50  # Frames per period
max_r = 20.0
plot_range = [-10, 10]
omega = 1.0  # Arbitrary for animation; physical omega_L ~ 6.65e9 rad/s

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter([], [], [], s=1, c='green', alpha=0.02)  # Adjust alpha for visibility

def update(frame):
    t = frame * (2 * np.pi / num_frames)  # t from 0 to 2pi
    x, y, z = generate_points_t(t, num_points, max_r, omega)
    sc._offsets3d = (x, y, z)
    ax.set_title(f'Superposition Density at t = {t:.2f} (arb. units)')
    return sc,

ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False, repeat=True)

ax.set_xlabel('x ($a_0$)')
ax.set_ylabel('y ($a_0$)')
ax.set_zlabel('z ($a_0$)')
ax.set_xlim(plot_range)
ax.set_ylim(plot_range)
ax.set_zlim(plot_range)

plt.show()