import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import cumulative_trapezoid

# Radial functions
def R20(r):
    return (1 / np.sqrt(2)) * (1 - 0.5 * r) * np.exp(-0.5 * r)

def R21(r):
    return (1 / np.sqrt(24)) * r * np.exp(-0.5 * r)

# Densities
def rho_S(r):
    return R20(r)**2 / (4 * np.pi)

def rho_P(r):
    return R21(r)**2 / (4 * np.pi)

# Sample r from P(r) = 4 pi r^2 rho(r)
def sample_r(rho_func, num_samples, max_r=20.0):
    r_vals = np.linspace(0, max_r, 1000)
    P = 4 * np.pi * r_vals**2 * rho_func(r_vals)
    cdf = cumulative_trapezoid(P, r_vals, initial=0)
    cdf /= cdf[-1]
    u = np.random.uniform(0, 1, num_samples)
    r_sampled = np.interp(u, cdf, r_vals)
    return r_sampled

# Generate points
def generate_points(rho_func, num_points, max_r=20.0):
    np.random.seed(42)  # Fixed seed for reproducibility
    r = sample_r(rho_func, num_points, max_r)
    cos_theta = np.random.uniform(-1, 1, num_points)
    theta = np.arccos(cos_theta)
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

# Plotting
num_points = 50000  # Denser for visibility
plot_range = [-10, 10]

fig = plt.figure(figsize=(12, 6))

# 2S
ax1 = fig.add_subplot(121, projection='3d')
x_s, y_s, z_s = generate_points(rho_S, num_points)
ax1.scatter(x_s, y_s, z_s, s=0.5, c='blue', alpha=0.05)
ax1.set_title(r'$\rho_S$ for $2^2S_{1/2,1/2}$')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_xlim(plot_range)
ax1.set_ylim(plot_range)
ax1.set_zlim(plot_range)

# 2P
ax2 = fig.add_subplot(122, projection='3d')
x_p, y_p, z_p = generate_points(rho_P, num_points)
ax2.scatter(x_p, y_p, z_p, s=0.5, c='red', alpha=0.05)
ax2.set_title(r'$\rho_P$ for $2^2P_{1/2,1/2}$')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
ax2.set_xlim(plot_range)
ax2.set_ylim(plot_range)
ax2.set_zlim(plot_range)

plt.tight_layout()
plt.show()