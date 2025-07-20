import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import cumtrapz

# Define the atomic densities (in atomic units, a=1)
def rho_S(r, a=1.0):
    return (1 / (8 * np.pi * a**3)) * (1 - r / (2 * a))**2 * np.exp(-r / a)

def rho_P(r, a=1.0):
    return (1 / (96 * np.pi * a**3)) * (r / a)**2 * np.exp(-r / a)

# Function to sample r from the radial probability P(r) = 4 pi r^2 rho(r)
def sample_r(rho_func, num_samples, max_r=20, a=1.0):
    r_vals = np.linspace(0, max_r, 1000)
    P = 4 * np.pi * r_vals**2 * np.array([rho_func(r, a) for r in r_vals])
    cdf = cumtrapz(P, r_vals, initial=0)
    cdf /= cdf[-1]  # Normalize to 1
    u = np.random.uniform(0, 1, num_samples)
    r_sampled = np.interp(u, cdf, r_vals)
    return r_sampled

# Generate 3D points distributed according to rho
def generate_points(rho_func, num_points, max_r=20, a=1.0):
    r = sample_r(rho_func, num_points, max_r, a)
    cos_theta = np.random.uniform(-1, 1, num_points)  # Uniform in cos(theta) for isotropic
    theta = np.arccos(cos_theta)
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

# Plotting
num_points = 10000  # Increase for denser clouds (e.g., 50000)
max_r = 20  # In atomic units; sufficient for n=2
plot_range = [-10, 10]  # Axis limits for visibility

fig = plt.figure(figsize=(12, 6))

# 2S state plot
ax1 = fig.add_subplot(121, projection='3d')
x_s, y_s, z_s = generate_points(rho_S, num_points, max_r)
ax1.scatter(x_s, y_s, z_s, s=1, c='blue', alpha=0.05)
ax1.set_title(r'Atomic Density for $2^2S_{1/2,1/2}$')
ax1.set_xlabel('x ($a_0$)')
ax1.set_ylabel('y ($a_0$)')
ax1.set_zlabel('z ($a_0$)')
ax1.set_xlim(plot_range)
ax1.set_ylim(plot_range)
ax1.set_zlim(plot_range)

# 2P state plot
ax2 = fig.add_subplot(122, projection='3d')
x_p, y_p, z_p = generate_points(rho_P, num_points, max_r)
ax2.scatter(x_p, y_p, z_p, s=1, c='red', alpha=0.05)
ax2.set_title(r'Atomic Density for $2^2P_{1/2,1/2}$')
ax2.set_xlabel('x ($a_0$)')
ax2.set_ylabel('y ($a_0$)')
ax2.set_zlabel('z ($a_0$)')
ax2.set_xlim(plot_range)
ax2.set_ylim(plot_range)
ax2.set_zlim(plot_range)

plt.tight_layout()
plt.show()