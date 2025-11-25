import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import cumulative_trapezoid
import matplotlib.animation as animation

# Radial wavefunctions for n=3
def R32(r):
    # 3d
    return (4 / (81 * np.sqrt(30))) * r**2 * np.exp(-r/3)

def sample_r_3d(num_samples, max_r=40.0):
    r_vals = np.linspace(0, max_r, 2000)
    R = R32(r_vals)
    P = r_vals**2 * R**2
    cdf = cumulative_trapezoid(P, r_vals, initial=0)
    cdf /= cdf[-1]
    u = np.random.uniform(0, 1, num_samples)
    r_sampled = np.interp(u, cdf, r_vals)
    return r_sampled

def sample_theta_3d(num_samples):
    # Y20 ~ (3 cos^2 theta - 1)
    x_vals = np.linspace(-1, 1, 1000)
    P_x = (3 * x_vals**2 - 1)**2
    cdf = cumulative_trapezoid(P_x, x_vals, initial=0)
    cdf /= cdf[-1]
    u = np.random.uniform(0, 1, num_samples)
    x_sampled = np.interp(u, cdf, x_vals)
    return np.arccos(x_sampled)

def generate_points_3d(num_points, max_r=40.0):
    r = sample_r_3d(num_points, max_r)
    theta = sample_theta_3d(num_points)
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

# Animation setup
num_points = 50000
num_frames = 60
max_r = 35.0
plot_range = [-25, 25]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Initial plot
x, y, z = generate_points_3d(num_points, max_r)
sc = ax.scatter(x, y, z, s=0.5, c='green', alpha=0.05)

ax.set_xlabel('x ($a_0$)')
ax.set_ylabel('y ($a_0$)')
ax.set_zlabel('z ($a_0$)')
ax.set_xlim(plot_range)
ax.set_ylim(plot_range)
ax.set_zlim(plot_range)
ax.set_title('Hydrogen 3d ($m=0$) Orbital Cloud')

# Animate by regenerating points (shimmering effect)
def update(frame):
    x, y, z = generate_points_3d(num_points, max_r)
    sc._offsets3d = (x, y, z)
    return sc,

ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False)

print("Saving animation...")
ani.save('3d_orbital_animation.mp4', writer='ffmpeg', fps=30)
print("Done.")
# plt.show()
