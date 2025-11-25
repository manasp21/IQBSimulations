import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import cumulative_trapezoid
import matplotlib.animation as animation

# Superposition of 3d (n=3, l=2, m=0) and 2p (n=2, l=1, m=0) states

def R21(r):
    return (1 / np.sqrt(24)) * r * np.exp(-0.5 * r)

def R32(r):
    return (4 / (81 * np.sqrt(30))) * r**2 * np.exp(-r/3)

def Y10(theta):
    return np.sqrt(3 / (4 * np.pi)) * np.cos(theta)

def Y20(theta):
    return np.sqrt(5 / (16 * np.pi)) * (3 * np.cos(theta)**2 - 1)

def psi_2p(r, theta):
    return R21(r) * Y10(theta)

def psi_3d(r, theta):
    return R32(r) * Y20(theta)

def sample_from_3d_state(num_samples):
    r_vals = np.linspace(0, 40, 2000)
    P_r = r_vals**2 * R32(r_vals)**2
    cdf_r = cumulative_trapezoid(P_r, r_vals, initial=0)
    cdf_r /= cdf_r[-1]
    u = np.random.uniform(0, 1, num_samples)
    r = np.interp(u, cdf_r, r_vals)
    
    x_vals = np.linspace(-1, 1, 1000)
    P_x = (3 * x_vals**2 - 1)**2
    cdf_x = cumulative_trapezoid(P_x, x_vals, initial=0)
    cdf_x /= cdf_x[-1]
    u_x = np.random.uniform(0, 1, num_samples)
    x = np.interp(u_x, cdf_x, x_vals)
    theta = np.arccos(x)
    return r, theta

def sample_from_2p_state(num_samples):
    r_vals = np.linspace(0, 40, 2000)
    P_r = r_vals**2 * R21(r_vals)**2
    cdf_r = cumulative_trapezoid(P_r, r_vals, initial=0)
    cdf_r /= cdf_r[-1]
    u = np.random.uniform(0, 1, num_samples)
    r = np.interp(u, cdf_r, r_vals)
    
    u_x = np.random.uniform(0, 1, num_samples)
    x = np.cbrt(2 * u_x - 1)
    theta = np.arccos(x)
    return r, theta

def generate_points_t(t, num_points, omega=1.0):
    x_out, y_out, z_out = [], [], []
    
    while len(x_out) < num_points:
        batch_size = (num_points - len(x_out)) * 4
        
        mask_3d = np.random.rand(batch_size) < 0.5
        n_3d = np.sum(mask_3d)
        n_2p = batch_size - n_3d
        
        r_3d, theta_3d = sample_from_3d_state(n_3d)
        r_2p, theta_2p = sample_from_2p_state(n_2p)
        
        r = np.concatenate([r_3d, r_2p])
        theta = np.concatenate([theta_3d, theta_2p])
        
        p3 = psi_3d(r, theta)
        p2 = psi_2p(r, theta)
        
        denom = p3**2 + p2**2
        denom[denom < 1e-20] = 1e-20
        
        ratio = 1 + (2 * p3 * p2 * np.cos(omega * t)) / denom
        accept_prob = ratio / 2.0
        
        u = np.random.rand(batch_size)
        accepted = u < accept_prob
        
        r_acc = r[accepted]
        theta_acc = theta[accepted]
        phi_acc = np.random.uniform(0, 2 * np.pi, len(r_acc))
        
        x_acc = r_acc * np.sin(theta_acc) * np.cos(phi_acc)
        y_acc = r_acc * np.sin(theta_acc) * np.sin(phi_acc)
        z_acc = r_acc * np.cos(theta_acc)
        
        x_out.extend(x_acc)
        y_out.extend(y_acc)
        z_out.extend(z_acc)
        
    return np.array(x_out[:num_points]), np.array(y_out[:num_points]), np.array(z_out[:num_points])

# Animation setup
num_points = 50000
num_frames = 100
max_r = 30.0
plot_range = [-20, 20]
omega = 1.0 

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
# Standard style: white background, blue points
sc = ax.scatter([], [], [], s=0.5, c='blue', alpha=0.05)

ax.set_xlabel('x ($a_0$)')
ax.set_ylabel('y ($a_0$)')
ax.set_zlabel('z ($a_0$)')
ax.set_xlim(plot_range)
ax.set_ylim(plot_range)
ax.set_zlim(plot_range)
ax.set_title('Superposition: 3d ($m=0$) + 2p ($m=0$)')

def update(frame):
    t = frame * (2 * np.pi / num_frames)
    x, y, z = generate_points_t(t, num_points, omega=omega)
    sc._offsets3d = (x, y, z)
    ax.set_title(f'Superposition t={t:.2f}')
    return sc,

ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)

print("Saving animation...")
ani.save('3d_2p_superposition.mp4', writer='ffmpeg', fps=30)
print("Done.")
# plt.show()
