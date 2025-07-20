import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm_y
from skimage import measure

def plot_density(ax_3d, ax_2d, x_coords, z_coords, density, time_str):
    """
    Helper function to plot the 3D isosurface and a 2D slice of the probability density.
    
    Args:
        ax_3d: Matplotlib 3D axis object for the isosurface plot.
        ax_2d: Matplotlib 2D axis object for the slice plot.
        x_coords: 1D array of x-coordinates for the grid.
        z_coords: 1D array of z-coordinates for the grid.
        density: 3D numpy array of the probability density.
        time_str: String representation of the time for plot titles.
    """
    # --- 3D Isosurface Plot ---
    # We find an isosurface level that represents a fraction of the maximum density.
    # This value is chosen empirically to give a good visualization.
    iso_value = density.max() / 7.0
    
    # The 'spacing' argument requires a tuple of scalar values (dy, dx, dz)
    # representing the step size between points along each axis of the density grid.
    # Since our grid is cubic, the spacing is the same in all directions.
    step = x_coords[1] - x_coords[0]
    spacing = (step, step, step)
    
    # Use the marching cubes algorithm to compute the isosurface.
    verts, faces, _, _ = measure.marching_cubes(density, level=iso_value, spacing=spacing)

    # The marching_cubes function returns vertices in a grid starting at (0,0,0).
    # We must shift them to match our coordinate system, which is centered at (0,0,0).
    verts += np.array([x_coords.min(), x_coords.min(), z_coords.min()])

    # Plot the triangular mesh of the isosurface.
    ax_3d.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                       cmap='viridis', lw=0.1, alpha=0.8)
    
    ax_3d.set_title(f'3D Isosurface at {time_str}', fontsize=14)
    ax_3d.set_xlabel('x ($a_0$)', fontsize=12)
    ax_3d.set_ylabel('y ($a_0$)', fontsize=12)
    ax_3d.set_zlabel('z ($a_0$)', fontsize=12)
    ax_3d.set_aspect('equal')
    ax_3d.view_init(elev=20, azim=45)


    # --- 2D Slice Plot (in the x-z plane) ---
    # We take a slice of the 3D density data at y=0.
    slice_index = density.shape[1] // 2
    density_slice = density[:, slice_index, :]

    # Use pcolormesh for a 2D heatmap.
    c = ax_2d.pcolormesh(x_coords, z_coords, density_slice.T, cmap='magma', shading='gouraud')
    
    ax_2d.set_title(f'2D Slice (y=0) at {time_str}', fontsize=14)
    ax_2d.set_xlabel('x ($a_0$)', fontsize=12)
    ax_2d.set_ylabel('z ($a_0$)', fontsize=12)
    ax_2d.set_aspect('equal')
    plt.colorbar(c, ax=ax_2d, label='Probability Density')


def main():
    """
    Main function to calculate and plot the time-evolving probability density.
    """
    # --- 1. Setup and Constants ---
    # We work in atomic units where the Bohr radius a_0 = 1.
    a0 = 1.0
    
    # --- 2. Define Wavefunctions ---
    # Radial parts for n=2, l=0 and n=2, l=1
    def R20(r):
        return (1.0 / (2.0 * np.sqrt(2.0) * a0**1.5)) * (2.0 - r / a0) * np.exp(-r / (2.0 * a0))

    def R21(r):
        return (1.0 / (2.0 * np.sqrt(6.0) * a0**1.5)) * (r / a0) * np.exp(-r / (2.0 * a0))

    # Spatial parts (radial * angular).
    # Note: Using modern sph_harm_y(l, m, theta, phi) instead of deprecated sph_harm.
    def psi(r, theta, phi, n, l, ml):
        if n == 2 and l == 0 and ml == 0:
            return R20(r) * sph_harm_y(l, ml, theta, phi)
        elif n == 2 and l == 1:
            return R21(r) * sph_harm_y(l, ml, theta, phi)
        return np.zeros_like(r, dtype=complex)

    # --- 3. Define Probability Density Function ---
    def get_prob_density(r, theta, phi, cos_omega_t):
        # Calculate the constituent spatial wavefunctions
        psi_200 = psi(r, theta, phi, 2, 0, 0)
        psi_210 = psi(r, theta, phi, 2, 1, 0)
        psi_211 = psi(r, theta, phi, 2, 1, 1)

        # The interference term involves the product of psi_200 and psi_210.
        # Since Y_0,0 and Y_1,0 are real, their product is real.
        # The sph_harm function returns complex numbers, so we take the real part.
        interference_term = 2.0 * np.sqrt(1.0/3.0) * np.real(psi_200) * np.real(psi_210) * cos_omega_t
        
        # Sum the squared magnitudes of all components plus the interference term
        density = 0.5 * (
            np.abs(psi_200)**2 + 
            (1.0/3.0) * np.abs(psi_210)**2 + 
            (2.0/3.0) * np.abs(psi_211)**2 + 
            interference_term
        )
        return density

    # --- 4. Create a 3D Grid ---
    # We define a cubic grid in Cartesian coordinates to calculate the density on.
    grid_points = 80
    plot_range = 14.0 * a0  # Plot out to 14 Bohr radii
    x = np.linspace(-plot_range, plot_range, grid_points)
    y = np.linspace(-plot_range, plot_range, grid_points)
    z = np.linspace(-plot_range, plot_range, grid_points)
    # Note: meshgrid indexing is 'xy' by default, so output shapes are (len(y), len(x), len(z))
    X, Y, Z = np.meshgrid(x, y, z)

    # Convert Cartesian grid to spherical coordinates for the calculation
    R = np.sqrt(X**2 + Y**2 + Z**2)
    # np.arctan2(y, x) gives phi. We need theta (polar angle from z-axis).
    Theta = np.arctan2(np.sqrt(X**2 + Y**2), Z)
    Phi = np.arctan2(Y, X)

    # --- 5. Calculate Densities at Different Times ---
    # t = 0 corresponds to cos(omega_L * t) = 1
    density_t0 = get_prob_density(R, Theta, Phi, 1.0)
    
    # t = T/2 (half a period) corresponds to cos(omega_L * t) = cos(pi) = -1
    density_t_half = get_prob_density(R, Theta, Phi, -1.0)

    # --- 6. Generate Plots ---
    # Plot for t = 0
    fig1 = plt.figure(figsize=(16, 8))
    fig1.suptitle('Superposition State at $t=0$', fontsize=18, y=0.95)
    ax1_3d = fig1.add_subplot(1, 2, 1, projection='3d')
    ax1_2d = fig1.add_subplot(1, 2, 2)
    plot_density(ax1_3d, ax1_2d, x, z, density_t0, '$t=0$')
    
    # Plot for t = T/2
    fig2 = plt.figure(figsize=(16, 8))
    fig2.suptitle('Superposition State at $t = T/2 = \\pi/\\omega_L$', fontsize=18, y=0.95)
    ax2_3d = fig2.add_subplot(1, 2, 1, projection='3d')
    ax2_2d = fig2.add_subplot(1, 2, 2)
    plot_density(ax2_3d, ax2_2d, x, z, density_t_half, '$t=T/2$')

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()

if __name__ == '__main__':
    main()
