from manim import *
import numpy as np

class DensityAnimation(ThreeDScene):
    def construct(self):
        # Parameters (verified for n=2 hydrogen, atomic units)
        num_points = 5000  # Balanced for Manim performance; increase if hardware allows
        max_r = 20.0
        omega = 1.0  # Arbitrary; physical ~6.65e9 rad/s
        num_cycles = 3  # For longer animation
        animation_time = 10.0  # Total run time in seconds

        # Radial wavefunctions (verified against Laguerre polynomials)
        def R20(r):
            return (1 / np.sqrt(2)) * (1 - 0.5 * r) * np.exp(-0.5 * r)

        def R21(r):
            return (1 / np.sqrt(24)) * r * np.exp(-0.5 * r)

        # Average rho for radial sampling
        def rho_avg(r):
            return 0.5 * (R20(r)**2 / (4 * np.pi)) + 0.5 * (R21(r)**2 / (4 * np.pi))

        # Sample fixed r (using np.cumsum for CDF, no SciPy needed)
        np.random.seed(42)  # Fixed for coherence
        r_vals = np.linspace(0, max_r, 1000)
        dr = r_vals[1] - r_vals[0]
        P = 4 * np.pi * r_vals**2 * rho_avg(r_vals)
        cdf = np.cumsum(P) * dr  # Approximate integral
        cdf /= cdf[-1]  # Normalize
        u = np.random.uniform(0, 1, num_points)
        r_fixed = np.interp(u, cdf, r_vals)

        # Fixed phi
        phi_fixed = np.random.uniform(0, 2 * np.pi, num_points)

        # Interference k(r, t)
        def get_k(r, t):
            r20 = R20(r)
            r21 = R21(r)
            denom = r20**2 + r21**2
            if denom == 0:
                return 0.0
            return -2.0 * (r20 * r21 / denom) * np.cos(omega * t)

        # Sample cos_theta given k (quadratic inversion)
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

        # Create initial point cloud as VGroup of Dot3D
        cloud = VGroup()
        for i in range(num_points):
            dot = Dot3D(radius=0.02, color=GREEN)  # Small points for cloud
            cloud.add(dot)

        # Function to update positions based on t
        def update_cloud(cloud, dt):  # dt not used; use tracker
            t = time_tracker.get_value()
            np.random.seed(42 + int(t * 1000) % 100000)  # Evolving seed for smooth variation? No, fix sequence but t-dependent sampling
            # Critical: To mimic fixed random sequence with t-dependent output
            # Reset seed each update, then sample y in order
            np.random.seed(42)
            for i, dot in enumerate(cloud):
                # Sample y once per point (fixed sequence)
                _ = np.random.uniform()  # Dummy to advance if needed; but actually, since seed reset, sequence fixed
                k = get_k(r_fixed[i], t)
                cos_theta = sample_cos_theta(k)
                theta = np.arccos(cos_theta)
                x = r_fixed[i] * np.sin(theta) * np.cos(phi_fixed[i])
                y = r_fixed[i] * np.sin(theta) * np.sin(phi_fixed[i])
                z = r_fixed[i] * np.cos(theta)
                dot.move_to([x, y, z])

        # Set initial camera (fixed orientation, ambient rotation for view)
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.1)  # Slow rotation for 3D feel, no axis spin

        # Time tracker for evolution
        time_tracker = ValueTracker(0)

        # Add cloud and updater
        cloud.add_updater(update_cloud)
        self.add(cloud)

        # Animate time from 0 to 2*pi*num_cycles
        self.play(time_tracker.animate.set_value(2 * PI * num_cycles), run_time=animation_time, rate_func=linear)

        # Cleanup
        cloud.clear_updaters()
        self.wait(1)