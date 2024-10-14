import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
from io import BytesIO
import time

class ViscekSimulation:
    def __init__(self, v0=1.0, eta=0.5, L=10, R=1, dt=0.2, Nt=100, N=500, plotRealTime=False, use_gpu=False):
        self.v0 = v0              # velocity
        self.eta = eta            # random fluctuation in angle (in radians)
        self.L = L                # size of box
        self.R = R                # interaction radius
        self.dt = dt              # time step
        self.Nt = Nt              # number of time steps
        self.N = N                # number of particles
        self.plotRealTime = plotRealTime
        self.plt_cmap = 'cool'

        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        # Record the start time for initialization
        self.start_time_init = time.time()

        # Initialize positions and velocities
        self.positions = self.initialize_positions()
        self.velocities, self.theta, self.phi = self.initialize_velocities()

        # Calculate the time taken for initialization
        self.initialization_time = time.time() - self.start_time_init

        # Prep figure for plotting
        self.fig = plt.figure(figsize=(8, 8), dpi=80)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.frames = []

        # Create the color map and color bar
        self.norm = plt.Normalize(0, self.v0)  # Normalize based on max velocity
        self.cmap = plt.cm.ScalarMappable(norm=self.norm, cmap=self.plt_cmap)
        self.cbar = self.fig.colorbar(self.cmap, ax=self.ax, pad=0.1, shrink=0.5)
        self.cbar.set_label('Velocity Magnitude', rotation=270, labelpad=15)

        self.positions_np = self.positions.cpu().numpy()
        self.velocities_np = self.velocities.cpu().numpy()

    def initialize_positions(self):
        """Initialize random positions for particles within the box."""
        torch.manual_seed(17)
        positions = torch.rand(self.N, 3, device=self.device) * self.L
        return positions

    def initialize_velocities(self):
        """Initialize random velocities for particles."""
        theta = 2 * np.pi * torch.rand(self.N, 1, device=self.device)
        phi = np.pi * (torch.rand(self.N, 1, device=self.device) - 0.5)
        vx = self.v0 * torch.cos(theta) * torch.cos(phi)
        vy = self.v0 * torch.sin(theta) * torch.cos(phi)
        vz = self.v0 * torch.sin(phi)
        velocities = torch.cat((vx, vy, vz), dim=1)
        return velocities, theta, phi

    def apply_periodic_boundary_conditions(self):
        """Apply periodic boundary conditions to positions."""
        self.positions = self.positions % self.L
        self.positions_np = self.positions.cpu().numpy() 

    def update_velocities(self):
        """Update particle velocities based on mean angles of neighbors"""
        mean_theta = self.theta.clone()
        mean_phi = self.phi.clone()

        # Compute distance matrices 
        expanded_positions = self.positions.unsqueeze(1)  # Shape: (N, 1, 3)
        diff = expanded_positions - self.positions  # Shape: (N, N, 3)
        distances = torch.sum(diff**2, dim=2)  # Squared distances, Shape: (N, N)

        neighbors = distances < self.R**2  # Identify neighbors within interaction radius

        # Summation over neighbors
        cos_theta_cos_phi = torch.cos(self.theta) * torch.cos(self.phi)
        sin_theta_cos_phi = torch.sin(self.theta) * torch.cos(self.phi)
        sin_phi = torch.sin(self.phi)

        sx = torch.einsum('ij,i->j', neighbors.float(), cos_theta_cos_phi.flatten())
        sy = torch.einsum('ij,i->j', neighbors.float(), sin_theta_cos_phi.flatten())
        sz = torch.einsum('ij,i->j', neighbors.float(), sin_phi.flatten())

        # Compute mean angles based on the sums
        mean_theta = torch.atan2(sy, sx).view(-1, 1)
        mean_phi = torch.atan2(sz, torch.sqrt(sx**2 + sy**2)).view(-1, 1)

        # Add random perturbations
        self.theta = mean_theta + self.eta * (torch.rand(self.N, 1, device=self.device) - 0.5)
        self.phi = mean_phi + self.eta * (torch.rand(self.N, 1, device=self.device) - 0.5)
        
        # Update velocities
        vx = self.v0 * torch.cos(self.theta) * torch.cos(self.phi)
        vy = self.v0 * torch.sin(self.theta) * torch.cos(self.phi)
        vz = self.v0 * torch.sin(self.phi)
        self.velocities = torch.cat((vx, vy, vz), dim=1)

        # Updated velocities NumPy array
        self.velocities_np = self.velocities.cpu().numpy()

    def plot_frame(self, step):
        self.ax.cla()

        velocity_magnitude = torch.norm(self.velocities, dim=1).cpu().numpy()
        norm = plt.Normalize(velocity_magnitude.min(), velocity_magnitude.max())
        self.cmap = plt.cm.ScalarMappable(norm=norm, cmap=self.plt_cmap)

        # Plot the particles
        self.ax.quiver(self.positions_np[:, 0], self.positions_np[:, 1], self.positions_np[:, 2], self.velocities_np[:, 0], self.velocities_np[:, 1], self.velocities_np[:, 2], 
                       color=self.cmap.to_rgba(velocity_magnitude), length=0.5, linewidth=1.5, alpha=0.8)

        self.ax.quiver(0, 0, 0, 1, 0, 0, color='r', linewidths=3, arrow_length_ratio=0.2)
        self.ax.quiver(0, 0, 0, 0, 1, 0, color='g', linewidths=3, arrow_length_ratio=0.1)
        self.ax.quiver(0, 0, 0, 0, 0, 1, color='b', linewidths=3, arrow_length_ratio=0.1)
        self.ax.scatter(0, 0, 0, color='k', s=100)

        # Set background color
        self.ax.xaxis.pane.fill = True
        self.ax.xaxis.pane.set_facecolor('white')
        self.ax.yaxis.pane.fill = True
        self.ax.yaxis.pane.set_facecolor('white')
        self.ax.zaxis.pane.fill = True
        self.ax.zaxis.pane.set_facecolor('white')

        # Set view angle
        #self.ax.view_init(elev=20, azim=step % 360)
        self.ax.view_init(elev=20, azim=45 % 360)
        self.ax.set_xlim(0, self.L)
        self.ax.set_ylim(0, self.L)
        self.ax.set_zlim(0, self.L)
        self.ax.set_aspect('auto')
        self.ax.set_title(f'Viscek Simulation - Step {step + 1}')

        # Save the frame to a buffer
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        self.frames.append(imageio.imread(buf))
        buf.close()

    def run_simulation(self):
        """Run the entire simulation and save frames to create a video."""
        self.start_time_simulation = time.time()  # Start timing the simulation

        for step in range(self.Nt):
            self.positions += self.velocities * self.dt
            self.apply_periodic_boundary_conditions()
            self.update_velocities()
            self.plot_frame(step)
            print(f"Step {step + 1}/{self.Nt} completed")
        
        self.total_simulation_time = time.time() - self.start_time_simulation  # Total time for simulation
        self.save_video()

    def save_video(self):
        """Save the frames as a video."""
        with imageio.get_writer('Viscek_simulation.mp4', fps=5) as writer:
            for frame in self.frames:
                writer.append_data(frame)
        print("Video saved as Viscek_simulation.mp4")
        print(f"Time taken for initialization: {self.initialization_time:.4f} seconds")
        print(f"Total time taken for simulation: {self.total_simulation_time:.4f} seconds")

if __name__ == "__main__":
    sim = ViscekSimulation(use_gpu=True)
    sim.run_simulation()