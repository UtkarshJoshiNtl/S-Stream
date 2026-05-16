import numpy as np

class CPULBM2D:
    """
    CPU-based D2Q9 Lattice Boltzmann Method implementation.
    
    D2Q9 lattice: 9 velocities in 2D
    - 0: (0, 0) - rest
    - 1: (1, 0) - east
    - 2: (0, 1) - north
    - 3: (-1, 0) - west
    - 4: (0, -1) - south
    - 5: (1, 1) - northeast
    - 6: (-1, 1) - northwest
    - 7: (-1, -1) - southwest
    - 8: (1, -1) - southeast
    """
    
    def __init__(self, width=128, height=128, viscosity=0.02):
        self.width = width
        self.height = height
        self.viscosity = viscosity
        
        # Relaxation parameter (omega = 1 / (3*viscosity + 0.5))
        self.omega = 1.0 / (3.0 * viscosity + 0.5)
        
        # D2Q9 lattice constants
        self.w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
        self.cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
        self.cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
        self.opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])  # Opposite directions
        
        # Initialize distribution functions
        # Shape: (9, height, width) - SOA-like structure
        self.f = np.zeros((9, height, width))
        
        # Macroscopic quantities
        self.rho = np.ones((height, width))
        self.u = np.zeros((height, width))
        self.v = np.zeros((height, width))
        
        # Obstacle mask
        self.obstacles = np.zeros((height, width), dtype=bool)
        
        # Initialize with equilibrium
        self.initialize(rho=1.0, u=0.1, v=0.0)
    
    def equilibrium(self, rho, u, v):
        """
        Compute equilibrium distribution function.
        
        Args:
            rho: Density field (height, width)
            u: Velocity x-component (height, width)
            v: Velocity y-component (height, width)
        
        Returns:
            feq: Equilibrium distribution (9, height, width)
        """
        feq = np.zeros((9, self.height, self.width))
        u2 = u**2 + v**2
        
        for i in range(9):
            cu = self.cx[i] * u + self.cy[i] * v
            feq[i] = self.w[i] * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*u2)
        
        return feq
    
    def initialize(self, rho=1.0, u=0.1, v=0.0):
        """
        Initialize simulation with uniform flow.
        
        Args:
            rho: Initial density
            u: Initial velocity x-component
            v: Initial velocity y-component
        """
        self.rho[:] = rho
        self.u[:] = u
        self.v[:] = v
        self.f = self.equilibrium(self.rho, self.u, self.v)
    
    def collision(self):
        """
        BGK collision step.
        """
        # Compute macroscopic quantities
        self.rho = np.sum(self.f, axis=0)
        self.u = np.sum(self.f * self.cx[:, np.newaxis, np.newaxis], axis=0) / self.rho
        self.v = np.sum(self.f * self.cy[:, np.newaxis, np.newaxis], axis=0) / self.rho
        
        # BGK collision
        feq = self.equilibrium(self.rho, self.u, self.v)
        self.f = self.f * (1 - self.omega) + feq * self.omega
    
    def streaming(self):
        """
        Streaming step with periodic boundary conditions.
        """
        for i in range(9):
            self.f[i] = np.roll(self.f[i], shift=(self.cx[i], self.cy[i]), axis=(1, 0))
    
    def apply_obstacles(self):
        """
        Apply bounce-back boundary condition for obstacles.
        """
        for i in range(9):
            self.f[i][self.obstacles] = self.f[self.opp[i]][self.obstacles]
    
    def apply_inflow(self, u_inflow=0.1):
        """
        Apply inflow boundary condition on left side.
        Sets equilibrium distribution with specified velocity.
        """
        # Left boundary (x=0)
        rho_inlet = 1.0
        u_inlet = np.full(self.height, u_inflow)
        v_inlet = np.zeros(self.height)
        
        for i in range(9):
            cu = self.cx[i] * u_inlet + self.cy[i] * v_inlet
            u2 = u_inlet**2 + v_inlet**2
            feq = self.w[i] * rho_inlet * (1 + 3*cu + 4.5*cu**2 - 1.5*u2)
            self.f[i, :, 0] = feq
    
    def apply_outflow(self):
        """
        Apply outflow boundary condition on right side.
        Zero-gradient (Neumann) boundary condition.
        """
        # Right boundary (x=width-1)
        for i in range(9):
            self.f[i, :, -1] = self.f[i, :, -2]
    
    def apply_walls(self):
        """
        Apply wall boundary conditions on top and bottom.
        Bounce-back for no-slip walls.
        """
        # Top wall (y=0)
        for i in range(9):
            self.f[i, 0, :] = self.f[self.opp[i], 0, :]
        
        # Bottom wall (y=height-1)
        for i in range(9):
            self.f[i, -1, :] = self.f[self.opp[i], -1, :]
    
    def step(self):
        """
        Execute one LBM time step.
        """
        self.streaming()
        self.apply_obstacles()
        self.apply_inflow(u_inflow=0.1)
        self.apply_outflow()
        self.apply_walls()
        self.collision()
    
    def run(self, steps):
        """
        Run simulation for specified number of steps.
        
        Args:
            steps: Number of time steps
        """
        for _ in range(steps):
            self.step()
    
    def get_density(self):
        """
        Get density field.
        
        Returns:
            Density field (height, width)
        """
        return self.rho.copy()
    
    def get_velocity(self):
        """
        Get velocity field.
        
        Returns:
            Velocity field (height, width, 2) with (u, v) components
        """
        return np.stack([self.u, self.v], axis=2)
    
    def add_obstacle(self, x, y, radius=5):
        """
        Add circular obstacle.
        
        Args:
            x: Center x-coordinate
            y: Center y-coordinate
            radius: Obstacle radius
        """
        y_grid, x_grid = np.ogrid[:self.height, :self.width]
        mask = (x_grid - x)**2 + (y_grid - y)**2 <= radius**2
        self.obstacles[mask] = True
    
    def clear_obstacles(self):
        """Clear all obstacles."""
        self.obstacles[:] = False
