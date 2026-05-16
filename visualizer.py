import pygame
import numpy as np

class FluidVisualizer:
    """
    PyGame-based visualization for fluid simulation.
    Displays density field as grayscale image.
    """
    
    def __init__(self, width, height, scale=4):
        """
        Initialize visualizer.
        
        Args:
            width: Simulation grid width
            height: Simulation grid height
            scale: Display scale factor (pixels per grid cell)
        """
        self.width = width
        self.height = height
        self.scale = scale
        
        # Initialize PyGame
        pygame.init()
        self.display_width = width * scale
        self.display_height = height * scale
        self.screen = pygame.display.set_mode((self.display_width, self.display_height))
        pygame.display.set_caption("CuFloda - Fluid Simulation")
        
        # Create surface for density field
        self.surface = pygame.Surface((width, height))
        
        # Colors
        self.bg_color = (0, 0, 0)
        self.obstacle_color = (128, 128, 128)
        
        # Font for info display
        self.font = pygame.font.Font(None, 36)
        
        # State
        self.paused = False
        self.running = True
        self.drawing_obstacle = False
    
    def render_density(self, density):
        """
        Render density field as grayscale image.
        
        Args:
            density: Density field (height, width)
        """
        # Normalize density to 0-255 range
        # Typical density range is around 0.9-1.1
        d_min, d_max = 0.95, 1.05
        d_norm = (density - d_min) / (d_max - d_min)
        d_norm = np.clip(d_norm, 0, 1)
        
        # Convert to grayscale (0-255)
        gray = (d_norm * 255).astype(np.uint8)
        
        # Create RGB array (grayscale)
        rgb = np.stack([gray, gray, gray], axis=2)
        
        # Create PyGame surface
        pygame.surfarray.blit_array(self.surface, np.transpose(rgb, (1, 0, 2)))
        
        # Scale up for display
        scaled_surface = pygame.transform.scale(self.surface, 
                                                (self.display_width, self.display_height))
        self.screen.blit(scaled_surface, (0, 0))
    
    def render_obstacles(self, obstacles):
        """
        Render obstacles as gray blocks.
        
        Args:
            obstacles: Obstacle mask (height, width)
        """
        for y in range(self.height):
            for x in range(self.width):
                if obstacles[y, x]:
                    rect = pygame.Rect(x * self.scale, y * self.scale, 
                                      self.scale, self.scale)
                    pygame.draw.rect(self.screen, self.obstacle_color, rect)
    
    def render_info(self, fps, step_count):
        """
        Render simulation info overlay.
        
        Args:
            fps: Current FPS
            step_count: Number of simulation steps
        """
        info_text = f"FPS: {fps:.1f} | Steps: {step_count} | {'PAUSED' if self.paused else 'RUNNING'}"
        text_surface = self.font.render(info_text, True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 10))
        
        controls_text = "Space: Pause | R: Reset | ESC: Quit | Mouse: Draw obstacles"
        controls_surface = self.font.render(controls_text, True, (200, 200, 200))
        self.screen.blit(controls_surface, (10, self.display_height - 40))
    
    def update(self, density, obstacles, fps, step_count):
        """
        Update display.
        
        Args:
            density: Density field
            obstacles: Obstacle mask
            fps: Current FPS
            step_count: Number of simulation steps
        """
        # Clear screen
        self.screen.fill(self.bg_color)
        
        # Render density field
        self.render_density(density)
        
        # Render obstacles
        self.render_obstacles(obstacles)
        
        # Render info overlay
        self.render_info(fps, step_count)
        
        # Update display
        pygame.display.flip()
    
    def handle_events(self, sim):
        """
        Handle PyGame events.
        
        Args:
            sim: Simulation instance (for obstacle interaction)
        
        Returns:
            False if should quit, True otherwise
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    sim.initialize(rho=1.0, u=0.1, v=0.0)
                    sim.clear_obstacles()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.drawing_obstacle = True
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.drawing_obstacle = False
            
            elif event.type == pygame.MOUSEMOTION:
                if self.drawing_obstacle:
                    x, y = pygame.mouse.get_pos()
                    grid_x = x // self.scale
                    grid_y = y // self.scale
                    sim.add_obstacle(grid_x, grid_y, radius=3)
        
        return True
    
    def close(self):
        """Clean up PyGame resources."""
        pygame.quit()
