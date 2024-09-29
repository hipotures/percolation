import random
import pygame
import sys
import copy

# Initialize Pygame
pygame.init()

class ForestFire:
    def __init__(self, min_burned_fraction=0.01):
        self.size = 500  # Grid size (500x500)
        self.probability = 0.58  # Probability of a cell being a tree
        self.cell_size = 3  # Size of each cell in pixels
        self.screen = pygame.display.set_mode((self.size * self.cell_size, self.size * self.cell_size))
        pygame.display.set_caption('Forest Fire Simulation')
        self.clock = pygame.time.Clock()
        self.best_point = None
        self.best_epoch = 0
        self.burned_clusters = []  # List of sets, each set contains coordinates of burned trees in a cluster
        self.min_burned_fraction = min_burned_fraction  # Minimum fraction of grid to consider a cluster
        self.cluster_colors = self.generate_color_palette(1024)  # Precompute color palette
        self.current_cluster_index = 0  # To assign unique colors to clusters
        self.cluster_grid = [[-1 for _ in range(self.size)] for _ in range(self.size)]  # Grid to map cells to cluster indices

        # Initialize the forest
        self.generate_forest()
        self.initial_grid = copy.deepcopy(self.grid)  # Store the initial forest grid

    def generate_color_palette(self, num_colors):
        """
        Generates a list of unique colors.

        :param num_colors: Number of unique colors to generate
        :return: List of RGB tuples
        """
        palette = []
        for i in range(num_colors):
            hue = (i * 137.508) % 360  # Use golden angle approximation for even distribution
            saturation = 0.5 + (random.random() * 0.5)  # Saturation between 0.5 and 1
            value = 0.95  # High brightness
            color = self.hsv_to_rgb(hue, saturation, value)
            palette.append(color)
        return palette

    def hsv_to_rgb(self, h, s, v):
        """
        Converts HSV color space to RGB.

        :param h: Hue angle in degrees (0-360)
        :param s: Saturation (0-1)
        :param v: Value (0-1)
        :return: Tuple of RGB values (0-255)
        """
        import colorsys
        h_norm = h / 360
        r, g, b = colorsys.hsv_to_rgb(h_norm, s, v)
        return (int(r * 255), int(g * 255), int(b * 255))

    def generate_forest(self):
        """Generates the initial forest grid."""
        self.grid = [[' ' for _ in range(self.size)] for _ in range(self.size)]
        self.tree_positions = []  # List to store coordinates of trees

        for i in range(self.size):
            for j in range(self.size):
                if random.random() < self.probability:
                    self.grid[i][j] = (0, 255, 0)  # Green for trees
                    self.tree_positions.append((i, j))
                else:
                    self.grid[i][j] = (255, 255, 255)  # White for empty space

    def display_forest(self):
        """Displays the current state of the forest."""
        self.screen.fill((255, 255, 255))  # Clear screen with white background

        # Draw all burned clusters with their unique colors, only if they meet the size threshold
        min_burned_cells = self.min_burned_fraction * self.size * self.size
        for cluster_index, cluster in enumerate(self.burned_clusters):
            if len(cluster) < min_burned_cells:
                continue  # Skip clusters smaller than the threshold
            color = self.cluster_colors[cluster_index % len(self.cluster_colors)]
            for (i, j) in cluster:
                pygame.draw.rect(self.screen, color, (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size))

        # Draw remaining trees (green)
        for (i, j) in self.tree_positions:
            if self.cluster_grid[i][j] == -1 and self.grid[i][j] == (0, 255, 0):
                pygame.draw.rect(self.screen, (0, 255, 0), (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size))

        pygame.display.flip()

    def reset_forest(self):
        """Resets the forest to its initial state."""
        self.generate_forest()
        self.initial_grid = copy.deepcopy(self.grid)  # Update the initial grid
        self.best_point = None
        self.best_epoch = 0
        self.burned_clusters = []
        self.cluster_grid = [[-1 for _ in range(self.size)] for _ in range(self.size)]
        self.current_cluster_index = 0
        pygame.display.set_caption('Forest Fire Simulation')

    def is_burned(self, point):
        """
        Checks if the given point is already in any burned cluster.

        :param point: Tuple (i, j) coordinates of the tree
        :return: True if burned, False otherwise
        """
        i, j = point
        return self.cluster_grid[i][j] != -1

    def simulate_fire(self, start_point):
        """
        Simulates the forest fire starting from the given point.

        :param start_point: Tuple (i, j) coordinates of the ignition point
        :return: Number of epochs the fire lasted and the burned trees set
        """
        # Deep copy the grid to simulate on
        grid_sim = copy.deepcopy(self.grid)
        red_objects = [start_point]  # List to keep track of burning trees
        grid_sim[start_point[0]][start_point[1]] = (255, 0, 0)  # Set ignition point to red
        epoch = 0
        burned_trees = set()

        while red_objects:
            new_red_objects = set()
            for (i, j) in red_objects:
                # Check neighbors with periodic boundary conditions (PBC)
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, Down, Left, Right
                    ni = (i + di) % self.size  # Wrap around vertically
                    nj = (j + dj) % self.size  # Wrap around horizontally
                    if grid_sim[ni][nj] == (0, 255, 0):  # If neighbor is green
                        new_red_objects.add((ni, nj))  # Add new red object
                        grid_sim[ni][nj] = (255, 0, 0)  # Mark as red immediately
                grid_sim[i][j] = (0, 0, 0)  # Change self to black
                burned_trees.add((i, j))
            red_objects = list(new_red_objects)
            epoch += 1

        return epoch, burned_trees

    def run_simulations(self):
        """
        Runs simulations for each tree in the forest to find the best ignition point.
        """
        total_trees = len(self.tree_positions)
        print(f"Total number of trees to simulate: {total_trees}")
        min_burned_cells = self.min_burned_fraction * self.size * self.size
        for idx, point in enumerate(self.tree_positions):
            if self.is_burned(point):
                continue  # Skip already burned trees

            epoch, burned_trees = self.simulate_fire(point)

            # Check if the burned cluster meets the size threshold
            if len(burned_trees) < min_burned_cells:
                continue  # Skip small clusters

            # Assign cluster index and color
            cluster_index = self.current_cluster_index
            if cluster_index >= len(self.cluster_colors):
                print("Maximum number of clusters reached. Cannot assign new colors.")
                break  # Stop if we exceed the color palette

            self.burned_clusters.append(burned_trees)
            for (i, j) in burned_trees:
                self.cluster_grid[i][j] = cluster_index
            self.current_cluster_index += 1

            # Update best point if necessary
            if epoch > self.best_epoch:
                self.best_epoch = epoch
                self.best_point = point
                print(f"New best point found at {point} with {epoch} epochs.")
                self.display_forest()  # Update display to show all clusters

            if (idx + 1) % 100 == 0 or (idx + 1) == total_trees:
                print(f"Simulated {idx + 1} / {total_trees} trees.")

        print(f"\nSimulation complete. Best ignition point: {self.best_point} with {self.best_epoch} epochs.")
        self.display_forest()
        # Update the window caption with best point and epoch
        pygame.display.set_caption(f'Best Point: {self.best_point} - Epochs: {self.best_epoch}')

    def visualize_best_simulation(self):
        """
        Final visualization after all simulations are done.
        """
        self.display_forest()

        # Update the window caption with best point and epoch
        pygame.display.set_caption(f'Best Point: {self.best_point} - Epochs: {self.best_epoch}')

    def run(self):
        """
        Main loop to handle user interactions and run simulations.
        """
        running = True
        simulation_done = False

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_0:
                        self.reset_forest()
                        simulation_done = False
                    elif event.key == pygame.K_SPACE:
                        if not simulation_done:
                            print("Starting simulations...")
                            self.run_simulations()
                            simulation_done = True

            if not simulation_done:
                self.display_forest()
                pygame.display.set_caption(f'Forest Fire Simulation - P: {self.probability:.2f}')
                self.clock.tick(60)  # Limit to 60 FPS
            else:
                self.clock.tick(1)  # After simulation, slow down the loop

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = ForestFire(min_burned_fraction=0.01)  # 1% threshold
    game.run()

