import random
import pygame
import sys
import copy

# Initialize Pygame
pygame.init()

class ForestFire:
    def __init__(self):
        self.size = 100  # Grid size (100x100)
        self.probability = 0.60  # Probability of a cell being a tree
        self.cell_size = 5  # Size of each cell in pixels for better visibility
        self.screen = pygame.display.set_mode((self.size * self.cell_size, self.size * self.cell_size))
        pygame.display.set_caption('Forest Fire Simulation')
        self.clock = pygame.time.Clock()
        self.best_point = None
        self.best_epoch = 0
        self.best_grid = None

        # Initialize the forest
        self.generate_forest()

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
        for i in range(self.size):
            for j in range(self.size):
                color = self.grid[i][j]
                pygame.draw.rect(self.screen, color, (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size))
        pygame.display.flip()

    def reset_forest(self):
        """Resets the forest to its initial state."""
        self.generate_forest()
        self.best_point = None
        self.best_epoch = 0
        self.best_grid = None
        pygame.display.set_caption('Forest Fire Simulation')

    def simulate_fire(self, start_point):
        """
        Simulates the forest fire starting from the given point.

        :param start_point: Tuple (i, j) coordinates of the ignition point
        :return: Number of epochs the fire lasted
        """
        # Deep copy the grid to simulate on
        grid_sim = copy.deepcopy(self.grid)
        red_objects = [start_point]  # List to keep track of burning trees
        grid_sim[start_point[0]][start_point[1]] = (255, 0, 0)  # Set ignition point to red
        epoch = 0

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
            red_objects = list(new_red_objects)
            epoch += 1

        return epoch, grid_sim

    def run_simulations(self):
        """
        Runs simulations for each tree in the forest to find the best ignition point.
        """
        total_trees = len(self.tree_positions)
        print(f"Total number of trees to simulate: {total_trees}")
        for idx, point in enumerate(self.tree_positions):
            epoch, final_grid = self.simulate_fire(point)
            if epoch > self.best_epoch:
                self.best_epoch = epoch
                self.best_point = point
                self.best_grid = final_grid
                print(f"New best point found at {point} with {epoch} epochs.")
            if (idx + 1) % 100 == 0:
                print(f"Simulated {idx + 1} / {total_trees} trees.")

        print(f"\nSimulation complete. Best ignition point: {self.best_point} with {self.best_epoch} epochs.")
        self.visualize_best_simulation()

    def visualize_best_simulation(self):
        """
        Visualizes the forest after the best simulation.
        """
        if self.best_grid is None:
            print("No simulation data to display.")
            return

        # Update the grid to the best simulation
        self.grid = self.best_grid
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
    game = ForestFire()
    game.run()

