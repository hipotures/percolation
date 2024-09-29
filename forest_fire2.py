import random
import numpy as np
import pygame
import sys
from numba import njit

# Initialize Pygame
pygame.init()

class ForestFire:
    def __init__(self):
        self.size = 1000
        self.probability = 0.50
        self.cell_size = 1
        self.screen = pygame.display.set_mode((self.size * self.cell_size, self.size * self.cell_size))
        pygame.display.set_caption('RedMuts')
        self.clock = pygame.time.Clock()
        self.current_epoch = 0
        self.red_objects = []
        self.generate_forest()

    def generate_forest(self):
        # 0: empty, 1: tree, 2: burning, 3: burned
        self.grid = np.random.choice([1, 0], size=(self.size, self.size), p=[self.probability, 1 - self.probability]).astype(np.int8)
        mutation_position = (np.random.randint(0, self.size), np.random.randint(0, self.size))
        self.grid[mutation_position] = 2  # Burning
        self.red_objects = [mutation_position]

    def display_forest(self):
        color_map = {
            0: (255, 255, 255),  # Empty
            1: (0, 255, 0),      # Tree
            2: (255, 0, 0),      # Burning
            3: (0, 0, 0)         # Burned
        }
        # Create RGB array
        rgb_array = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        for state, color in color_map.items():
            mask = self.grid == state
            rgb_array[mask] = color
        # Update Pygame surface
        pygame.surfarray.blit_array(self.screen, rgb_array)
        pygame.display.flip()

    def reset_forest(self):
        self.generate_forest()
        self.current_epoch = 0

    def change_probability(self, change):
        self.probability = max(0, min(1, self.probability + change))
        self.reset_forest()

    @staticmethod
    @njit
    def process_epoch_numba(grid, burning_positions, size):
        new_burning = []
        for pos in burning_positions:
            i, j = pos
            grid[i, j] = 3  # Burned
            # Check neighbors
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < size and 0 <= nj < size:
                    if grid[ni, nj] == 1:
                        grid[ni, nj] = 2  # Burning
                        new_burning.append((ni, nj))
        return new_burning

    def run(self):
        while True:
            self.clock.tick(60)  # Limit to 60 FPS
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    elif event.key == pygame.K_0:
                        self.reset_forest()
                    elif event.key == pygame.K_UP:
                        self.change_probability(0.01)
                    elif event.key == pygame.K_DOWN:
                        self.change_probability(-0.01)
                    elif event.key == pygame.K_LEFT:
                        self.change_probability(-0.10)
                    elif event.key == pygame.K_RIGHT:
                        self.change_probability(0.10)
                    # Dodaj inne klawisze według potrzeby

            if self.red_objects:
                self.red_objects = self.process_epoch_numba(self.grid, self.red_objects, self.size)
                self.current_epoch += 1
                pygame.display.set_caption(f'RedMuts - P: {self.probability:.2f} - E: {self.current_epoch} - R: {len(self.red_objects)}')
            else:
                pygame.display.set_caption(f'RedMuts - P: {self.probability:.2f} - E: {self.current_epoch} - R: 0 (finished)')

            self.display_forest()

if __name__ == "__main__":
    game = ForestFire()
    game.run()

