import random
import pygame
import sys

# Initialize Pygame
pygame.init()

class ForestFire:
    def __init__(self):
        self.size = 500  # Increase size to 500x500
        self.grid = [[' ' for _ in range(self.size)] for _ in range(self.size)]
        self.probability = 0.50
        self.cell_size = 1  # Size of each cell in pixels for larger grid
        self.screen = pygame.display.set_mode((self.size * self.cell_size, self.size * self.cell_size))
        self.epoch_speed = 1  # Epoch speed in epochs per second
        self.clock = pygame.time.Clock()
        self.mutation_position = None
        self.current_epoch = 0  # Initialize current epoch
        self.red_objects = []  # List to keep track of red objects

        # Initialize the forest
        self.generate_forest()

    def generate_forest(self):
        for i in range(self.size):
            for j in range(self.size):
                if random.random() < self.probability:
                    self.grid[i][j] = (0, 255, 0)  # Green for trees
                else:
                    self.grid[i][j] = (255, 255, 255)  # White for empty space

        # Generate mutation
        self.mutation_position = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
        self.grid[self.mutation_position[0]][self.mutation_position[1]] = (255, 0, 0)  # Red for mutation
        self.red_objects = [self.mutation_position]  # Initialize red objects list with the mutation

    def display_forest(self):
        for i in range(self.size):
            for j in range(self.size):
                color = self.grid[i][j]
                pygame.draw.rect(self.screen, color, (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size))
        pygame.display.flip()

    def reset_forest(self):
        self.grid = [[' ' for _ in range(self.size)] for _ in range(self.size)]
        self.red_objects = []  # Reset red objects list
        self.generate_forest()
        self.current_epoch = 0  # Reset epoch on forest reset

    def change_probability(self, change):
        self.probability = max(0, min(1, self.probability + change))

    def process_epoch(self):
        new_red_objects = set()  # Use a set to prevent duplicates
        for (i, j) in self.red_objects:  # Iterate only over red objects
            # Check neighbors with periodic boundary conditions (PBC)
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, Down, Left, Right
                ni = (i + di) % self.size  # Wrap around vertically
                nj = (j + dj) % self.size  # Wrap around horizontally
                if self.grid[ni][nj] == (0, 255, 0):  # If neighbor is green
                    new_red_objects.add((ni, nj))  # Add new red object
                    self.grid[ni][nj] = (255, 0, 0)  # Mark as red immediately
            self.grid[i][j] = (0, 0, 0)  # Change self to black
        self.red_objects = list(new_red_objects)  # Update red objects list

    def run(self):
        while True:
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
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        self.change_epoch_speed(1)
                    elif event.key == pygame.K_MINUS:
                        self.change_epoch_speed(-1)
            self.process_epoch()  # Process the current epoch
            self.display_forest()
            red_count = len(self.red_objects)  # Count red objects
            if red_count == 0:  # Check if there are no red cells
                pygame.display.set_caption('RedMuts - P: {:.2f} - E: {} - R: {} (finished)'.format(self.probability, self.current_epoch, red_count))
                continue  # Wait for user intervention
            self.current_epoch += 1  # Increment epoch by 1
            pygame.display.set_caption('RedMuts - P: {:.2f} - E: {} - R: {}'.format(self.probability, self.current_epoch, red_count))

if __name__ == "__main__":
    game = ForestFire()
    game.run()