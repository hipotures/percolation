import numpy as np
import random
from numba import njit
import time

@njit
def process_epoch_numba(grid, burning_positions, size):
    new_burning = []
    for idx in range(len(burning_positions)):
        i, j = burning_positions[idx]
        grid[i, j] = 3  # Spalone
        # Sprawdzenie sąsiadów
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < size and 0 <= nj < size:
                if grid[ni, nj] == 1:
                    grid[ni, nj] = 2  # Płonące
                    new_burning.append((ni, nj))
    return new_burning

class ForestFireSimulation:
    def __init__(self, size=1000, probability=0.50):
        """
        Inicjalizacja symulacji pożaru lasu.

        :param size: Rozmiar siatki (size x size)
        :param probability: Prawdopodobieństwo, że komórka będzie drzewem
        """
        self.size = size
        self.probability = probability
        self.current_epoch = 0
        self.red_objects = []
        self.grid = np.zeros((self.size, self.size), dtype=np.int8)
        self.generate_forest()

    def generate_forest(self):
        """
        Generuje las z drzewami na podstawie podanego prawdopodobieństwa.
        """
        # 0: empty, 1: tree, 2: burning, 3: burned
        self.grid = np.random.choice(
            [1, 0], 
            size=(self.size, self.size), 
            p=[self.probability, 1 - self.probability]
        ).astype(np.int8)
        # Wybór losowego punktu inicjalnego pożaru
        mutation_position = (
            random.randint(0, self.size - 1), 
            random.randint(0, self.size - 1)
        )
        self.grid[mutation_position] = 2  # 2: burning
        self.red_objects = [mutation_position]

    def run_simulation(self):
        """
        Uruchamia symulację aż do całkowitego zgaszenia pożaru.
        """
        start_time = time.time()
        while self.red_objects:
            self.red_objects = process_epoch_numba(self.grid, self.red_objects, self.size)
            self.current_epoch += 1
        end_time = time.time()
        total_time = end_time - start_time
        return self.current_epoch, total_time

    def get_grid_state_counts(self):
        """
        Zwraca liczbę komórek w każdym stanie.

        :return: Słownik z liczbą komórek w stanach 0, 1, 2, 3
        """
        unique, counts = np.unique(self.grid, return_counts=True)
        return dict(zip(unique, counts))

def main():
    # Przykładowe prawdopodobieństwa do testowania
    probabilities = [0.60, 0.61, 0.62, 0.63, 0.64, 0.65]

    # Rozmiar siatki
    size = 1000

    # Uruchomienie symulacji dla różnych prawdopodobieństw
    for p in probabilities:
        print(f'\nSymulacja dla prawdopodobieństwa P = {p:.2f}')
        simulation = ForestFireSimulation(size=size, probability=p)
        epochs, elapsed_time = simulation.run_simulation()
        print(f'Liczba epok: {epochs}')
        print(f'Czas symulacji: {elapsed_time:.4f} sekund')
        # Opcjonalnie, wyświetl liczby komórek w różnych stanach po zakończeniu
        state_counts = simulation.get_grid_state_counts()
        print('Stan siatki po zakończeniu:')
        print(f'Pusta (0): {state_counts.get(0, 0)}')
        print(f'Drzewo (1): {state_counts.get(1, 0)}')
        print(f'Burning (2): {state_counts.get(2, 0)}')
        print(f'Spalone (3): {state_counts.get(3, 0)}')

if __name__ == "__main__":
    main()

