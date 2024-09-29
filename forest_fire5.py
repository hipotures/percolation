import cupy as cp
import random
import time

class ForestFireSimulationGPU:
    def __init__(self, size=500, probabilities=[0.30, 0.40, 0.50, 0.60, 0.70]):
        """
        Inicjalizacja symulacji pożaru lasu na GPU.

        :param size: Rozmiar siatki (size x size)
        :param probabilities: Lista prawdopodobieństw, dla których przeprowadzimy symulację
        """
        self.size = size
        self.probabilities = probabilities
        self.num_simulations = len(probabilities)
        self.current_epochs = cp.zeros(self.num_simulations, dtype=cp.int32)
        self.simulation_finished = cp.zeros(self.num_simulations, dtype=bool)
        self.grids = cp.zeros((self.num_simulations, self.size, self.size), dtype=cp.int8)
        self.burning_positions = [ [] for _ in range(self.num_simulations) ]
        self.initialize_grids()

    def initialize_grids(self):
        """
        Inicjalizuje siatki dla wszystkich symulacji na GPU.
        """
        for idx, p in enumerate(self.probabilities):
            # 0: empty, 1: tree, 2: burning, 3: burned
            # Generowanie lasu
            random_grid = cp.random.choice(
                [1, 0], 
                size=(self.size, self.size), 
                p=[p, 1 - p]
            ).astype(cp.int8)
            self.grids[idx] = random_grid

            # Wybór losowego punktu inicjalnego pożaru
            i = random.randint(0, self.size - 1)
            j = random.randint(0, self.size - 1)
            self.grids[idx, i, j] = 2  # 2: burning
            self.burning_positions[idx] = [(i, j)]

    def process_epoch(self):
        """
        Przetwarza jedną epokę dla wszystkich symulacji.
        """
        new_burning_positions = [ [] for _ in range(self.num_simulations) ]

        for idx in range(self.num_simulations):
            if self.simulation_finished[idx]:
                continue  # Pomiń zakończone symulacje

            for pos in self.burning_positions[idx]:
                i, j = pos
                # Zmiana stanu na spalone
                self.grids[idx, i, j] = 3  # 3: burned

                # Sprawdzanie sąsiadów
                neighbors = [
                    (i-1, j), (i+1, j),
                    (i, j-1), (i, j+1)
                ]
                for ni, nj in neighbors:
                    if 0 <= ni < self.size and 0 <= nj < self.size:
                        if self.grids[idx, ni, nj] == 1:
                            self.grids[idx, ni, nj] = 2  # 2: burning
                            new_burning_positions[idx].append((ni, nj))

            # Aktualizacja liczby epok
            self.current_epochs[idx] += 1

            # Sprawdzenie, czy symulacja się zakończyła
            if not new_burning_positions[idx]:
                self.simulation_finished[idx] = True

        self.burning_positions = new_burning_positions

    def run_simulation(self):
        """
        Uruchamia symulacje aż do zakończenia wszystkich pożarów.
        """
        start_time = time.time()
        while not cp.all(self.simulation_finished):
            self.process_epoch()
        end_time = time.time()
        total_time = end_time - start_time

        # Przeniesienie wyników z GPU do CPU
        epochs_cpu = self.current_epochs.get()
        times_cpu = total_time  # Czas całkowity dla wszystkich symulacji

        return epochs_cpu, times_cpu

    def get_final_states(self):
        """
        Zwraca liczbę komórek w każdym stanie dla każdej symulacji.

        :return: Lista słowników z liczbą komórek w stanach 0, 1, 2, 3 dla każdej symulacji
        """
        final_states = []
        grids_cpu = self.grids.get()
        for idx in range(self.num_simulations):
            unique, counts = cp.unique(self.grids[idx], return_counts=True)
            state_count = dict(zip(unique.tolist(), counts.tolist()))
            final_states.append(state_count)
        return final_states

def main():
    # Definicja prawdopodobieństw
    probabilities = [0.60, 0.61, 0.62, 0.63, 0.64, 0.65]

    # Rozmiar siatki
    size = 1000

    # Inicjalizacja symulacji
    simulation = ForestFireSimulationGPU(size=size, probabilities=probabilities)

    # Uruchomienie symulacji
    print("Rozpoczynanie symulacji na GPU...")
    epochs, elapsed_time = simulation.run_simulation()
    print(f"Symulacja zakończona w {elapsed_time:.4f} sekund.")

    # Wyświetlenie wyników
    for idx, p in enumerate(probabilities):
        print(f"\nPrawdopodobieństwo P = {p:.2f}")
        print(f"Liczba epok: {epochs[idx]}")
        final_states = simulation.get_final_states()[idx]
        print("Stan siatki po zakończeniu:")
        print(f"Pusta (0): {final_states.get(0, 0)}")
        print(f"Drzewo (1): {final_states.get(1, 0)}")
        print(f"Burning (2): {final_states.get(2, 0)}")
        print(f"Spalone (3): {final_states.get(3, 0)}")

if __name__ == "__main__":
    main()

