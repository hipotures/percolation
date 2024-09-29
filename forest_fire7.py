import cupy as cp
import random
import time
import numpy as np

class ForestFireSimulationGPU:
    def __init__(self, size=500, probability=0.30):
        r"""
        Inicjalizacja symulacji pożaru lasu na GPU.

        :param size: Rozmiar siatki (size x size)
        :param probability: Prawdopodobieństwo \( P \) dla danej symulacji
        """
        self.size = size
        self.probability = probability
        self.current_epochs = cp.int32(0)
        self.simulation_finished = False
        self.grid = cp.zeros((self.size, self.size), dtype=cp.int8)
        self.burning_positions = []
        self.initialize_grid()

    def initialize_grid(self):
        """
        Inicjalizuje siatkę dla symulacji na GPU.
        """
        # 0: empty, 1: tree, 2: burning, 3: burned
        self.grid = cp.random.choice(
            [1, 0],
            size=(self.size, self.size),
            p=[self.probability, 1 - self.probability]
        ).astype(cp.int8)

        # Wybór losowego punktu inicjalnego pożaru
        i = random.randint(0, self.size - 1)
        j = random.randint(0, self.size - 1)
        self.grid[i, j] = 2  # 2: burning
        self.burning_positions = [(i, j)]

    def process_epoch(self):
        """
        Przetwarza jedną epokę dla symulacji.
        """
        new_burning_positions = []

        for pos in self.burning_positions:
            i, j = pos
            # Zmiana stanu na spalone
            self.grid[i, j] = 3  # 3: burned

            # Sprawdzanie sąsiadów
            neighbors = [
                (i-1, j), (i+1, j),
                (i, j-1), (i, j+1)
            ]
            for ni, nj in neighbors:
                if 0 <= ni < self.size and 0 <= nj < self.size:
                    if self.grid[ni, nj] == 1:
                        self.grid[ni, nj] = 2  # 2: burning
                        new_burning_positions.append((ni, nj))

        # Aktualizacja liczby epok
        self.current_epochs += 1

        # Aktualizacja pozycji płonących drzew
        self.burning_positions = new_burning_positions

        # Sprawdzenie, czy symulacja się zakończyła
        if not self.burning_positions:
            self.simulation_finished = True

    def run_simulation(self):
        """
        Uruchamia symulację aż do zakończenia wszystkich pożarów.

        :return: Liczba epok symulacji
        """
        while not self.simulation_finished:
            self.process_epoch()
        return self.current_epochs.item()

def run_single_simulation(p, size=500):
    """
    Uruchamia pojedynczą symulację dla danego P.

    :param p: Prawdopodobieństwo
    :param size: Rozmiar siatki
    :return: Liczba epok symulacji
    """
    simulation = ForestFireSimulationGPU(size=size, probability=p)
    epochs = simulation.run_simulation()
    return epochs

def run_multiple_simulations(p, runs=10, size=500):
    """
    Uruchamia wiele symulacji dla danego P i uśrednia wyniki.

    :param p: Prawdopodobieństwo
    :param runs: Liczba powtórzeń symulacji
    :param size: Rozmiar siatki
    :return: Średni czas [s], Średnia liczba epok
    """
    epochs_list = []
    start_time = time.time()
    for _ in range(runs):
        epochs = run_single_simulation(p, size=size)
        epochs_list.append(epochs)
    end_time = time.time()
    avg_time = (end_time - start_time) / runs
    avg_epochs = np.mean(epochs_list)
    return avg_time, avg_epochs

def generate_p_values(p_min, p_max, steps):
    """
    Generuje listę wartości P w danym przedziale.

    :param p_min: Minimalna wartość P
    :param p_max: Maksymalna wartość P
    :param steps: Liczba kroków
    :return: Lista wartości P
    """
    return np.linspace(p_min, p_max, steps)

def save_results(filename, iteration, p_values, avg_times, avg_epochs):
    """
    Zapisuje wyniki do pliku.

    :param filename: Nazwa pliku
    :param iteration: Numer iteracji
    :param p_values: Lista wartości P
    :param avg_times: Lista średnich czasów
    :param avg_epochs: Lista średnich epok
    """
    with open(filename, 'a') as f:
        f.write(f"Iteracja {iteration}:\n")
        f.write("P\tŚredni czas [s]\tŚrednia liczba epok\n")
        for p, t, e in zip(p_values, avg_times, avg_epochs):
            f.write(f"{p:.4f}\t{t:.4f}\t{e:.2f}\n")
        f.write("\n")

def main():
    # Parametry początkowe
    p_start = 0.1
    p_end = 1.0
    steps = 10
    runs_per_p = 10
    iterations = 5  # Liczba iteracji podziału
    size = 1000
    wynik_file = "wyniki_symulacji.txt"

    # Inicjalizacja pliku wynikowego
    with open(wynik_file, 'w') as f:
        f.write("Wyniki symulacji pożaru lasu\n\n")

    current_p_min = p_start
    current_p_max = p_end

    for it in range(1, iterations + 1):
        print(f"\n--- Iteracja {it} ---")
        p_values = generate_p_values(current_p_min, current_p_max, steps)
        avg_times = []
        avg_epochs = []

        for p in p_values:
            print(f"Symulacja dla P = {p:.4f}...", end='', flush=True)
            avg_time, avg_epoch = run_multiple_simulations(p, runs=runs_per_p, size=size)
            avg_times.append(avg_time)
            avg_epochs.append(avg_epoch)
            print(f" Średni czas = {avg_time:.4f} s, Średnia epok = {avg_epoch:.2f}")

        # Zapis wyników do pliku
        save_results(wynik_file, it, p_values, avg_times, avg_epochs)

        # Znalezienie P z maksymalnym średnim czasem
        max_time_idx = np.argmax(avg_times)
        best_p = p_values[max_time_idx]
        print(f"Najlepsze P w iteracji {it}: {best_p:.4f} z czasem {avg_times[max_time_idx]:.4f} s")

        # Ustalenie nowego przedziału wokół najlepszego P
        # Zakładamy krok jako (current_p_max - current_p_min) / (steps - 1)
        step_size = (current_p_max - current_p_min) / (steps - 1)
        new_step = step_size / steps  # Nowy mniejszy krok

        new_p_min = max(best_p - new_step, p_start)
        new_p_max = min(best_p + new_step, p_end)

        print(f"Nowy przedział P: {new_p_min:.4f} - {new_p_max:.4f}")

        current_p_min = new_p_min
        current_p_max = new_p_max

    print(f"\nWszystkie wyniki zapisano w pliku {wynik_file}")

if __name__ == "__main__":
    main()

