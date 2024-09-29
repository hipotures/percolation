import cupy as cp
import numpy as np
import time

class ForestFireSimulationGPU:
    def __init__(self, size=500, num_simulations=10, p_values=None):
        """
        Inicjalizacja symulacji pożaru lasu na GPU.

        :param size: Rozmiar siatki (size x size)
        :param num_simulations: Liczba symulacji do uruchomienia równolegle
        :param p_values: Tablica prawdopodobieństw P dla każdej symulacji
        """
        self.size = size
        self.num_simulations = num_simulations
        if p_values is None:
            self.p_values = cp.full((num_simulations,), 0.5, dtype=cp.float32)
        else:
            if len(p_values) != num_simulations:
                raise ValueError("Długość p_values musi być równa num_simulations")
            self.p_values = cp.array(p_values, dtype=cp.float32)
        self.initialize_grids()

    def initialize_grids(self):
        """
        Inicjalizuje siatki dla wszystkich symulacji na GPU.
        """
        # Stan: 0 = puste, 1 = drzewo, 2 = płonie, 3 = spalone
        # Używamy cp.random.rand do wektoryzowanego generowania siatek
        random_vals = cp.random.rand(self.num_simulations, self.size, self.size)
        self.grids = (random_vals < self.p_values[:, None, None]).astype(cp.int8)

        # Losowy początek płomienia dla każdej symulacji
        initial_fire_i = cp.random.randint(0, self.size, size=(self.num_simulations,))
        initial_fire_j = cp.random.randint(0, self.size, size=(self.num_simulations,))
        self.grids[cp.arange(self.num_simulations), initial_fire_i, initial_fire_j] = 2

        # Flaga, czy symulacja jest zakończona
        self.simulation_finished = cp.zeros(self.num_simulations, dtype=cp.bool_)

        # Liczba epok dla każdej symulacji
        self.current_epochs = cp.zeros(self.num_simulations, dtype=cp.int32)

    def process_epoch(self):
        """
        Przetwarza jedną epokę dla wszystkich symulacji równocześnie.
        """
        # Identyfikacja komórek płonących
        burning = self.grids == 2

        # Zmiana stanu płonących komórek na spalone
        self.grids = cp.where(burning, 3, self.grids)

        # Zwiększenie liczby epok
        self.current_epochs += 1

        # Przesunięcia siatki, aby znaleźć sąsiadów płonących komórek
        # Każdy kierunek przesunięcia
        neighbors = [
            cp.roll(burning, shift=1, axis=1),   # Góra
            cp.roll(burning, shift=-1, axis=1),  # Dół
            cp.roll(burning, shift=1, axis=2),   # Lewo
            cp.roll(burning, shift=-1, axis=2)   # Prawo
        ]

        # Znalezienie nowych komórek do zapłonu
        new_burning = cp.zeros_like(self.grids, dtype=cp.bool_)
        for neighbor in neighbors:
            new_burning |= neighbor

        # Tylko tam, gdzie jest drzewo (1), zmień na płonące (2)
        self.grids = cp.where((new_burning) & (self.grids == 1), 2, self.grids)

        # Aktualizacja flagi zakończenia symulacji
        still_burning = cp.any(self.grids == 2, axis=(1,2))
        self.simulation_finished = cp.logical_or(self.simulation_finished, ~still_burning)

    def run_simulation(self):
        """
        Uruchamia symulacje aż do zakończenia wszystkich pożarów.

        :return: Liczba epok dla każdej symulacji, czas trwania symulacji
        """
        start_time = time.time()
        while not cp.all(self.simulation_finished):
            self.process_epoch()
        end_time = time.time()
        total_time = end_time - start_time
        epochs_cpu = self.current_epochs.get()
        return epochs_cpu, total_time

    def get_final_states(self):
        """
        Zwraca liczbę komórek w każdym stanie dla każdej symulacji.

        :return: Tablica zliczeń dla stanów 0, 1, 2, 3 dla każdej symulacji
        """
        final_states = []
        grids_cpu = self.grids.get()
        for idx in range(self.num_simulations):
            unique, counts = np.unique(grids_cpu[idx], return_counts=True)
            state_count = dict(zip(unique.tolist(), counts.tolist()))
            final_states.append(state_count)
        return final_states

def run_multiple_simulations(p_values, runs=10, size=500):
    """
    Uruchamia wiele symulacji dla różnych wartości P i uśrednia wyniki.

    :param p_values: Lista lub tablica wartości P
    :param runs: Liczba powtórzeń symulacji dla każdej wartości P
    :param size: Rozmiar siatki
    :return: Średnie liczby epok i czasy dla każdej wartości P
    """
    num_ps = len(p_values)
    avg_epochs = np.zeros(num_ps)
    avg_times = np.zeros(num_ps)

    for i, p in enumerate(p_values):
        total_epochs = []
        total_time = 0.0
        for run in range(runs):
            simulation = ForestFireSimulationGPU(size=size, num_simulations=1, p_values=[p])
            epochs, elapsed_time = simulation.run_simulation()
            total_epochs.append(epochs[0])
            total_time += elapsed_time
        avg_epochs[i] = np.mean(total_epochs)
        avg_times[i] = total_time / runs
        print(f"P = {p:.4f}: Średnia epok = {avg_epochs[i]:.2f}, Średni czas = {avg_times[i]:.4f} s")
    return avg_epochs, avg_times

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
    with open(filename, 'a', encoding='utf-8') as f:
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
    with open(wynik_file, 'w', encoding='utf-8') as f:
        f.write("Wyniki symulacji pożaru lasu\n\n")

    current_p_min = p_start
    current_p_max = p_end

    for it in range(1, iterations + 1):
        print(f"\n--- Iteracja {it} ---")
        p_values = generate_p_values(current_p_min, current_p_max, steps)
        avg_epochs, avg_times = run_multiple_simulations(p_values, runs=runs_per_p, size=size)

        # Zapis wyników do pliku
        save_results(wynik_file, it, p_values, avg_times, avg_epochs)

        # Znalezienie P z maksymalnym średnim czasem
        max_time_idx = np.argmax(avg_times)
        best_p = p_values[max_time_idx]
        print(f"Najlepsze P w iteracji {it}: {best_p:.4f} z czasem {avg_times[max_time_idx]:.4f} s")

        # Ustalenie nowego przedziału wokół najlepszego P
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
