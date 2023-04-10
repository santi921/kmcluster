import numpy as np
from kmcluster.core.trajectory import trajectory, trajectory_minimum
from numpy import random


class initializer:
    def __init__(self):
        pass

    def get_init_populations(self):
        pass


# randomly initialize population
class random_init(initializer):
    def __init__(self, size, n_states):
        super().__init__()
        self.size = size
        self.n_states = n_states
        self.population = None

    def get_init_populations(self):
        if self.population is None:
            # randomly assign size number of members to n_states number of states
            population = [0 for i in range(self.n_states)]
            for i in range(self.size):
                ind = random.randint(0, self.n_states)
                population[ind] = population[ind] + 1
            self.population = population

        return self.population


# initialize population based on boltzman distribution
class boltz(initializer):
    def __init__(self, size, T, energies):
        super().__init__()
        self.size = size
        self.T = T
        self.energies = energies
        # make energies relative
        self.energies_relative = energies - np.max(energies)
        self.probabilities = [np.exp(-e / T) for e in self.energies_relative]
        self.sum_probabilities = sum(self.probabilities)
        self.probabilities_normalized = [
            p / self.sum_probabilities for p in self.probabilities
        ]
        self.population = None

    def get_init_populations(self):
        # randomly assign population baseed on boltzman population
        if self.population is None:
            population = [0 for i in range(len(self.energies))]

            for i in range(self.size):
                ind = random.choice(
                    range(len(self.probabilities)), p=self.probabilities_normalized
                )
                population[ind] = population[ind] + 1

            self.population = population

        return self.population


# initalize on global min structure only
class global_minimum_only(initializer):
    def __init__(self, size, energies, n_states=None):
        super().__init__()
        self.energies = energies
        self.size = size
        self.population = None
        if n_states is None:
            self.n_states = len(energies)
        else:
            self.n_states = n_states

    def get_init_populations(self):
        # find lowest energy and put all pop into it
        if self.population is None:
            population = [0 for i in range(self.n_states)]
            min_ind = np.argmin(self.energies)
            population[min_ind] = self.size
            self.population = population

        return self.population


# user defined proportions in population
class selected(initializer):
    def __init__(self, size, selected_proportions, n_states):
        """
        fills list of populations with selection_proportions
        """
        super().__init__()
        self.size = size
        self.selected_proportions = selected_proportions
        self.n_states = n_states
        self.population = None

    def get_init_populations(self):
        if self.population is None:
            population = [0 for i in range(self.n_states)]
            for k, v in self.selected_proportions.items():
                # check that no value v is greater than 1
                assert v <= 1, "selected proportions must be less than 1"
                population[k] = int(self.size * v)
            self.population = population

        return self.population


# convert population list to list of trajectories
def population_ind_to_trajectories(population_list):
    as_indicies = []
    for ind, i in enumerate(population_list):
        if i > 0:
            for j in range(i):
                as_indicies.append(ind)
    return [trajectory(i) for i in as_indicies]


def population_ind_to_minimum_trajectories(population_list):
    as_indicies = []
    for ind, i in enumerate(population_list):
        if i > 0:
            for j in range(i):
                as_indicies.append(ind)
    return [trajectory_minimum(i, 0) for i in as_indicies]