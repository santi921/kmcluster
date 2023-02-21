import numpy as np
import random
from copy import deepcopy

k_b_ev = 8.614 * 10**-5
k_b_j = 1.38064852 * 10**-23


class rfkmc:
    def __init__(self, k_b_t=1):
        self.k_b_t = k_b_t

    def call(self, energies_total):
        # filter all states with energy = 0
        # energies_total = energies_total[energies_total != 0]
        rate_list = []
        for i in range(len(energies_total)):
            if energies_total[i] == 0:
                rate_list.append(0)
            else:
                rate = (self.k_b_t / (4.1357 * 10**-15)) * np.exp(
                    -(energies_total[i] / (self.k_b_t))
                )
                rate_list.append(rate)

        rates_total = np.array(rate_list)
        sum_rates = np.sum(rates_total)
        if sum_rates == 0:
            return -1, 10e6
        rates_cum = rates_to_cum_rates(rates_total)

        # randomly select a number between 0 and rates_rotal
        rand_state = random.uniform(0, sum_rates)
        rand_time = random.uniform(0, 1)

        # find the index of the first rate that is greater than or equal to rand
        # this is the index of the state that the particle will transition to
        return_state = np.argmax(rates_cum >= rand_state)
        time_to_transition = -np.log(rand_time) / sum_rates
        # print(time_to_transition)
        return return_state, time_to_transition


class rkmc:
    def __init__(self, r_0, k_b_t=1):
        self.rejection = r_0
        self.k_b_t = k_b_t

    def call(self, energies_total):
        rate_list = []
        for i in range(len(energies_total)):
            if energies_total[i] == 0:
                rate_list.append(0)
            else:
                rate = (self.k_b_t / (4.1357 * 10**-15)) * np.exp(
                    -(energies_total[i] / (self.k_b_t))
                )
                rate_list.append(rate)

        rates_total = np.array(rate_list)
        sum_rates = np.sum(rate_list)
        rates_cum = rates_to_cum_rates(rate_list)
        if sum_rates == 0:
            return -1, 10e6

        while True:
            # find the index of the first probability that is greater than or equal to rand
            # this is the index of the state that the particle will transition to
            # randomly select a number between 0 and sum_probs
            rand_state = random.uniform(0, sum_rates)
            rand_time = random.uniform(0, 1)

            return_state = np.argmax(rates_cum >= rand_state)
            rate_selected = rates_total[return_state]

            # draw 1 with probability rate_selected/r_0
            n_k = int(len(rates_total))

            time_to_transition = np.log(1 / rand_time) / (n_k * self.rejection)

            if rand_time <= rate_selected / self.rejection:
                break

        return return_state, time_to_transition


def rates_to_cum_rates(rates):
    rates_cum = []
    rates_cum.append(rates[0])
    for i in range(1, len(rates)):
        rates_cum.append(rates_cum[i - 1] + rates[i])
    return rates_cum
