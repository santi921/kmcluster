import numpy as np
from numpy.random import uniform
import random
from copy import deepcopy
from bisect import bisect_left
k_b_ev = 8.614 * 10**-5
k_b_j = 1.38064852 * 10**-23


class rfkmc:
    def __init__(self, k_b_t=1, energy_mat=None, rate_mat=None):
        self.k_b_t = k_b_t
        self.energy_mat = energy_mat
        
        rate_mat = np.zeros((len(energy_mat), len(energy_mat[0])))
        for i in range(len(self.energy_mat)):
            for j in range(len(self.energy_mat[0])):
                if self.energy_mat[i, j] != 0:
                    rate_mat[i, j] = (self.k_b_t / (4.1357 * 10**-15)) * np.exp(
                        -(self.energy_mat[i, j] / (self.k_b_t))
                    )
        self.rate_mat = np.array(rate_mat)
        self.sum_rates = np.sum(self.rate_mat, axis=1)
        self.cum_rates = []
        #self.valid_states = []
        for i in range(len(self.rate_mat)):
            valid_states = np.where(self.rate_mat[i, :] != 0)[0]
            #self.valid_states.append(valid_states)
            self.cum_rates.append(rates_to_cum_rates(self.rate_mat[i, valid_states]))
        
        #self.cum_rates = np.array(self.cum_rates)
    
    
    def get_rates_total(self, state_index):
        return self.rate_mat[state_index, :]
    

    def call(self, state_index, rand_state, neg_log_time_sample):
        
        sum_rates = self.sum_rates[state_index]
        rates_cum = self.cum_rates[state_index]
        
        if sum_rates == 0: return -1, 10e6
        rand_state = rand_state * sum_rates
        
        return_state = bisect_left(rates_cum, rand_state)
        time_to_transition = neg_log_time_sample / sum_rates
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
    return np.array(rates_cum)
