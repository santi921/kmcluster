import numpy as np
import random
from bisect import bisect_left
import numba, brisk
from numba import jit
#k_b_ev = 8.614 * 10**-5
#k_b_j = 1.38064852 * 10**-23



class rfkmc:
    def __init__(self, k_b_t=1, energy_mat=None, rate_mat=None):
        self.k_b_t = k_b_t
        self.energy_mat = energy_mat
        
        rate_mat = np.zeros((len(energy_mat), len(energy_mat)))
        for i in range(len(self.energy_mat)):
            for j in range(len(self.energy_mat[0])):
                if self.energy_mat[i, j] != 0:
                    rate_mat[i, j] = (self.k_b_t / (4.1357 * 10**-15)) * np.exp(
                        -(self.energy_mat[i, j] / (self.k_b_t))
                    )
        self.rate_mat = np.array(rate_mat)
        self.sum_rates = np.sum(self.rate_mat, axis=1)
        
        
        self.cum_rates = []
        for i in range(len(self.rate_mat)):
            self.cum_rates.append(rates_to_cum_rates(self.rate_mat[i]))
        
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


    def get_sum_cum_rates(self, ind): 
        return self.sum_rates[ind], self.cum_rates[ind]
    
    def inst_ret_vars(self, n): 
        return_states = np.zeros(n)
        time_to_transitions = np.zeros(n)
        return return_states, time_to_transitions
    

    def call_batched(self, state, rand_states, neg_log_time_samples):
        '''
            given a starting state, a list of random states, and a list of negative log time samples
            return a list of states and a list of times to transition
        '''
        return_states, time_to_transitions = self.inst_ret_vars(len(neg_log_time_samples))

        last_state = int(state)
        
        for i in range(len(rand_states)): 
            #sum_rates, rates_cum = self.get_sum_cum_rates(last_state)
            sum_rates, rates_cum = self.sum_rates[last_state], self.cum_rates[last_state]
            if sum_rates == 0:
                print("sheesh")
                return_states[i] = last_state
                time_to_transitions[i] = multiply(10e6, i)
                continue

            rand_state = multiply(rand_states[i], sum_rates) # hot
            last_state = int(brisk.bisect_left(rates_cum, rand_state))       
            
            return_states[i] = last_state

            if i == 0: 
                #time_to_transitions[i] = div(neg_log_time_samples[i], sum_rates)
                time_to_transitions[i] = div_w_index(neg_log_time_samples, sum_rates, i)
            else:
                time_to_transitions[i] = div_and_sum(neg_log_time_samples[i], sum_rates, time_to_transitions[i - 1])  # expensive

        return return_states, time_to_transitions
        #return compile_base(state, rand_states, neg_log_time_samples, self.sum_rates, self.cum_rates)

@jit(nopython=True)
def multiply(a, b): 
    return a * b

@jit(nopython=True)
def div(a, b): 
    return a / b

@jit(nopython=True)
def div_and_sum(a, b, c): 
    return a / b + c

@jit(nopython=True)
def div_w_index(a, b, i):
    return a[i] / b

@jit(nopython=True)
def div_w_index_and_sum(a, b, i, c):
    return a[i] / b + c

# deprecated
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
    #print(len(rates))
    rates_cum = [rates[0]]
    for i in range(1, len(rates)):
        rates_cum.append(rates_cum[i - 1] + rates[i])
    
    return np.array(rates_cum)
