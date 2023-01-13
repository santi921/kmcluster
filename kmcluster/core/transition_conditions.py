import numpy as np 
import random 
from copy import deepcopy

class rfkmc:

    def __init__(self): 
        pass

    def call(self, rates_total):
        
        sum_rates = np.sum(rates_total)
        if sum_rates == 0:
            return -1, 10E6
        
        rates_cum =  rates_to_cum_rates(rates_total)
    
        # randomly select a number between 0 and rates_rotal
        rand_state = random.uniform(0, sum_rates)
        rand_time = random.uniform(0, 1)

        # find the index of the first rate that is greater than or equal to rand
        # this is the index of the state that the particle will transition to
        return_state = np.argmax(rates_cum >= rand_state)
        time_to_transition = -np.log(rand_time)/sum_rates

        return return_state, time_to_transition


class rkmc:

    def __init__(self, r_0):
        self.rejection = r_0

    def call(self, rates_total):
        sum_rates = np.sum(rates_total) 
        rates_cum =  rates_to_cum_rates(rates_total)
        if sum_rates == 0:
            return -1, 10E6
        
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

            time_to_transition = np.log(1/rand_time) / (n_k * self.rejection)
            
            if rand_time <= rate_selected/self.rejection:
                break
        
        return return_state, time_to_transition


def rates_to_cum_rates(rates):
    rates_cum = []
    rates_cum.append(rates[0])
    for i in range(1, len(rates)):
        rates_cum.append(rates_cum[i-1] + rates[i])   
    return rates_cum