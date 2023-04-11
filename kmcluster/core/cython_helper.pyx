import cython
#from cython import lib
import numpy as np
cimport numpy as np# import cpython.lib 
DTYPE = np.float
ctypedef np.float_t DTYPE_t

def cython_helper(
        int state, 
        np.ndarray[DTYPE_t, ndim=2] sum_rates, 
        np.ndarray[DTYPE_t, ndim=2] cum_rates, 
        np.ndarray[DTYPE_t, ndim=1] rand_states, 
        np.ndarray[DTYPE_t, ndim=1] neg_log_time_samples):
        """
        function in cython that takes a starting state, a list of random states, and a list of negative log time samples
        and returns a list of states and a list of times to transition using the rfkmc algorithm 
        """
        cdef int n = len(neg_log_time_samples)
        cdef np.ndarray[DTYPE_t, ndim=1] return_states = np.zeros(n, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1] time_to_transitions = np.zeros(n, dtype=DTYPE)

        cdef int last_state = state
        cdef float rand_state


        for i in range(n): 
            #sum_rates, rates_cum = self.get_sum_cum_rates(last_state)
            #cdef np.ndarray[DTYPE_t, ndim=1] sum_rates = sum_rates[last_state]
            #cdef np.ndarray[DTYPE_t, ndim=1] rates_cum =  cum_rates[last_state]

            if sum_rates == 0:
                
                return_states[i] = last_state
                time_to_transitions[i] = 10e6 * i
                continue

            rand_state = rand_states[i] * sum_rates[last_state]
            last_state = lib.bisect_left(cum_rates[last_state], rand_state)      
            
            return_states[i] = last_state

            if i == 0: 
                #time_to_transitions[i] = div_w_index(neg_log_time_samples, sum_rates, i)
                time_to_transitions[i] = neg_log_time_samples[i] / sum_rates
            else:
                #time_to_transitions[i] = div_and_sum(neg_log_time_samples[i], sum_rates, time_to_transitions[i - 1])  # expensive
                time_to_transitions[i] = (neg_log_time_samples[i] / sum_rates) + time_to_transitions[i - 1]
        return return_states, time_to_transitions
