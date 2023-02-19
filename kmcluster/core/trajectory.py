import bisect 

class trajectory():
    def __init__(self, init_state, init_time=None, init_history=None):
        self.states = [init_state]
        if init_time is None:
            self.transition_times = [0]        
        else:
            self.transition_times = [init_time]

        if init_history is not None:
            self.states = init_history[0]
            self.transition_times = init_history[1]
    
    
    def draw_new_state(self, rates_from_i, draw_crit):
        # get rates of transition from i to j in probabilities matrix 
        new_state, time = draw_crit.call(rates_from_i)
        return new_state, time


    def get_history_as_dict(self):
        ret_dict = {}
        for i, state in enumerate(self.states):
            ret_dict[self.transition_times[i]] = state
        return ret_dict


    def add_new_state(self, new_state, time_transition): 
        #append state to states
        self.states.append(new_state)
        self.transition_times.append(self.transition_times[-1] + time_transition)


    def step(self, rates_from_i, draw_crit, time_stop = 10e9): 
        
        if time_stop > 0:
            # check that time of last state
            last_transition = self.last_time()
            if last_transition > time_stop:
                return 

        new_state, time_to_transition = self.draw_new_state(rates_from_i, draw_crit)
        
        if new_state == -1: # checks if rates out of a state are 0 
            new_state = self.last_state()

        self.add_new_state(
            new_state, 
            time_to_transition)
        
        if time_to_transition < 10**-15:
            return 1
        else:
            return 0
        #    print("warning: time step is less than 10^-15\n")
    

    def get_history(self):
        return self.states, self.transition_times


    def last_state(self):
        return self.states[-1]


    def last_time(self): 
        return self.transition_times[-1]


    def state_at_time(self, time):
        """
        given a trajectory return what state it's in a time t
        Takes 
            trajectory: trajectory object
            time(float): time to get state at
        Returns
            index(int): of state
        """
        # get index of time
        index = bisect.bisect_right(self.transition_times, time)
        return self.states[index-1]