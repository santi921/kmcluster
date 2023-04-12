import numpy as np 
from bisect import bisect_right, bisect_left
import brisk


class trajectory:
    def __init__(self, init_state, init_time=None, init_history=None):
        self.states = [init_state]
        if init_time is None:
            self.transition_times = [0]
        else:
            self.transition_times = [init_time]

        if init_history is not None:
            self.states = init_history[0]
            self.transition_times = init_history[1]

    """    def draw_new_state(self, rates_from_i, draw_crit):
        # get rates of transition from i to j in probabilities matrix
        new_state, time = draw_crit.call(rates_from_i)
        return new_state, time"""
    def draw_new_state(self, index, draw_crit, rand_state, neg_log_time_sample):  
        # get rates of transition from i to j in probabilities matrix
        new_state, time = draw_crit.call(index, rand_state, neg_log_time_sample)
        return new_state, time
    
    def get_history_as_dict(self):
        ret_dict = {}
        for i, state in enumerate(self.states):
            ret_dict[self.transition_times[i]] = state
        return ret_dict

    def add_new_state(self, new_state, time_transition):
        # append state to states
        self.states.append(new_state)
        self.transition_times.append(self.transition_times[-1] + time_transition)

    def step(self, index, draw_crit, rand_state, neg_log_time_sample, time_stop=10e9):
        if time_stop > 0:
            # check that time of last state
            last_transition = self.last_time()
            if last_transition > time_stop:
                return

        new_state, time_to_transition = self.draw_new_state(index, draw_crit, rand_state, neg_log_time_sample)

        if new_state == -1:  # checks if rates out of a state are 0
            new_state = self.last_state()

        self.add_new_state(new_state, time_to_transition)

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
        index = brisk.bisect_right(self.transition_times, time)
        return self.states[index - 1]
 
    def states_at_times(self, times):
        """
        given a trajectory return what state it's in a time t
        Takes
            trajectory: trajectory object
            time(float): time to get state at
        Returns
            index(int): of state
        """
        states = [self.states[brisk.bisect_right(self.transition_times, time)-1] for time in times]
        return states


class trajectory_minimum:
    def __init__(self, init_state, init_time, index_of_last_sample=0):
        """
        A basic trajectory object. Stores basically no data for effecieny.
            init_state is the current state of the traj
            init_time is the array of times at which the traj has transitioned
            init_history is the array of states the traj has been in
        """
        __slots__ = ()
        self.current_state = init_state
        self.current_time = init_time
        self.index_of_last_sample = index_of_last_sample

    #setter and getter for current state
    def set_current_state(self, state):
        self.current_state = state
    def get_current_state(self):
        return self.current_state

    #setter and getter for current time
    def set_current_time(self, time):
        self.current_time = time
    def get_current_time(self):
        return self.current_time
    
    def get_index_of_last_sample(self):
        return self.index_of_last_sample
    def set_index_of_last_sample(self, index):
        self.index_of_last_sample = index
    def last_time(self):
        return self.current_time
    
    def batched_step(self, draw_crit, state_samples, neg_log_time_samples, sample_frequency, time_stop, ret_all=False, probe=False):
        last_state = self.get_current_state() 
        last_time = self.get_current_time()

        new_states, times = draw_crit.call_batched(
            last_state, 
            state_samples, 
            neg_log_time_samples,
            debug=probe)

        
        times += last_time
        self.set_current_state(new_states[-1])
        self.set_current_time(times[-1])

        probe_inds, probe_states = [], []
        time_check = sample_frequency * self.index_of_last_sample

        if self.current_time > time_check: # this means that at LEAST THE NEXT SAMPLING POINT HIT
            if probe: 
                    print("generated times and states:")
                    print(times, time_check)
            #test_ind = 0 
            while self.current_time  > time_check and time_stop >= time_check:
                # find the index of the first time that is greater than the time check ising bisect
                probe_inds.append(self.index_of_last_sample)
                #sample_trigger = bisect_left([last_time]+list(times), time_check)
                sample_trigger = brisk.bisect_left(times, time_check)
                if sample_trigger == 0: 
                    probe_states.append(last_state)
                else:
                    #probe_states.append(int(new_states[sample_trigger-2]))
                    probe_states.append(int(new_states[sample_trigger-1])) # for surre this the error
                self.index_of_last_sample = self.index_of_last_sample + 1
                time_check = sample_frequency * self.index_of_last_sample
            #else: 
            #    print(time_stop, time_check)

            if ret_all:
                return probe_states, probe_inds, self.current_state, self.current_time
            
            return probe_states, probe_inds             
        
        else: # no sampling points hit
            if ret_all:
                return [-1], [-1], self.current_state, self.current_time
            return [-1], [-1]
        
    #def last_state(self):
    #    return self.current_state



def trajectory_from_list(list, start_time, end_time):
    """
    Creates a trajectory from a list of states
    Takes:
        list: list of states
        start_time: start time
        end_time: end time
        steps: step size
    Returns:
        trajectory: trajectory object
    """
    times, states = [], []
    steps = (end_time - start_time) / len(list)
    for ind, t in enumerate(np.arange(start_time, end_time, steps)):
        states += int(list[ind]),
        times += t,

    return trajectory(
        init_state=states[-1], init_time=times[-1], init_history=[states, times]
    )


def add_history_to_trajectory(trajectory, history, history_times):
    for i in range(len(history)):
        trajectory.add_new_state(history[i], history_times[i])
    return trajectory


def sample_trajectory(trajectory, start, end, step):
    """
    Samples a trajectory at a given time
    Takes:
        trajectory: trajectory to sample
        start: start time
        end: end time
        step: step size
    Returns:
        ret_dict: dictionary of states and their counts
    """
    states = []
    times = []
    for t in np.arange(start, end, step):
        state_temp = str(trajectory.state_at_time(t))
        states.append(int(state_temp))
        times.append(t)

    return [states, times]


def trajectory_from_list(list, start_time, end_time):
    """
    Creates a trajectory from a list of states
    Takes:
        list: list of states
        start_time: start time
        end_time: end time
        steps: step size
    Returns:
        trajectory: trajectory object
    """
    times, states = [], []
    steps = (end_time - start_time) / len(list)
    for ind, t in enumerate(np.arange(start_time, end_time, steps)):
        states.append(int(list[ind]))
        times.append(t)

    return trajectory(
        init_state=states[-1], init_time=times[-1], init_history=[states, times]
    )


def add_history_to_trajectory(trajectory, history, history_times):
    for i in range(len(history)):
        trajectory.add_new_state(history[i], history_times[i])
    return trajectory
