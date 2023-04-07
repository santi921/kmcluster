import numpy as np 
from bisect import bisect_right


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

    def draw_new_state(self, energies_from_i, draw_crit, state_sample=None, neg_log_time_sample=None):
        # get rates of transition from i to j in probabilities matrix
        new_state, time = draw_crit.call(energies_from_i, state_sample, neg_log_time_sample)
        return new_state, time

    def get_history_as_dict(self):
        ret_dict = {}
        for i, state in enumerate(self.states):
            ret_dict[self.transition_times[i]] = state
        return ret_dict

    def add_new_state(self, new_state, last_transition, time_transition):
        # append state to states
        self.states += new_state,
        self.transition_times += last_transition + time_transition,

    def step(self, energies_from_i, draw_crit, time_stop=10e9, state_sample=None, neg_log_time_sample=None):
        if time_stop > 0:
            # check that time of last state
            last_transition = self.last_time() #
            if last_transition > time_stop: #
                return #

        new_state, time_to_transition = self.draw_new_state(
            energies_from_i, 
            draw_crit, 
            state_sample, 
            neg_log_time_sample)

        if new_state == -1:  
            new_state = self.last_state()

        self.add_new_state(new_state, last_transition, time_to_transition) #

        if time_to_transition < 10**-15:
            return 1, time_to_transition
        return 0, time_to_transition
        
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
        #index = bisect_right(self.transition_times, time)
        #index = np.argmax(np.array(self.transition_times) >= time)
        index = np.searchsorted(np.array(self.transition_times), np.array(time), side='right')
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
        states = [self.states[bisect_right(self.transition_times, time)-1] for time in times]
        return states


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
    states = trajectory.states_at_times(np.arange(start, end, step))
    times = np.arange(start, end, step)
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
        states += int(list[ind]),
        times += t,

    return trajectory(
        init_state=states[-1], init_time=times[-1], init_history=[states, times]
    )


def add_history_to_trajectory(trajectory, history, history_times):
    for i in range(len(history)):
        trajectory.add_new_state(history[i], history_times[i])
    return trajectory