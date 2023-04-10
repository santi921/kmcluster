import numpy as np 
from bisect import bisect_right, bisect_left

class trajectory:
    def __init__(self, init_state, init_time=None, init_history=None):
        """
        A basic trajectory object.
            init_state is the current state of the traj
            init_time is the array of times at which the traj has transitioned
            init_history is the array of states the traj has been in
        """
        __slots__ = ()
        
        cond_pre = True
        if init_time is not None or init_history is not None:
            if init_time is None or init_history is None: 
                cond_pre = False 
        if init_time is not None or init_history is not None:
            assert len(init_time) == len(init_history), "init_time and init_history must be the same length"
        assert cond_pre, "init_time and init_history must be either both None or both not None"

        
        if init_time is None:
            self.states_long_term = np.array([], dtype=np.uint8)
            self.transition_times_long_term = np.array([], dtype=np.float32)
            self.transition_times_short_term = [0]
            self.states_short_term = [init_state] 

        else:
            self.states_long_term = np.array(init_history, dtype=np.uint8)
            self.transition_times_long_term = np.array(init_time, dtype=np.float32)
            self.states_short_term = []
            self.transition_times_short_term = []


    def get_history_as_dict(self):
        ret_dict = {}
        for i, state in enumerate(self.states):
            ret_dict[self.transition_times[i]] = state
        return ret_dict


    def add_new_state(self, new_state, last_transition, time_transition, update_long_term=False):
        # append state to states
        if update_long_term:
            self.add_new_states(self.states_short_term, self.transition_times_short_term, batched=True)
            
        else: 
            self.states_short_term += np.uint8(new_state),
            self.transition_times_short_term += np.float32(last_transition + time_transition),
        

    def add_new_states(self, states, transition_times, batched=False):
        if batched: 
            #self.states_long_term = array('I', np.array(self.states_long_term, states))
            #self.transition_times_long_term = array('d', np.array(self.transition_times_long_term, transition_times))
            #print(self.states_long_term.dtype, type(states[0]))
            self.states_long_term = np.concatenate(
                [self.states_long_term, np.array(states, dtype=np.uint8)], dtype=np.uint8)
            self.transition_times_long_term = np.concatenate(
                [self.transition_times_long_term, np.array(transition_times, dtype=np.float32)], dtype=np.float32)
            self.states_short_term = []
            self.transition_times_short_term = []
        
        else: 
            self.states_short_term = self.states_short_term + states,
            self.transition_times_short_term = self.transition_times_short_term + transition_times,
        

    def batched_step(self, draw_crit, state_samples, neg_log_time_samples, time_stop=10e9,):
        time_to_transition_min = 10**-10
        
        for ind in range(len(state_samples)):
            if time_stop > 0:
                # check that time of last state
                if ind == 0:
                    last_transition = self.last_time() #
                    new_states, transition_times = [], []
                else:
                    last_transition = last_transition + time_to_transition 
                    new_states += new_state,
                    transition_times += last_transition,
                
                if last_transition > time_stop: #
                    break 
        
            if ind == 0:
                cur_state = self.last_state()
            else:
                cur_state = new_state
            
            new_state, time_to_transition = draw_crit.call(
                cur_state, state_samples[ind], neg_log_time_samples[ind])
            
            if time_to_transition < time_to_transition_min:
                time_to_transition_min = time_to_transition
            
            if new_state == -1:  
                new_state = self.last_state()
            
        self.add_new_states(new_states, transition_times, batched=True) 
        
        del new_states, transition_times
        if time_to_transition_min < 10**-15:
            return 1, time_to_transition_min
        return 0, time_to_transition_min


    def batched_step_old(self, draw_crit, state_samples, neg_log_time_samples, time_stop=10e9,):
        time_to_transition_min = 10**-10
        for ind in range(len(state_samples)):
            if time_stop > 0:
                # check that time of last state
                if ind == 0:
                    last_transition = self.last_time() #
                else:
                    last_transition = last_transition + time_to_transition #
                
                if last_transition > time_stop: #
                    break 
        
            if ind == 0:
                cur_state = self.last_state()
            else:
                cur_state = new_state
            
            new_state, time_to_transition = draw_crit.call(cur_state, state_samples[ind], neg_log_time_samples[ind])
            
            if time_to_transition < time_to_transition_min:
                time_to_transition_min = time_to_transition
            
            if new_state == -1:  
                new_state = self.last_state()

            self.add_new_state(new_state, last_transition, time_to_transition) #
        
        del new_state, last_transition, time_to_transition
        if time_to_transition_min < 10**-15:
            return 1, time_to_transition_min
        return 0, time_to_transition_min


    def batched_step_base_parallel(self, draw_crit, state_samples, neg_log_time_samples, time_stop=10e9,):
        time_to_transition_min = 10**-10
        for ind in range(len(state_samples)):
            if time_stop > 0:
                # check that time of last state
                if ind == 0:
                    last_transition = self.last_time() #
                else:
                    last_transition = last_transition + time_to_transition #
                
                if last_transition > time_stop: #
                    break 
        
            if ind == 0:
                cur_state = self.last_state()
            else:
                cur_state = new_state
            
            new_state, time_to_transition = draw_crit.call(cur_state, state_samples[ind], neg_log_time_samples[ind])
            
            if time_to_transition < time_to_transition_min:
                time_to_transition_min = time_to_transition
            
            if new_state == -1:  
                new_state = self.last_state()

            #self.add_new_state(new_state, last_transition, time_to_transition)
            return new_state, last_transition, time_to_transition


    def batched_steps_parallel(self, draw_crit, state_samples, neg_log_time_samples, time_stop=10e9,):
   
        for ind in range(len(state_samples)):
            if time_stop > 0:
                # check that time of last state
                if ind == 0:
                    last_transition = self.last_time() #
                    new_states, transition_times = [], []
                else:
                    last_transition = last_transition + time_to_transition 
                    new_states += new_state,
                    transition_times += last_transition,
                
                if last_transition > time_stop: #
                    break 

            if ind == 0:
                cur_state = self.last_state()
            else:
                cur_state = new_state
            
            new_state, time_to_transition = draw_crit.call(cur_state, state_samples[ind], neg_log_time_samples[ind])
            
            if new_state == -1:  
                new_state = self.last_state()
        
        return new_states, transition_times


    def step(self, traj_last_ind, draw_crit, time_stop=10e9, state_sample=None, neg_log_time_sample=None):
        if time_stop > 0:
            # check that time of last state
            last_transition = self.last_time() #
        
        new_state, time_to_transition = draw_crit.call(traj_last_ind, state_sample, neg_log_time_sample)
            
        if new_state == -1:  
            new_state = self.last_state()

        self.add_new_state(new_state, last_transition, time_to_transition) #
        
        del(new_state)
        del(last_transition)

        if time_to_transition < 10**-15:
            return 1, time_to_transition
        
        return 0, time_to_transition
        

    def get_history(self, merge_short_term=True):
        if merge_short_term:
            #self.states = array('I', self.states + self.states_short_term)
            #self.transition_times = array('f', self.transition_times + self.transition_times_short_term)
            self.states_long_term = np.concatenate(
                [self.states_long_term, np.array(self.states_short_term, dtype=np.uint8)], dtype=np.uint8)
            self.transition_times_long_term = np.concatenate(
                [self.transition_times_long_term, np.array(self.transition_times_short_term, dtype=np.float32)], dtype=np.float32)

            self.states_short_term = []
            self.transition_times_short_term = []

        return self.states_long_term, self.transition_times_long_term


    def last_state(self):
        if len(self.states_short_term) > 0:
            return self.states_short_term[-1]
        return self.states_long_term[-1]


    def last_time(self):
        if len(self.transition_times_short_term) > 0:
            return self.transition_times_short_term[-1]
        return self.transition_times_long_term[-1]


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
        index = np.searchsorted(np.array(self.transition_times_long_term), np.array(time), side='right')
        return self.states_long_term[index - 1]


    def states_at_times(self, times):
        """
        given a trajectory return what state it's in a time t
        Takes
            trajectory: trajectory object
            time(float): time to get state at
        Returns
            index(int): of state
        """
        states = [self.states_long_term[bisect_right(self.transition_times_long_term, time)-1] for time in times]
        return states

class trajectory_minimum:
    def __init__(self, init_state, init_time):
        """
        A basic trajectory object. Stores basically no data for effecieny.
            init_state is the current state of the traj
            init_time is the array of times at which the traj has transitioned
            init_history is the array of states the traj has been in
        """
        __slots__ = ()
        self.current_state = init_state
        self.current_time = init_time
        self.index_of_last_sample = 0

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

    def batched_step(self, draw_crit, state_samples, neg_log_time_samples, sample_frequency, time_stop, ret_all=False):
        last_state = self.current_state 
        last_time = self.current_time

        new_states, times = draw_crit.call_batched(
            last_state, 
            state_samples, 
            neg_log_time_samples)
        times += last_time
        #print(times)
        self.current_state = new_states[-1]
        self.current_time = times[-1]
        probe_inds, probe_states = [], []
        time_check = sample_frequency * self.index_of_last_sample


        if times[-1] > time_check: # this means that at LEAST THE NEXT SAMPLING POINT HIT
            while times[-1] > time_check and time_check < time_stop:
                # find the index of the first time that is greater than the time check ising bisect
                sample_trigger = bisect_right(times, time_check)
                probe_inds.append(self.index_of_last_sample)
                probe_states.append(int(new_states[sample_trigger]))
                self.index_of_last_sample = self.index_of_last_sample + 1
                time_check = sample_frequency * self.index_of_last_sample
            
            if ret_all:
                return probe_states, probe_inds, self.current_state, self.current_time
            #print(new_states)
            #print(probe_states, probe_inds)
            return probe_states, probe_inds             
        else: 
            if ret_all:

                return [-1], [-1], self.current_state, self.current_time
            return [-1], [-1]
        
    #def last_state(self):
    #    return self.current_state
    def last_time(self):
        return self.current_time


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