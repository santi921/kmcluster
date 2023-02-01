from kmcluster.core.intialize import population_ind_to_trajectories
from tqdm import tqdm
import json 
import numpy as np

class kmc():
    def __init__(self, steps, pop_size, draw_crit, initialization, rates, time_stop=-1):
        self.steps = steps
        self.size = pop_size
        self.draw_crit = draw_crit
        self.rates = rates
        self.time_stop = time_stop
        self.initialization = initialization
        self.pop_init = initialization.get_init_populations()
        self.trajectories = population_ind_to_trajectories(self.pop_init, draw_crit)
    

    def step(self):

        for traj in self.trajectories:
            # get last state in traj 
            traj_last_ind = traj.last_state()
            # get row from numpy array
            # get row of transitions from state traj_last_ind
            rates_from_i = self.rates[traj_last_ind]
            traj.step(rates_from_i, self.draw_crit, time_stop=self.time_stop)
            


    def run(self, n_steps=10):
        for _ in tqdm(range(n_steps)):
            self.step()


    def get_state_dict_at_time(self, t = 0):
        """
        Returns a dictionary of states and their counts at time t
        Takes: 
            t: time to get state counts at
        Returns:
            ret_dict: dictionary of states and their counts
        """
        ret_dict = {}

        for i in self.trajectories:
            state_temp = i.get_state_dict_at_time(t)
            if state_temp is not None:
                ret_dict[state_temp] = 1 
            else:
                ret_dict[state_temp] += 1

        return ret_dict
    
    def save_as_matrix(self, file, start_time=0, end_time=100, step=1):
        """
        Saves states to file
        Takes:
            file: file to save to
            start_time: time to start saving
            end_time: time to end saving
            step: step size
        Returns: 
            None
        """
        mat_save = np.zeros((int(end_time-start_time / step), self.size))
        for t in range(start_time, end_time, step):
            for i in self.trajectories:
                mat_save[t][i] = i.get_state_at_time(t)
        np.save(file, mat_save)


    def save_as_dict(self, file, start_time=0, end_time=100, step=1):
        """
        Saves states to json file
        Takes:
            file: file to save to
            start_time: time to start saving
            end_time: time to end saving
            step: step size
        Returns: 
            None
        """
        master_dict = {}
        with open(file, 'w') as f:
            for t in range(start_time, end_time, step):
                master_dict[t] = self.get_state_dict_at_time(t)
            json.dump(master_dict, f)
        