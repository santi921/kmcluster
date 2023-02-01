from kmcluster.core.intialize import population_ind_to_trajectories
from tqdm import tqdm
import json 
import numpy as np

class kmc():
    def __init__(self, pop_size, draw_crit, initialization, energies, time_stop=-1):
        #self.steps = steps
        self.pop_size = pop_size
        self.draw_crit = draw_crit
        self.energies = energies
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
            energies_from_i = self.energies[traj_last_ind]
            traj.step(energies_from_i, self.draw_crit, time_stop=self.time_stop)
            
            
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
            state_temp = str(i.state_at_time(t))
            if state_temp not in ret_dict.keys():
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
        mat_save = np.zeros((self.pop_size, int(np.ceil(end_time-start_time / step))))
        sampling_array = np.arange(start_time, end_time, step)

        for ind_t, t in enumerate(sampling_array):
            for ind, i in enumerate(self.trajectories):
                mat_save[ind][ind_t] = i.state_at_time(t)
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
        