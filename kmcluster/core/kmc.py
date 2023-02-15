from kmcluster.core.intialize import population_ind_to_trajectories
from tqdm import tqdm
import json, os 
import numpy as np

class kmc():
    def __init__(self, draw_crit, initialization, energies, memory_friendly, check_converged, time_stop=-1):
        #self.steps = steps
        self.pop_size = initialization.size 
        self.draw_crit = draw_crit
        
        self.energies = energies
        self.time_stop = time_stop
        self.initialization = initialization
        
        self.check_converged = check_converged
        self.check_converged_frequency = 1000
        self.check_converged_patience = 100
        self.pop_prop_hist = []
        
        self.save_ind = 1
        self.memory_friendly = memory_friendly
        #assert if memory friendly is true then step size save is not none
        #if self.memory_friendly:
        #    assert step_size_save is not None, "step_size_save must be set if memory_friendly is True, it's needed to save trajectories"
        self.pop_init = initialization.get_init_populations()
        self.trajectories = population_ind_to_trajectories(self.pop_init, draw_crit)
    

    def step(self):
        for traj in self.trajectories:
            # get last state in traj 
            traj_last_ind = traj.last_state()
            if self.time_stop > 0:
                traj_last_time = traj.last_time()
                if traj_last_time > self.time_stop:
                    continue
                else: 
                    energies_from_i = self.energies[traj_last_ind]
                    traj.step(energies_from_i, self.draw_crit, time_stop=self.time_stop)
        
            
    def run(self, n_steps=10):
        if n_steps == -1:
            self.step_count = 0 
            # check if all trajectories have reached time_stop
            last_time_arr = np.array([i.last_time() for i in self.trajectories])
            while not all([ i > self.time_stop for i in last_time_arr]):
                # print lowest time and clear output
                lowest_time = np.min(last_time_arr)
    
                if self.step_count % 100 == 0:
                    print(
                        "Lowest time at step {}: ".format(self.step_count), 
                        np.min(lowest_time), end='\r')
                    
                self.step_count = self.step_count + 1
                self.step()
                last_time_arr = np.array([i.last_time() for i in self.trajectories])    
            
                if self.check_converged: 
                    # get the last state of each trajectory
                    last_state_arr = np.array([i.last_state() for i in self.trajectories])
                    # get the number of times each state occurs
                    state_counts = np.bincount(last_state_arr)
                    # get the proportion of each state
                    state_prop = state_counts / self.pop_size
                    # get the number of times the state proportions have been the same with a tolerance of 1e-2
                    self.pop_prop_hist.append(state_prop)
                    if len(self.pop_prop_hist) > self.check_converged_patience: # only keep the last 100 proportions
                        self.pop_prop_hist.pop(0)
                    
                    if len(self.pop_prop_hist) == self.check_converged_patience: # check if the last 100 proportions are the same
                        if all([np.allclose(i, self.pop_prop_hist[0], atol=5e-2) for i in self.pop_prop_hist[-self.check_converged_patience:]]):
                            print(
                                "Converged at step {}, proportions haven't changed >5% in 100 steps".format(self.step_count)
                                )
                        
                if self.memory_friendly: 
                    # if lowest_time is 1/10 of the time_stop then save the trajectories
                    if lowest_time > self.time_stop * self.save_ind / 10:
                        time_save = self.time_stop * self.save_ind / 10
                        time_save = np.min(self.time_stop * (self.save_ind - 1) / 10, 0)
                        save_step = time_save / 100
                        
                        self.save_as_matrix(
                            file="trajectories_{}_ckpt.json".format(self.save_ind),
                            start_time=0, 
                            end_time=save_step, 
                            step=save_step,
                            append = True)
                        
                        # get the last state of each trajectory at time_save
                        last_state_arr = np.array([i.state_at_time(time_save) for i in self.trajectories])
                        self.trajectories = population_ind_to_trajectories(last_state_arr, self.draw_crit)                        
                
        else:        
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
    

    def save_as_matrix(self, file, start_time=0, end_time=100, step=1, append=False):
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

        if append: 
            # check if file exists
            if os.path.isfile(file):
                # load file
                mat_load = np.load(file)
                # append to file
                mat_save = np.concatenate((mat_load, mat_save), axis=1)
                # save file
                np.save(file, mat_save)
            else: 
                np.save(file, mat_save)
        else: 
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
        