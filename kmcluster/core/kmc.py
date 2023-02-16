import json, os 
import numpy as np
from tqdm import tqdm
from kmcluster.core.trajectory import trajectory
from kmcluster.core.intialize import population_ind_to_trajectories

class kmc():
    def __init__(
            self, 
            draw_crit, 
            energies, 
            memory_friendly, 
            check_converged, 
            initialization=None,
            checkpoint=False,
            time_stop=-1, 
            trajectories = None,):
        
        
        self.draw_crit = draw_crit
        self.memory_friendly = memory_friendly 
        self.energies = energies
        self.time_stop = time_stop
        self.initialization = initialization
        self.checkpoint = checkpoint
        self.check_converged = check_converged
        self.check_converged_frequency = 1000
        self.check_converged_patience = 100
        self.pop_prop_hist = []
        self.save_ind = 1
        
        assert trajectories is not None or initialization is not None, "init and trajectories cannot both be not None"

        if trajectories is None: 
            self.pop_init = initialization.get_init_populations()
            self.trajectories = population_ind_to_trajectories(self.pop_init)
            self.pop_size = initialization.size
        else: 
            self.pop_size = len(trajectories)
            self.trajectories = trajectories

    

    def step(self):
        for traj in self.trajectories:
            # get last state in traj 
            traj_last_ind = traj.last_state()
            if self.time_stop > 0:
                traj_last_time = traj.last_time()
                #print(traj_last_time)
                if traj_last_time > self.time_stop:
                    continue
                else: 
                    #print("last traj_ind: " + str(traj_last_ind))
                    energies_from_i = self.energies[traj_last_ind]
                    traj.step(energies_from_i, self.draw_crit, time_stop=self.time_stop)
            else: 
                #print("last traj_ind: " + str(traj_last_ind))
                energies_from_i = self.energies[traj_last_ind]
                traj.step(energies_from_i, self.draw_crit, time_stop=self.time_stop)
            

    def run(self, n_steps=10):
        if n_steps == -1:
            self.step_count = 0 
            # check if all trajectories have reached time_stop
            last_time_arr = np.array([i.last_time() for i in self.trajectories])
            
            while not all([ i > self.time_stop for i in last_time_arr]):
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
                    state_counts = np.bincount(last_state_arr, minlength=self.energies.shape[0])
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


                if self.checkpoint: 
                    if lowest_time > self.time_stop * self.save_ind / 10:
                        print("saving checkpoint at step {}".format(self.step_count))

                        time_save = self.time_stop * self.save_ind / 10

                        save_step = time_save / 100
                        #print(time_save, save_step)
                        self.save_as_matrix(
                            file="trajectories_{}_ckpt.json".format(self.save_ind),
                            start_time=0, 
                            end_time=save_step, 
                            step=save_step,
                            append = True)
                        self.save_ind = self.save_ind + 1

                if self.memory_friendly: 
                    if lowest_time > self.time_stop * self.save_ind / 10:
                        print("coarsening trajectories")
                        time_save = self.time_stop * self.save_ind / 10
                        #time_save = np.min(self.time_stop * (self.save_ind - 1) / 10, 0)
                        save_step = time_save / (10 * self.save_ind)
                        
                        traj_lists = []
                        print(self.trajectories)
                        for i in self.trajectories:
                            traj_lists.append(sample_trajectory(i, 0, time_save, save_step))
                        traj_new = [trajectory_from_list(i[0], 0, time_save) for i in traj_lists]
                        self.trajectories = traj_new
            # save run 
            start_time = 0 
            end_time = self.time_stop
            step = float((end_time - start_time) / 100)
            
            self.save_as_matrix(
                file="./save_trajectories".format(start_time, end_time, step),
                start_time=start_time,
                end_time=end_time,
                step=step,
                append = False)
            
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

        mat_save = np.zeros((self.pop_size, int(np.ceil((end_time-start_time) / step))))
        sampling_array = np.arange(start_time, end_time, step)

        for ind_t, t in enumerate(sampling_array):
            for ind, i in enumerate(self.trajectories):
                mat_save[ind][ind_t] = i.state_at_time(t)

        file = "{}_start_{}_end_{}_step_{}".format(file, start_time, end_time, step)
        
        if append: 
            # TODO: append to file
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

    return trajectory(init_state=states[-1], init_time=times[-1], init_history=[states, times])

def load_kmc_from_matrix(file, energies_mat, draw_crit, time_stop):
    """
    Loads states from file
    Takes:
        file: file to load from
    Returns: 
        ret_dict: dictionary of states and their counts
    """
    # get end time for simulation
    file_name = file[:-4]
    info = file_name.split("_")
    end_time = float(info[-3])
    start_time = float(info[-5])
    step = float(info[-1])

    mat_load = np.load(file)
    #print("start, end, step {}, {}, {}".format(start_time, end_time, step))
    # get the last column of the matrix
    # print number of trajectories to be loaded


    trajectories_loaded = [
        trajectory_from_list(i, start_time, end_time) 
        for i in mat_load.T]
    print("Loading {} trajectories".format(len(trajectories_loaded)))
    print("Starting at time {}".format(end_time))
    kmc_obj = kmc(
            draw_crit, 
            initialization = None, 
            energies = energies_mat, 
            memory_friendly=True, 
            check_converged=False, 
            checkpoint=True, 
            time_stop=time_stop, 
            trajectories = trajectories_loaded,
        )

    
    return kmc_obj

