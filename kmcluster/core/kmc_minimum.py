import os, time, json
import pandas as pd 
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 
import plotly.express as px
from numpy.random import uniform
from multiprocessing.pool import Pool
from kmcluster.core.trajectory import (
    trajectory_minimum,
)
from tabulate import tabulate
import itertools
from bisect import bisect_right
import pickle as pkl
from kmcluster.core.trajectory import trajectory_minimum
from kmcluster.core.intialize import population_ind_to_minimum_trajectories
#from kmcluster.core.viz import compute_state_counts


class kmc:
    def __init__(
        self,
        draw_crit,
        energies,
        initialization=None,
        checkpoint=False,
        checkpoint_dir="./checkpoints/",
        final_save_prefix="saved_data",
        time_stop=-1,
        sample_frequency=-1,
        state_dict_file=None,
        batch_size=1000,
    ):
        self.draw_crit = draw_crit
        self.energies = energies
        self.time_stop = time_stop
        self.initialization = initialization
        self.checkpoint = checkpoint
        self.final_save_prefix = final_save_prefix
        self.checkpoint_dir = checkpoint_dir
        self.pop_prop_hist = []
        self.save_ind = 1
        self.sample_index = 0
        self.n_states = energies.shape[0]
        self.batch_size = batch_size
        self.state_dict_file = state_dict_file
        
        
        if sample_frequency == -1:
            self.sample_frequency = time_stop/100
        
        self.results_mat = np.zeros((self.n_states, 1+int(self.time_stop/self.sample_frequency)))
        self.probe_status = [True] + [False for i in range(int(self.time_stop/self.sample_frequency)-1)]
        self.probe_times = np.array([i*self.sample_frequency for i in range(int(self.time_stop/self.sample_frequency))])
        assert (
            initialization is not None or state_dict_file is not None
        ), "init, load_from_state_mat cannot all be not None"

        if checkpoint:
            # check if the checkpoint directory exists'
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
        
        # load optionality
        if self.state_dict_file is not None:
            self.load_from_state_dict()
    
        else:             
            self.pop_init = initialization.get_init_populations()
            self.trajectories = population_ind_to_minimum_trajectories(self.pop_init)
            self.pop_size = initialization.size
        
        #### from rfkmc #### from rfkmc #### from rfkmc 
        """rate_mat = np.zeros((len(energies), len(energies)))
        for i in range(len(self.energies)):
            for j in range(len(self.energies[0])):
                if self.energy_mat[i, j] != 0:
                    rate_mat[i, j] = (self.k_b_t / (4.1357 * 10**-15)) * np.exp(
                        -(self.energies[i, j] / (self.k_b_t))
                    )
        self.rate_mat = np.array(rate_mat)
        self.sum_rates = np.sum(self.rate_mat, axis=1)"""
        #### from rfkmc #### from rfkmc #### from rfkmc

        print("done initializing....")


    def get_sampling(self):
        """
        Precomputes the random state and time samples for the trajectories - for speed
        """
        n_traj = len(self.trajectories)
        
        batch_size = self.batch_size
        rand_state_samples = uniform(0, 1, (n_traj, batch_size))
        rand_time_samples = uniform(0.0000001, 1, (n_traj, batch_size))
        neg_log_rand_time_samples = -np.log(rand_time_samples)
        
        rand_state_samples = np.float32(rand_state_samples)
        neg_log_rand_time_samples = np.float32(neg_log_rand_time_samples)

        return  rand_state_samples, neg_log_rand_time_samples
    

    def task_batch_single(self, ind, queue):
        #print(ind, queue)
        traj_last_time = self.trajectories[ind].last_time()  
        if traj_last_time > self.time_stop:
            print("stopping")
            return -1
       
        else:
            queue.get()
            probe, current_state, current_time = self.trajectories[ind].batched_step(
                self.draw_crit, 
                state_samples=self.rand_state_samples[ind],
                neg_log_time_samples=self.neg_log_rand_time_samples[ind],
                sample_frequency=self.sample_freq,
                ret_all=True
            )
            queue.put(probe)
            #return_list.append(probe)
            return probe, current_state, current_time

                #if state > 0: 
                #    self.results_mat[int(state), ind] += 1


    def step_batched_parallel(self):
        import multiprocessing
        manager = multiprocessing.Manager()
        #return_list = manager.list()
        return_list = multiprocessing.Queue()
        self.rand_state_samples, self.neg_log_rand_time_samples = self.get_sampling()
        ind_list = list(range(len(self.trajectories)))
        # creat iterator with ind_list, return_list 
        pass_iter = zip(ind_list, itertools.repeat(return_list))
        n_threads = 20
        
        """for i in range(n_threads):
            p = multiprocessing.Process(target=task_sadfasdfdasasdfasdfbatch_single, args=(i, return_dict))
            jobs.append(p)
            p.start()"""
        
        with Pool(n_threads) as pool:
            task = pool.starmap(
                self.task_batch_single, 
                pass_iter, 
                chunksize=int(len(ind_list)/(n_threads))
            )
            for ind, res in enumerate(task):
                #print(res)
                if res == -1: 
                    if res[0] > 0: 
                        self.results_mat[int(res[0]), ind] += 1
                    self.trajectories[ind].set_current_state(res[1])
                    self.trajectories[ind].set_current_time(res[2])
        
        print(return_list.get())  



        #print(np.max(self.results_mat))
        #print(np.max(self.results_mat))


    def step_batched(self):
        self.rand_state_samples, self.neg_log_rand_time_samples = self.get_sampling()
        for ind, traj in enumerate(self.trajectories):  
            traj_last_time = traj.last_time() # 
            if traj_last_time > self.time_stop:
                continue
            else:
                probe_states, probe_ind = traj.batched_step(
                    self.draw_crit, 
                    state_samples=self.rand_state_samples[ind],
                    neg_log_time_samples=self.neg_log_rand_time_samples[ind],
                    sample_frequency=self.sample_frequency,
                    time_stop=self.time_stop
                )
    
                for ind in range(len(probe_states)): 
                    if probe_ind[ind] > -1: 
                        self.results_mat[int(probe_states[ind]),int(probe_ind[ind])] += 1
                

    def run(self, n_steps=10):
        time_list = []
        if n_steps == -1:
            trigger = False
            self.step_count = 0
            ind_tracker = 1
            last_time_arr = np.array([i.last_time() for i in self.trajectories])
            
            print("starting run")
            while np.min(last_time_arr) < self.time_stop:
                #print("batch")
                timer_start = time.time()
                self.step_batched()
                timer_end = time.time()
                time_list.append(timer_end - timer_start)
                
                if self.step_count > 5000 * ind_tracker:
                    lowest_time = np.min(last_time_arr)
                    mean_time = np.mean(last_time_arr)
                    ind_tracker += 1
                    print("-"*40)
                    print(">>> step: {}".format(self.step_count))
                    print("Lowest time at step: {:.5e}".format(lowest_time))
                    print("mean time at step: {:.5e}".format(mean_time))
                    print("time to step: {}\n".format(np.mean(time_list)))
                    print("-"*40)
                    # show first_column of results_mat
                    rolling_ind = bisect_right(self.probe_times, lowest_time)
                    print("rolling index: {} out of {}".format(rolling_ind, len(self.probe_times)))
                    header = ["{:.1e}".format(i) for i in self.probe_times[rolling_ind-1:rolling_ind+15]]
                    table = tabulate(self.results_mat[:,rolling_ind-1:rolling_ind+15], tablefmt="fancy_grid", headers=header)
                    print(table)
                    # print sum of first 26 columns, 
                    self.save_as_dict("./test_dict.pkl")
                    print("rolling state sum: \n{}".format(np.sum(self.results_mat[:,rolling_ind-1:rolling_ind+26], axis=0)))

                #print("step count: {}".format(self.step_count))
                self.step_count = self.step_count + self.batch_size
                
               
                last_time_arr = np.array([i.last_time() for i in self.trajectories])
                lowest_time = np.min(last_time_arr)
                ind_lowest = bisect_right(self.probe_times, lowest_time)
                
                try:
                    self.probe_status[ind_lowest] = True
                except: 
                    print("Calc Done!")

                if self.checkpoint:
                    if lowest_time > self.time_stop * self.save_ind / 10:  
                        print("hit checkpoint {}/10".format(self.save_ind))
                        print("saving checkpoint at step {}".format(self.step_count))
                        time_save = self.time_stop * self.save_ind / 10
                        save_step = time_save / self.coarsening_mesh
                        self.save_as_dict("./test_dict.json")
                        trigger = True
                
                if trigger:
                    trigger = False
                    self.save_ind = self.save_ind + 1       
            
            print("done with kmc run to stop time {}".format(self.time_stop))
            print("this took {} steps".format(self.step_count))
            # save run
            self.probe_status = [True for i in self.probe_status]
            #check is self.checkpoint exists
            if self.checkpoint and not os.path.exists(self.checkpoint_dir):
                os.mkdir(self.checkpoint_dir)
            
            # TODO: renable after optimizing
            """self.save_as_matrix(
                file="{}{}_trajectories_{}_final_ckpt".format(
                    self.checkpoint_dir, self.final_save_prefix, self.save_ind
                ),
                start_time=start_time,
                end_time=end_time,
                step=step,
                append=False,
            )"""
    
            #if lowest_time is None:
            # check if lowest_time is instantiated
            # if not, instantiate it
            
            lowest_time = np.min(last_time_arr)
            mean_time = np.mean(last_time_arr)

            print(
                    "Lowest time at final step {}: {:.5e}".format(
                        self.step_count, lowest_time
                    )
                )
            print("mean time at final step: {:.5e}\n".format(mean_time))

        else:
            for _ in tqdm(range(n_steps)):
                self.step()
    

    def get_state_dict_at_time_as_pandas(self, t=0):
        """
        Returns a pandas dataframe of states and their counts at time t
        Takes:
            t: time to get state counts at
        Returns:
            ret_df: pandas dataframe of states and their counts
        """
        raise NotImplementedError("get_state_dict_at_time_as_pandas not implemented")


    def save_as_dict(self, file):
        """
        Saves states to json file
        Takes:
            file: file to save to
        Returns:
            None
        """
        ret_dict = {"running_state": {}}
        
        

        ret_dict['time_stop'] = self.time_stop
        ret_dict['population_size'] = self.pop_size
        ret_dict["sample_frequency"] = self.sample_frequency
        ret_dict["probe_status"] = self.probe_status

        # save all current run info
        ret_dict["results_mat"] = self.results_mat
        ret_dict["traj_times"] = [i.get_current_time() for i in self.trajectories]
        ret_dict["traj_states"] = [i.get_current_state() for i in self.trajectories]

        
        with open(file, 'wb') as output:
            # Pickle dictionary using protocol 0.
            pkl.dump(ret_dict, output)
        

    def load_from_state_dict(self):
        """
        Initializes kmc object from state dictionary
        """
        trajectories = []
        print("-"*20 + "Reload Module" + "-"*20)
        with open(self.state_dict_file, 'rb') as input:
            ret_dict = pkl.load(input)
        ########################################################
        ##### initializes the most current info on the run #####
        times_init = np.zeros(ret_dict["population_size"])
        traj_times = ret_dict["traj_times"]
        traj_states = ret_dict["traj_states"]

        for ind, i in enumerate(traj_times):
            traj_temp = ret_dict["traj_times"][ind]
            times_init[ind] = traj_temp
            trajectories.append(
                trajectory_minimum(init_state=traj_states[ind], init_time=i)
            )
        self.results_mat = np.array(ret_dict["results_mat"])
        self.trajectories = trajectories
        self.pop_size = len(trajectories)
        ########################################################

        ########################################################
        ########## initalizes probe states ##########
        self.probe_status = ret_dict["probe_status"] 
        assert ret_dict["sample_frequency"] == self.sample_frequency, "my brother in christ the sampling freq must be the same as the save dict"
        assert ret_dict["time_stop"] == self.time_stop, "my brother in christ the time stop must be the same as the save dict"
        self.sample_frequency = ret_dict["sample_frequency"]
        self.time_stop = ret_dict["time_stop"]

        ########################################################
        print("done reloading from file {}".format(self.state_dict_file))
        print("loaded {} trajectories".format(len(trajectories)))
        print("{}% of probe states are complete".format(100 * np.sum(self.probe_status) / len(self.probe_status)))
        print("sample frequency is {}s".format(self.sample_frequency))
        lowest_time = np.min(ret_dict["traj_times"])
        print("slowest trajectory is {}".format(lowest_time))
        print("-"*20 + "Reload Module" + "-"*20)    

        
    def plot_top_n_states_stacked(
            self, 
            n_show=5, 
            resolution=None, 
            max_time=None,
            title=None,
            xlabel=None, 
            ylabel=None,
            save=False,
            show=True,
            save_name="./output_top.png", ):
        """
        bin and count what states are in what bins
        Takes:
            list of trajectories objects
            resolution(float): bin size
            time_stop(float): time upper bound on counting
        """
        if n_show == -1:
            n_show = self.n_states

        if max_time is None:
            max_time = self.lowest_time
        
        if resolution is None:
            resolution = self.lowest_time / 100

        count_dict = self.results_mat.T
        # convert to pandas
        #count_df = pd.DataFrame.from_dict(count_dict, orient='index')
        count_df = pd.DataFrame(count_dict)
        #add column names as times 
        count_df.index = np.arange(0, self.time_stop, self.sample_frequency)
        #add row names as states
        count_df.columns = np.arange(0, self.n_states)
        print(count_df.head())
        print(count_df.keys())
    
        # sum all counts for each state
        sum_count = count_df.sum(axis=0)
        # get top n states
        keys_top_n = sum_count.nlargest(n_show).index.to_list()

        x_axis = np.arange(0, self.time_stop, self.sample_frequency)
        counts_per_state = np.zeros((n_show, len(x_axis)))
        self.pop_size

        df_show = count_df[keys_top_n]
        # divide by total population size
        df_show = df_show / self.pop_size    
        print(df_show.head())
        fig = px.area(df_show, title=title, x=x_axis, y=df_show.keys())

        # set xlim 
        fig.update_xaxes(range=[0, max_time])
        # set ylim
        fig.update_yaxes(range=[0, 1])
        # show legend 
        fig.update_layout(showlegend=True)
        
        if title: 
            fig.update_layout(title=title, title_font_size=16)        
        if xlabel:
            fig.update_xaxes(title=xlabel, title_font_size=14)
        if ylabel:
            fig.update_yaxes(title=ylabel, title_font_size=14)

        # save plotly express figure 
        if show:
            fig.show()

        if save:
            fig.write_image(save_name)        


    def plot_select_states_stacked(
            self, 
            states_to_plot,
            resolution=None, 
            max_time=None,
            title=None,
            xlabel=None, 
            ylabel=None,
            save=False,
            show=True,
            save_name="./output_top.png", ):
        """
        bin and count what states are in what bins
        Takes:
            list of trajectories objects
            resolution(float): bin size
            time_stop(float): time upper bound on counting
        """
        raise NotImplementedError
        

def load_kmc_from_matrix(
        state_file, 
        energies_mat, 
        draw_crit, 
        time_stop
    ):
    """
    Loads states from file
    Takes:
        file: file to load from
    Returns:
        ret_dict: dictionary of states and their counts
    """
    
    energies_nonzero = energies_mat[energies_mat != 0]
    print("smallest barrier: ", min(energies_nonzero))

    # initialize kmc object
    kmc_boltz = kmc(
        energies=energies_mat,
        draw_crit=draw_crit,
        time_stop=time_stop,
        initialization=None,
        load_from_dictionary=True, 
        save_dict_file=state_file,
        checkpoint=False,
        checkpoint_dir="./checkpoints/",
        final_save_prefix="saved_data",
        batch_size=1000, 
    )
     
    kmc.load_from_state_dict()
    

def rates_to_cum_rates(rates):
    #print(len(rates))
    rates_cum = [rates[0]]
    for i in range(1, len(rates)):
        rates_cum.append(rates_cum[i - 1] + rates[i])
    
    return np.array(rates_cum)


