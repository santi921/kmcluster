import json, os, time, gc
import pandas as pd 
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 
import plotly.express as px
from numpy.random import uniform
from multiprocessing.pool import Pool
from kmcluster.core.trajectory import (
    trajectory, 
    trajectory_from_list, 
    sample_trajectory, 
)
from kmcluster.core.intialize import population_ind_to_trajectories
from kmcluster.core.viz import compute_state_counts


class kmc:
    def __init__(
        self,
        draw_crit,
        energies,
        memory_friendly,
        initialization=None,
        checkpoint=False,
        checkpoint_dir="./checkpoints/",
        final_save_prefix="saved_data",
        time_stop=-1,
        sample_freq=-1,
        trajectories=None,
        run_batched=False,
        batch_size=1000,
        
    ):
        self.draw_crit = draw_crit
        self.memory_friendly = memory_friendly
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
        
        self.run_batched = run_batched
        self.batch_size = batch_size
        if sample_freq == -1:
            self.sample_freq = time_stop

        #self.coarsening_mesh = coarsening_mesh
        self.save_dask_parquet = './out'

        assert (
            trajectories is not None or initialization is not None
        ), "init and trajectories cannot both be not None"

        if checkpoint:
            # check if the checkpoint directory exists'
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)

        if trajectories is None:
            self.pop_init = initialization.get_init_populations()
            self.trajectories = population_ind_to_trajectories(self.pop_init)
            self.pop_size = initialization.size
        else:
            self.pop_size = len(trajectories)
            self.trajectories = trajectories


    def get_sampling(self):
        """
        Precomputes the random state and time samples for the trajectories - for speed
        """
        n_traj = len(self.trajectories)
        
        if self.run_batched:
            batch_size = self.batch_size
            rand_state_samples = uniform(0, 1, (n_traj, batch_size))
            rand_time_samples = uniform(0.0000001, 1, (n_traj, batch_size))
            neg_log_rand_time_samples = -np.log(rand_time_samples)
            
            rand_state_samples = np.float32(rand_state_samples)
            neg_log_rand_time_samples = np.float32(neg_log_rand_time_samples)

        else: 
            rand_state_samples = uniform(0, 1, n_traj)
            rand_time_samples = uniform(0.0000001, 1, n_traj)
            neg_log_rand_time_samples = -np.log(rand_time_samples)
            del rand_time_samples
        
        return  rand_state_samples, neg_log_rand_time_samples
    

    def step(self):
        sum_warning = 0
        small_transitions = []
        
        rand_state_samples, neg_log_rand_time_samples = self.get_sampling()
        #traj_last_time = np.array([i.last_time() for i in self.trajectories])
        
        for ind, traj in enumerate(self.trajectories):
            
            traj_last_time = traj.last_time() #
            if traj_last_time > self.time_stop:
                continue
            
            else:
                traj_last_ind = traj.last_state() #
                
                warning, time_to_transit = traj.step(
                    traj_last_ind, 
                    self.draw_crit, 
                    time_stop=self.time_stop,
                    state_sample=rand_state_samples[ind],
                    neg_log_time_sample=neg_log_rand_time_samples[ind]
                )
                
                if warning == 1:
                    sum_warning += warning
                    small_transitions += time_to_transit,
            
            """else:
                warning, time_to_transit = traj.step(
                    traj_last_ind, 
                    self.draw_crit, 
                    time_stop=self.time_stop,
                    state_sample=rand_state_samples[ind],
                    neg_log_time_sample=neg_log_rand_time_samples[ind]
                )
                if warning == 1:
                    sum_warning += warning
                    small_transitions += time_to_transit,"""
        # clear sample variables 
        del rand_state_samples
        del neg_log_rand_time_samples
                            
        if sum_warning > len(self.trajectories) / 100:
            print("Warning: trajectories in this step have steps sizes < 1e-15s")


    def task_batch_single(self, ind):
        rand_state_samples = self.rand_state_samples[ind]
        neg_log_rand_samples = self.neg_log_rand_time_samples[ind]
        traj_last_time = self.trajectories[ind].last_time()  
        if traj_last_time > self.time_stop:
            return
        else:
            new_state, transition_time = self.trajectories[ind].batched_step_base_parallel( 
                self.draw_crit, 
                time_stop=self.time_stop,
                state_samples=rand_state_samples,
                neg_log_time_samples=neg_log_rand_samples
            )
            return new_state, transition_time
        

    def task_batch_batched(self, ind):
        rand_state_samples = self.rand_state_samples[ind]
        neg_log_rand_samples = self.neg_log_rand_time_samples[ind]
        traj_last_time = self.trajectories[ind].last_time()  
        
        if traj_last_time > self.time_stop:
            return
        else:
            new_states, transition_times = self.trajectories[ind].batched_steps_parallel( 
                self.draw_crit, 
                time_stop=self.time_stop,
                state_samples=rand_state_samples,
                neg_log_time_samples=neg_log_rand_samples
            )
            return new_states, transition_times   
        
            #print(self.trajectories[ind].last_time())


    def step_batched_parallel(self):
        self.rand_state_samples, self.neg_log_rand_time_samples = self.get_sampling()
        ind_list = list(range(len(self.trajectories)))
        n_threads = 8
        with Pool(n_threads) as pool:
            task = pool.imap(
                self.task_batch_batched, 
                ind_list, 
                chunksize=int(len(ind_list)/(n_threads*16))
            )
            for ind, res in enumerate(task):
                if res is not None:
                    new_stats, transition_times = res
                    #self.trajectories[ind].add_state(new_state, transition_time)
                    self.trajectories[ind].add_states(new_stats, transition_times, batched=True)
                  

    def step_batched(self):
        rand_state_samples, neg_log_rand_time_samples = self.get_sampling()
        for ind, traj in enumerate(self.trajectories):  
            traj_last_time = traj.last_time() # 
            #print(traj_last_time) 
            if traj_last_time > self.time_stop:
                continue
            else:
                _, _ = traj.batched_step(
                    self.draw_crit, 
                    time_stop=self.time_stop,
                    state_samples=rand_state_samples[ind],
                    neg_log_time_samples=neg_log_rand_time_samples[ind]
                )
            del traj_last_time
        del rand_state_samples
        del neg_log_rand_time_samples
        gc.collect()


    def step_batched_v2(self):
        sum_warning = 0
        small_transitions = []
        rand_state_samples, neg_log_rand_time_samples = self.get_sampling()
        #traj_last_time = np.array([i.last_time() for i in self.trajectories])
        #print("number of batched steps: ", len(rand_state_samples[0]))
        for ind, traj in enumerate(self.trajectories):  
            traj_last_time = traj.last_time()   
            if traj_last_time > self.time_stop:
                continue
            else:
                traj_last_ind = traj.last_state() 
                _, _ = traj.batched_step(
                    traj_last_ind, 
                    self.draw_crit, 
                    time_stop=self.time_stop,
                    state_samples=rand_state_samples[ind],
                    neg_log_time_samples=neg_log_rand_time_samples[ind]
                )
            del traj_last_time        
        del rand_state_samples
        del neg_log_rand_time_samples


    def run(self, n_steps=10):
        time_list = []
        if n_steps == -1:
            
            trigger = False
            self.step_count = 0
            # check if all traject
            # ories have reached time_stop
            ind_tracker = 1
            last_time_arr = np.array([i.last_time() for i in self.trajectories])
            
            #while not all([i > self.time_stop for i in last_time_arr]):
            #while not last_time_arr.all([last_time_arr > self.time_stop]):
            while np.min(last_time_arr) < self.time_stop:
        
                if self.run_batched: 
                    timer_start = time.time()
                    self.step_batched()
                    timer_end = time.time()
                    time_list.append(timer_end - timer_start)


                    if self.step_count > 4000 * ind_tracker:
                        lowest_time = np.min(last_time_arr)
                        mean_time = np.mean(last_time_arr)
                        ind_tracker += 1
                        print("-"*40)
                        print(">>> step: {}".format(self.step_count))
                        print("Lowest time at step: {:.5e}".format(lowest_time))
                        print("mean time at step: {:.5e}".format(mean_time))
                        print("time to step: {}\n".format(np.mean(time_list)))
                        #print("dumping trajectories")
                        print("-"*40)
                    self.step_count = self.step_count + self.batch_size
                
                else:
                    if self.step_count > 1000 * ind_tracker:
                        lowest_time = np.min(last_time_arr)
                        mean_time = np.mean(last_time_arr)
                        ind_tracker += 1
                        print("-"*40)
                        print(">>> step: {}".format(self.step_count))
                        print("Lowest time at step: {:.5e}".format(lowest_time))
                        print("mean time at step: {:.5e}".format(mean_time))
                        print("time to step: {}\n".format(np.mean(time_list)))
                        self.lowest_time = lowest_time
                    
                    timer_start = time.time()
                    self.step_count = self.step_count + 1
                    self.step() 
                    timer_end = time.time()
                    time_list.append(timer_end - timer_start)
                        

                last_time_arr = np.array([i.last_time() for i in self.trajectories])
                #traj_probe = self.trajectories[0]
                #print("number of steps in probe traj ", len(traj_probe.states)) 
                
                if self.checkpoint:
                    if lowest_time > self.time_stop * self.save_ind / 10:  
                        print("hit checkpoint {}/10".format(self.save_ind))
                        print("saving checkpoint at step {}".format(self.step_count))
                        time_save = self.time_stop * self.save_ind / 10
                        save_step = time_save / self.coarsening_mesh

                        self.save_as_matrix(
                            file="{}trajectories_{}_ckpt".format(
                                self.checkpoint_dir, self.save_ind
                            ),
                            start_time=0,
                            end_time=time_save,
                            step=save_step,
                            append=True,
                        )
                        trigger = True


                    if self.memory_friendly:
                        print(
                            "coarsening trajectories at step {}\n".format(
                                self.step_count
                            )
                        )
                        time_coarsen = self.time_stop * self.save_ind / 10
                        coarsen_step = time_coarsen / (self.coarsening_mesh)

                        traj_lists = []

                        for i in self.trajectories:
                            traj_lists.append(
                                sample_trajectory(i, 0, time_coarsen, coarsen_step)
                            )
                        traj_new = [
                            trajectory_from_list(i[0], 0, time_coarsen) for i in traj_lists
                        ]
                        self.trajectories = traj_new
                        
                        trigger = True
                
                if trigger:
                    trigger = False
                    self.save_ind = self.save_ind + 1       
            
            print("done with kmc run to stop time {}".format(self.time_stop))
            print("this took {} steps".format(self.step_count))
            # save run
            start_time = 0
            end_time = self.time_stop
            step = float((end_time - start_time) / self.coarsening_mesh)
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
            #self.lowest_time = lowest_time
            #self.step_count = self.step_count + 1
            #self.step()
            
            #allObjects = muppy.get_objects()
            #sum = summary.summarize(allObjects)
            #summary.print_(sum)
        else:
            for _ in tqdm(range(n_steps)):
                self.step()


    def get_state_dict_at_time(self, t=0):
        """
        Returns a dictionary of states and their counts at time t
        Takes:
            t: time to get state counts at
        Returns:
            ret_dict: dictionary of states and their counts
        """
        if t > self.time_stop:
            raise ValueError("time t is greater than time_stop")
        
        if t < 0:
            raise ValueError("time t is less than 0")
        
        #if t > self.lowest_time:
        #    print("WARNING: time t is greater than lowest time in trajectories")

        ret_dict = {str(i):0 for i in range(self.n_states)}
        list_of_states = [0 for i in range(self.n_states)]
        #for i in self.trajectories:
        #    ret_dict[str(i.state_at_time(t))] += 1

        [ret_dict.update({str(i.state_at_time(t)): ret_dict[str(i.state_at_time(t))] + 1}) for i in self.trajectories]
        return ret_dict
    

    def get_state_dict_at_time_as_pandas(self, t=0):
        """
        Returns a pandas dataframe of states and their counts at time t
        Takes:
            t: time to get state counts at
        Returns:
            ret_df: pandas dataframe of states and their counts
        """
        ret_dict = self.get_state_dict_at_time(t=t)
        # fill in missing states with 0 
        
        for i in range(self.n_states):
            if str(i) not in ret_dict.keys():
                ret_dict[str(i)] = 0
        ret_df = pd.DataFrame.from_dict(ret_dict, orient='index')
        ret_df.columns = ['count']
        
        return ret_df


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
        sampling_array = np.arange(start_time, end_time, step)
        mat_save = np.zeros(
            (self.pop_size, len(sampling_array))
        )

        for ind, i in enumerate(self.trajectories):
            mat_save[ind] = i.states_at_times(sampling_array)
        # step as scientific notation with 5 decimal places
        step = "{:.5e}".format(step)
        end_time = "{:.5e}".format(end_time)
        file = "{}_start_{}_end_{}_step_{}".format(file, start_time, end_time, step)
        print("Saving to file: {}".format(file))

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
        sampling_array = np.arange(start_time, end_time, step)
        
        mat_save = np.zeros(
            (self.pop_size, len(sampling_array))
        )
        
        for ind, i in enumerate(self.trajectories):
            mat_save[ind] = i.states_at_times(sampling_array)
        
        # for each state get the count at each time
        for time in range(len(sampling_array)):
            state_dict = {}
            for state in range(self.n_states):
                state_dict[str(state)] = int(np.sum(mat_save[:, time] == state))
            master_dict[str(sampling_array[time])] = state_dict    


        with open(file, "w") as f:
            json.dump(master_dict, f)


    def plot_top_n_states(
            self, 
            n_show=5, 
            resolution=None, 
            max_time=None,
            xlabel=None, 
            ylabel=None,
            title=None,
            save=False,
            save_name="./output_top.png", ):
        """

        bin and count what states are in what bins
        Takes:
            list of trajectories objects
            resolution(float): bin size
            time_stop(float): time upper bound on counting
        """
        if max_time is None:
            max_time = self.lowest_time
        if resolution is None:
            resolution = self.lowest_time / 100
        plt.rcParams["figure.figsize"] = (20, 10)
        count_dict = compute_state_counts(self.trajectories, resolution, max_time, self.n_states)
        keys_top_n = sorted(count_dict, key=count_dict.get, reverse=True)[:n_show]
        x_axis = np.arange(0, max_time, resolution)
        counts_per_state = np.zeros((n_show, len(x_axis)))
        
        for ind, i in enumerate(x_axis):
            # get state dict as pandas 
            state_df = self.get_state_dict_at_time_as_pandas(t=i)
            # get counts for top n states
            sum_count = state_df['count'].sum()
            df_ind = [int(i) for i in state_df.index.to_list()]
            list_get = [i for i in keys_top_n if i in df_ind]
            
            # get rows list_get rows from state_df
            for ind_update in range(len(list_get)):
                #print(state_df.iloc[list_get[ind_overwrite]]['count'])
                counts_per_state[ind_update, ind] = state_df.iloc[list_get[ind_update]]['count']
        
        for i in range(n_show):
            plt.plot(x_axis, counts_per_state[i, :]/sum_count, label=keys_top_n[i])

        if title is not None:
            plt.title(title, fontsize=16)
        if xlabel is not None:
            plt.xlabel(xlabel, fontsize=14)
        if ylabel is not None:
            plt.ylabel(ylabel, fontsize=14)
        # set image size
        
        
        # adjust x axis to min, max time
        plt.xlim(0, max_time)
        plt.ylim(
            0.9 * counts_per_state.min()/sum_count , 
            1.1 * counts_per_state.max()/sum_count )
        
        plt.legend()

        if save:
            plt.savefig(save_name) 
        plt.show()  


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

        count_dict = compute_state_counts(self.trajectories, resolution, max_time, self.n_states)
        
        if n_show >= self.n_states:
            keys_top_n = sorted(count_dict, key=count_dict.get, reverse=True)

        else:  
            keys_top_n = sorted(count_dict, key=count_dict.get, reverse=True)[:n_show]

        x_axis = np.arange(0, max_time, resolution)
        counts_per_state = np.zeros((n_show, len(x_axis)))
        
        for ind, i in enumerate(x_axis):
            # get state dict as pandas 
            state_df = self.get_state_dict_at_time_as_pandas(t=i)
            sum_count = state_df['count'].sum()
            # get counts for top n states
            df_ind = [int(i) for i in state_df.index.to_list()]
            list_get = [i for i in keys_top_n if i in df_ind]
            
            # get rows list_get rows from state_df
            for ind_update in range(len(list_get)):
                counts_per_state[ind_update, ind] = state_df.iloc[list_get[ind_update]]['count']/sum_count
        

        df = pd.DataFrame(counts_per_state, index=keys_top_n, columns=x_axis)
        df_new_format = pd.DataFrame(columns=['time', 'state', 'percentage'])
        for i in range(n_show):
            df_new_format = df_new_format.append(pd.DataFrame({'time': df.columns, 'state': df.index[i], 'percentage': df.iloc[i, :]}))
        df = df_new_format
        df = df.sort_values(by=['time', 'state'])   

        fig = px.area(df, title=title, x='time', y='percentage', color='state')

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

        if max_time is None:
            max_time = self.lowest_time
        
        if resolution is None:
            resolution = self.lowest_time / 100

        x_axis = np.arange(0, max_time, resolution)
        counts_per_state = np.zeros((len(states_to_plot), len(x_axis)))
        
        for ind, i in enumerate(x_axis):
            # get state dict as pandas 
            state_df = self.get_state_dict_at_time_as_pandas(t=i)
            sum_count = state_df['count'].sum()
            # get counts for top n states
            df_ind = [int(i) for i in state_df.index.to_list()]
            list_get = [i for i in states_to_plot if i in df_ind]
            
            # get rows list_get rows from state_df
            for ind_update in range(len(list_get)):
                #print(state_df.iloc[list_get[ind_overwrite]]['count'])
                counts_per_state[ind_update, ind] = state_df.iloc[list_get[ind_update]]['count']/sum_count
        

        df = pd.DataFrame(counts_per_state, index=states_to_plot, columns=x_axis)
        df_new_format = pd.DataFrame(columns=['time', 'state', 'percentage'])
        for i in range(len(states_to_plot)):
            df_new_format = df_new_format.append(pd.DataFrame({'time': df.columns, 'state': df.index[i], 'percentage': df.iloc[i, :]}))
        df = df_new_format
        df = df.sort_values(by=['time', 'state'])   

        fig = px.area(df, title=title, x='time', y='percentage', color='state')

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
        
        
    def dump_history_and_reset_trajectories(self):
        time_start_dump = time.time()
        num_trajectories = len(self.trajectories)
        states = np.zeros(num_trajectories)
        probe_hist, _ = self.trajectories[0].get_history()
        last_transitions = np.zeros(num_trajectories)
        state_histories = np.zeros((num_trajectories, len(probe_hist)))
        time_histories = np.zeros((num_trajectories, len(probe_hist)))
        
        for ind, traj in enumerate(self.trajectories):
            states[ind] = traj.last_state()     
            last_transitions[ind] = traj.last_time()
            states_traj, transitions_traj = traj.get_history()
            state_histories[ind, :] = states_traj
            time_histories[ind, :] = transitions_traj


        del self.trajectories
        new_trajectories = []
        for i in range(num_trajectories):
            new_trajectories.append(
                trajectory(
                init_state = states[i], 
                init_history=[last_transitions[i]],
                init_time = [last_transitions[i]]))
        self.trajectories = new_trajectories
        time_trajectories_reset = time.time()
        print("Time to reset trajectories: ", time_trajectories_reset - time_start_dump)

        time_start_dump = time.time()
        # check if parquet files exist
        if not os.path.exists('states.parquet') and not os.path.exists('last_transitions.parquet'):
            
            ddf_states = dd.from_array(state_histories.T)
            ddf_time_histories = dd.from_array(time_histories.T)
            ddf_states.columns = [str(i) for i in range(num_trajectories)]
            ddf_time_histories.columns = [str(i) for i in range(num_trajectories)]
            
            ddf_states.to_parquet('states.parquet', engine='pyarrow')
            ddf_time_histories.to_parquet('last_transitions.parquet', engine='pyarrow')
        
        else:     
            read_start = time.time()
            ddf_states = dd.read_parquet('states.parquet')
            ddf_transitions = dd.read_parquet('last_transitions.parquet')
            # convert states_histories to dask dataframe
            ddf_states_histories = dd.from_array(state_histories.T)
            ddf_time_histories = dd.from_array(time_histories.T)
            ddf_states_histories.columns = [str(i) for i in range(num_trajectories)]
            ddf_time_histories.columns = [str(i) for i in range(num_trajectories)]
            dd.merge(ddf_states, ddf_states_histories)
            dd.merge(ddf_transitions, ddf_time_histories)
            read_end = time.time()
            print("Time to read from parquet: ", read_end - read_start)
            
            write_start = time.time()
            ddf_states.to_parquet('states.parquet', append=True, engine='pyarrow')
            ddf_transitions.to_parquet('last_transitions.parquet', append=True, engine='pyarrow')
            write_end = time.time()
            print("Time to write to parquet: ", write_end - write_start)

        time_dump_end = time.time()
            


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

    print("Loading from mat file: {}".format(file))
    print("Loading {} trajectories".format(mat_load.shape[0]))
    print("Starting at time {}".format(end_time))

    trajectories_loaded = [
        trajectory_from_list(i, start_time, end_time) for i in mat_load.T
    ]
    print("Loading {} trajectories".format(len(trajectories_loaded)))
    print("Starting at time {}".format(end_time))
    kmc_obj = kmc(
        draw_crit=draw_crit,
        initialization=None,
        energies=energies_mat,
        memory_friendly=False,
        checkpoint=True,
        time_stop=time_stop,
        trajectories=trajectories_loaded,
        final_save_prefix="saved_data",
        save_freq=1000,
    )

    return kmc_obj
