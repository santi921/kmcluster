import os, time
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

# from kmcluster.core.viz import compute_state_counts


class kmc:
    def __init__(
        self,
        draw_crit,
        energies,
        initialization=None,
        checkpoint=False,
        checkpoint_dir="./checkpoints/",
        ckptprefix="saved_data",
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
        self.ckptprefix = ckptprefix
        self.checkpoint_dir = checkpoint_dir
        self.pop_prop_hist = []
        self.save_ind = 1
        self.sample_index = 0
        self.n_states = energies.shape[0]
        self.batch_size = batch_size
        self.state_dict_file = state_dict_file

        # if checkpoint
        if checkpoint:
            # check if folder exists
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

        if sample_frequency == -1:
            self.sample_frequency = time_stop / 100
        else:
            self.sample_frequency = time_stop / sample_frequency

        self.results_mat = np.zeros(
            (self.n_states, 1 + int(self.time_stop / self.sample_frequency))
        )
        print("results mat shape: ", self.results_mat.shape)
        self.probe_status = [True] + [
            False for i in range(int(self.time_stop / self.sample_frequency) - 1)
        ]
        self.probe_times = np.array(
            [
                i * self.sample_frequency
                for i in range(int(self.time_stop / self.sample_frequency))
            ]
        )
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
            # state_dict(self.trajectories)
            self.pop_size = initialization.size

        print("done initializing....")

    def get_sampling(self):
        """
        Precomputes the random state and time samples for the trajectories - for speed
        """
        n_traj = len(self.trajectories)

        batch_size = self.batch_size
        rand_state_samples = uniform(0, 0.9999999, (n_traj, batch_size))
        rand_time_samples = uniform(0, 1, (n_traj, batch_size))
        neg_log_rand_time_samples = -np.log(rand_time_samples)

        rand_state_samples = np.float32(rand_state_samples)
        neg_log_rand_time_samples = np.float32(neg_log_rand_time_samples)

        return rand_state_samples, neg_log_rand_time_samples

    def task_batch_single(self, ind, queue):
        # print(ind, queue)
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
                ret_all=True,
            )
            queue.put(probe)
            # return_list.append(probe)
            return probe, current_state, current_time

            # if state > 0:
            #    self.results_mat[int(state), ind] += 1

    def step_batched_parallel(self):
        import multiprocessing

        manager = multiprocessing.Manager()
        # return_list = manager.list()
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
                chunksize=int(len(ind_list) / (n_threads)),
            )
            for ind, res in enumerate(task):
                # print(res)
                if res == -1:
                    if res[0] > 0:
                        self.results_mat[int(res[0]), ind] += 1
                    self.trajectories[ind].set_current_state(res[1])
                    self.trajectories[ind].set_current_time(res[2])

    def step_batched(self):
        self.rand_state_samples, self.neg_log_rand_time_samples = self.get_sampling()

        probe_start_list = []
        # traj_start_list = []
        # print("number of trajectories at batch start: ", len(self.trajectories))
        for ind, traj in enumerate(self.trajectories):
            traj_last_time = traj.last_time()
            traj_last_state = traj.get_current_state()

            if traj_last_time > self.time_stop:
                continue

            else:
                # traj_start_list.append(traj_last_state)

                probe_states, probe_ind = traj.batched_step(
                    self.draw_crit,
                    state_samples=self.rand_state_samples[ind],
                    neg_log_time_samples=self.neg_log_rand_time_samples[ind],
                    sample_frequency=self.sample_frequency,
                    time_stop=self.time_stop,
                    probe=False,
                )

                for ind_probe in range(len(probe_states)):
                    # print(probe_ind, probe_states)
                    if probe_ind[ind_probe] > -1:
                        self.results_mat[
                            int(probe_states[ind_probe]), int(probe_ind[ind_probe])
                        ] += 1
                # print("probe: ", probe_ind)
                # if probe_ind[0] > -1:
                #    probe_start_list.append(probe_states[0])

    def run(self, n_steps=10):
        time_list = []
        if n_steps == -1:
            trigger = False
            self.step_count = 0
            ind_tracker = 1
            last_time_arr = np.array([i.last_time() for i in self.trajectories])

            print("starting run")
            while np.min(last_time_arr) < self.time_stop:
                # print("batch")
                timer_start = time.time()
                self.step_batched()
                timer_end = time.time()
                time_list.append(timer_end - timer_start)

                if self.step_count > 10000 * ind_tracker:
                    lowest_time = np.min(last_time_arr)
                    mean_time = np.mean(last_time_arr)
                    ind_tracker += 1
                    print("-" * 40)
                    print(">>> step: {}".format(self.step_count))
                    print("Lowest time at step: {:.5e}".format(lowest_time))
                    print("mean time at step: {:.5e}".format(mean_time))
                    print("time to step: {}\n".format(np.mean(time_list)))
                    print("-" * 40)
                    # show first_column of results_mat
                    rolling_ind = bisect_right(self.probe_times, lowest_time)
                    print(
                        "rolling index: {} out of {}".format(
                            rolling_ind, len(self.probe_times)
                        )
                    )
                    # header = ["{:.1e}".format(i) for i in self.probe_times[rolling_ind-1:rolling_ind+15]]
                    # table = tabulate(self.results_mat[:,rolling_ind-1:rolling_ind+15], tablefmt="fancy_grid", headers=header)
                    header = [
                        "{:.1e}".format(i)
                        for i in self.probe_times[0 : rolling_ind + 15]
                    ]
                    table = tabulate(
                        self.results_mat[:, 0 : rolling_ind + 15],
                        tablefmt="fancy_grid",
                        headers=header,
                    )

                    print(table)
                    # print sum of first 26 columns,

                    print(
                        "rolling state sum: \n{}".format(
                            np.sum(
                                self.results_mat[:, rolling_ind - 1 : rolling_ind + 26],
                                axis=0,
                            )
                        )
                    )
                    self.save_as_dict(
                        "{}{}_trajectories_step_{}_ckpt.pkl".format(
                            self.checkpoint_dir, self.ckptprefix, self.step_count
                        )
                    )
                # print("step count: {}".format(self.step_count))
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
                        # save_step = time_save / self.coarsening_mesh
                        self.save_as_dict(
                            "{}{}_trajectories_{}_ckpt.pkl".format(
                                self.checkpoint_dir, self.ckptprefix, self.save_ind
                            )
                        )
                        trigger = True

                if trigger:
                    trigger = False
                    self.save_ind = self.save_ind + 1

            print("done with kmc run to stop time {}".format(self.time_stop))
            print("this took {} steps".format(self.step_count))
            # save run
            self.probe_status = [True for i in self.probe_status]
            # check is self.checkpoint exists
            if self.checkpoint and not os.path.exists(self.checkpoint_dir):
                os.mkdir(self.checkpoint_dir)

            self.save_as_dict(
                file="{}{}_trajectories_{}_final_ckpt.pkl".format(
                    self.checkpoint_dir, self.ckptprefix, self.save_ind
                )
            )

            # if lowest_time is None:
            # check if lowest_time is instantiated
            # if not, instantiate it

            lowest_time = np.min(last_time_arr)
            mean_time = np.mean(last_time_arr)
            print(
                "sum of results at each probe time: {}".format(
                    np.sum(self.results_mat, axis=0)
                )
            )
            print(
                "Lowest time at final step {}: {:.5e}".format(
                    self.step_count, lowest_time
                )
            )
            print("mean time at final step: {:.5e}\n".format(mean_time))

        else:
            for _ in tqdm(range(n_steps)):
                self.step()

    def save_as_dict(self, file):
        """
        Saves states to json file
        Takes:
            file: file to save to
        Returns:
            None
        """
        ret_dict = {}
        ret_dict["time_stop"] = self.time_stop
        ret_dict["population_size"] = self.pop_size
        ret_dict["sample_frequency"] = self.sample_frequency
        ret_dict["probe_status"] = self.probe_status
        ret_dict["results_mat"] = self.results_mat
        ret_dict["traj_times"] = [i.get_current_time() for i in self.trajectories]
        ret_dict["traj_states"] = [i.get_current_state() for i in self.trajectories]
        ret_dict["index_of_sample"] = [
            i.get_index_of_last_sample() for i in self.trajectories
        ]

        with open(file, "wb") as output:
            # Pickle dictionary using protocol 0.
            pkl.dump(ret_dict, output)

    def load_from_state_dict(self):
        """
        Initializes kmc object from state dictionary
        """
        trajectories = []
        print("-" * 20 + "Reload Module" + "-" * 20)
        with open(self.state_dict_file, "rb") as input:
            ret_dict = pkl.load(input)
        assert (
            ret_dict["sample_frequency"] == self.sample_frequency
        ), "my brother in christ the sampling freq must be the same as the save dict"
        assert (
            ret_dict["time_stop"] == self.time_stop
        ), "my brother in christ the time stop must be the same as the save dict"

        ########################################################
        ##### initializes the most current info on the run #####
        times_init = np.zeros(ret_dict["population_size"])
        traj_times = ret_dict["traj_times"]
        traj_states = ret_dict["traj_states"]
        traj_sample_index = ret_dict["index_of_sample"]
        lowest_time = np.min(traj_times)
        for time, state, sample_index in zip(
            traj_times, traj_states, traj_sample_index
        ):
            # traj_temp = ret_dict["traj_times"][ind]
            # times_init[ind] = traj_temp
            trajectories.append(
                trajectory_minimum(
                    init_state=state, init_time=time, index_of_last_sample=sample_index
                )
            )
        self.results_mat = np.array(ret_dict["results_mat"])
        self.trajectories = trajectories
        self.pop_size = len(trajectories)
        ########################################################

        ########################################################
        ########## initalizes probe states ##########
        self.probe_status = ret_dict["probe_status"]
        self.sample_frequency = ret_dict["sample_frequency"]
        self.time_stop = ret_dict["time_stop"]

        ########################################################
        print("done reloading from file {}".format(self.state_dict_file))
        print("loaded {} trajectories".format(len(trajectories)))
        print(
            "{}% of probe states are complete".format(
                100 * np.sum(self.probe_status) / len(self.probe_status)
            )
        )
        print("sample frequency is {}s".format(self.sample_frequency))

        # print("sum of results mat is {}".format(np.sum(self.results_mat)))

        print("slowest trajectory is {}".format(lowest_time))
        print("-" * 20 + "Reload Module" + "-" * 20)

    def plot_top_n_states_stacked(
        self,
        max_time,
        n_show=5,
        title=None,
        xlabel=None,
        ylabel=None,
        save=False,
        show=True,
        save_name="./output_top.png",
    ):
        """
        bin and count what states are in what bins
        Takes:
            list of trajectories objects
            resolution(float): bin size
            time_stop(float): time upper bound on counting
        """
        if n_show == -1:
            n_show = self.n_states

        count_dict = self.results_mat.T
        # convert to pandas
        # count_df = pd.DataFrame.from_dict(count_dict, orient='index')
        count_df = pd.DataFrame(count_dict)
        count_df = count_df.iloc[:-1]
        x_axis = np.linspace(0, max_time, len(count_df))
        # x_axis = np.arange(0, self.time_stop, self.sample_frequency)
        count_df.index = x_axis
        # add row names as states
        count_df.columns = np.arange(0, self.n_states)

        # sum all counts for each state
        sum_count = count_df.sum(axis=0)
        # get top n states
        keys_top_n = sum_count.nlargest(n_show).index.to_list()

        # make x_axis the same length as the number of bins
        # x_axis = np.arange(0)
        # counts_per_state = np.zeros((n_show, len(x_axis)))
        self.pop_size

        # sort keys
        keys_top_n.sort()
        df_show = count_df[keys_top_n]

        # divide by total population size
        df_show = df_show / self.pop_size

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
        max_time,
        title=None,
        xlabel=None,
        ylabel=None,
        save=False,
        show=True,
        save_name="./output_top.png",
    ):
        """
        bin and count what states are in what bins
        Takes:
            list of trajectories objects
            resolution(float): bin size
            time_stop(float): time upper bound on counting
        """

        count_dict = self.results_mat.T

        # convert to pandas
        # count_df = pd.DataFrame.from_dict(count_dict, orient='index')
        count_df = pd.DataFrame(count_dict)
        count_df = count_df.iloc[:-1]
        x_axis = np.linspace(0, max_time, len(count_df))
        # x_axis = np.arange(0, self.time_stop, self.sample_frequency)
        count_df.index = x_axis
        # add row names as states
        count_df.columns = np.arange(0, self.n_states)

        # get top n states
        x_axis = np.arange(0, self.time_stop, self.sample_frequency)

        df_show = count_df[states_to_plot]
        # divide by total population size
        df_show = df_show / self.pop_size
        fig = px.area(df_show, title=title, x=x_axis, y=df_show.keys())

        # set xlim
        fig.update_xaxes(range=[0, max_time])
        # set ylim
        fig.update_yaxes(range=[0, df_show.max()])
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

    def plot_select_states(
        self,
        states_to_plot,
        max_time,
        title=None,
        xlabel=None,
        ylabel=None,
        save=False,
        show=True,
        save_name="./output_top.png",
    ):
        """
        bin and count what states are in what bins
        Takes:
            list of trajectories objects
            resolution(float): bin size
            time_stop(float): time upper bound on counting
        """

        count_dict = self.results_mat.T

        # convert to pandas
        # count_df = pd.DataFrame.from_dict(count_dict, orient='index')
        count_df = pd.DataFrame(count_dict)
        # remove last row
        count_df = count_df.iloc[:-1]
        # add column names as times
        count_df = pd.DataFrame(count_dict)
        count_df = count_df.iloc[:-1]
        x_axis = np.linspace(0, max_time, len(count_df))
        # x_axis = np.arange(0, self.time_stop, self.sample_frequency)
        count_df.index = x_axis
        # add row names as states
        count_df.columns = np.arange(0, self.n_states)

        df_show = count_df[states_to_plot]
        # divide by total population size
        df_show = df_show / self.pop_size

        fig = px.line(df_show, title=title, x=x_axis, y=df_show.keys())

        # set xlim
        fig.update_xaxes(range=[0, max_time])
        # set ylim
        fig.update_yaxes(range=[0, max(df_show.max())])
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

    def plot_top_n_states(
        self,
        max_time,
        n_show=5,
        title=None,
        xlabel=None,
        ylabel=None,
        save=False,
        show=True,
        save_name="./output_top.png",
    ):
        """
        bin and count what states are in what bins
        Takes:
            list of trajectories objects
            resolution(float): bin size
            time_stop(float): time upper bound on counting
        """
        if n_show == -1:
            n_show = self.n_states

        count_dict = self.results_mat.T
        # convert to pandas
        # count_df = pd.DataFrame.from_dict(count_dict, orient='index')
        count_df = pd.DataFrame(count_dict)
        # remove last row
        count_df = count_df.iloc[:-1]
        # add column names as times

        x_axis = np.linspace(0, max_time, len(count_df))
        # x_axis = np.arange(0, self.time_stop, self.sample_frequency)
        count_df.index = x_axis
        # add row names as states
        count_df.columns = np.arange(0, self.n_states)

        # sum all counts for each state
        sum_count = count_df.sum(axis=0)
        # get top n states
        keys_top_n = sum_count.nlargest(n_show).index.to_list()

        x_axis = np.arange(0, self.time_stop, self.sample_frequency)
        
        df_show = count_df[keys_top_n]
        # divide by total population size
        df_show = df_show / self.pop_size

        # fig = px.area(df_show, title=title, x=x_axis, y=df_show.keys())
        fig = px.line(df_show, title=title, x=x_axis, y=df_show.keys())

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


def rates_to_cum_rates(rates):
    # print(len(rates))
    rates_cum = [rates[0]]
    for i in range(1, len(rates)):
        rates_cum.append(rates_cum[i - 1] + rates[i])

    return np.array(rates_cum)


def state_dict(trajectories):
    diag_dict = {}
    for traj in trajectories:
        if traj.get_current_state() in diag_dict:
            diag_dict[traj.get_current_state()] += 1
        else:
            diag_dict[traj.get_current_state()] = 1
    print("state probe diagnose: ", diag_dict)
