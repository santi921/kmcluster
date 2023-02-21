import json, os
import numpy as np
from tqdm import tqdm
from glob import glob
from kmcluster.core.trajectory import (
    trajectory, 
    trajectory_from_list, 
    sample_trajectory, 
    add_history_to_trajectory
)
from kmcluster.core.intialize import population_ind_to_trajectories
from kmcluster.core.transition_conditions import rfkmc, rkmc


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
        trajectories=None,
        save_freq=1000,
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

    def step(self):
        sum_warning = 0
        for traj in self.trajectories:
            # get last state in traj
            traj_last_ind = traj.last_state()

            if self.time_stop > 0:
                traj_last_time = traj.last_time()
                # print(traj_last_time)
                if traj_last_time > self.time_stop:
                    continue
                else:
                    # print("last traj_ind: " + str(traj_last_ind))
                    energies_from_i = self.energies[traj_last_ind]
                    warning = traj.step(
                        energies_from_i, self.draw_crit, time_stop=self.time_stop
                    )
                    sum_warning += warning
            else:
                energies_from_i = self.energies[traj_last_ind]
                warning = traj.step(
                    energies_from_i, self.draw_crit, time_stop=self.time_stop
                )
                sum_warning += warning

        if sum_warning > 1:
            print("Warning: trajectories in this step have steps sizes < 1e-15s\n")

    def run(self, n_steps=10):
        if n_steps == -1:
            self.step_count = 0
            # check if all trajectories have reached time_stop
            last_time_arr = np.array([i.last_time() for i in self.trajectories])

            while not all([i > self.time_stop for i in last_time_arr]):
                lowest_time = np.min(last_time_arr)

                if self.step_count % 100 == 0:
                    # print(
                    #    "Lowest time at step {}: {:.5f}\n".format(self.step_count, np.min(lowest_time)))
                    # print lowest time in scientific notation with 5 decimal places
                    print(
                        "Lowest time at step {}: {:.5e}\n".format(
                            self.step_count, lowest_time
                        )
                    )
                self.step_count = self.step_count + 1
                self.step()
                last_time_arr = np.array([i.last_time() for i in self.trajectories])

                if self.checkpoint:
                    if lowest_time > self.time_stop * self.save_ind / 10:
                        print("saving checkpoint at step {}".format(self.step_count))

                        time_save = self.time_stop * self.save_ind / 10

                        save_step = time_save / 100

                        self.save_as_matrix(
                            file="{}trajectories_{}_ckpt".format(
                                self.checkpoint_dir, self.save_ind
                            ),
                            start_time=0,
                            end_time=time_save,
                            step=save_step,
                            append=True,
                        )

                        self.save_ind = self.save_ind + 1

                if self.memory_friendly:
                    if lowest_time > self.time_stop * self.save_ind / 10:
                        print(
                            "coarsening trajectories at step {}\n".format(
                                self.step_count
                            )
                        )
                        time_save = self.time_stop * self.save_ind / 10
                        # time_save = np.min(self.time_stop * (self.save_ind - 1) / 10, 0)
                        save_step = time_save / (10 * self.save_ind)

                        traj_lists = []

                        for i in self.trajectories:
                            traj_lists.append(
                                sample_trajectory(i, 0, time_save, save_step)
                            )
                        traj_new = [
                            trajectory_from_list(i[0], 0, time_save) for i in traj_lists
                        ]
                        self.trajectories = traj_new
                        self.save_ind = self.save_ind + 1

            # save run
            start_time = 0
            end_time = self.time_stop
            step = float((end_time - start_time) / 1000)

            self.save_as_matrix(
                file="{}{}_trajectories_{}_ckpt".format(
                    self.checkpoint_dir, self.final_save_prefix, self.save_ind
                ),
                start_time=start_time,
                end_time=end_time,
                step=step,
                append=False,
            )

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

        mat_save = np.zeros(
            (self.pop_size, int(np.ceil((end_time - start_time) / step)))
        )
        sampling_array = np.arange(start_time, end_time, step)

        for ind_t, t in enumerate(sampling_array):
            for ind, i in enumerate(self.trajectories):
                mat_save[ind][ind_t] = i.state_at_time(t)
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
        with open(file, "w") as f:
            for t in range(start_time, end_time, step):
                master_dict[t] = self.get_state_dict_at_time(t)
            json.dump(master_dict, f)


# create class that inherits from kmc
class kmc_temp_ramp(kmc):
    def __init__(
        self,
        draw_crit,
        energies,
        memory_friendly,
        time_dict,
        initialization=None,
        checkpoint=False,
        checkpoint_dir="./checkpoints/",
        save_prefix="test_temp_ramp",
        trajectories=None,
        load=False,
        save_freq=1000,
    ):
        self.kb_const_ev = 8.617 * 10 ** (-5)
        self.draw_crit = draw_crit
        self.memory_friendly = memory_friendly
        self.energies = energies
        self.time_dict = time_dict
        self.initialization = initialization
        self.checkpoint = checkpoint
        self.save_prefix = save_prefix
        self.checkpoint_dir = checkpoint_dir
        self.pop_prop_hist = []
        self.save_ind = 1
        self.ramp_ind = 0 

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

    def step(self, draw_crit, time_stop):
        sum_warning = 0
        for traj in self.trajectories:
            # get last state in traj
            traj_last_ind = traj.last_state()

            if self.time_stop > 0:
                traj_last_time = traj.last_time()
                # print(traj_last_time)
                if traj_last_time > self.time_stop:
                    continue
                else:
                    # print("last traj_ind: " + str(traj_last_ind))
                    energies_from_i = self.energies[traj_last_ind]
                    warning = traj.step(
                        energies_from_i, draw_crit, time_stop=time_stop
                    )
                    sum_warning += warning
            else:
                energies_from_i = self.energies[traj_last_ind]
                warning = traj.step(
                    energies_from_i, draw_crit, time_stop=time_stop
                )
                sum_warning += warning

        if sum_warning > 1:
            print("Warning: trajectories in this step have steps sizes < 1e-15s\n")

    def run(self):
        time_dict = self.time_dict
        dict_ind = 0
        time_stop = 0
        
        for key, time_interval in time_dict.items():
            time_stop += time_interval 

            if self.draw_crit == "rfkmc":
                draw_crit_obj = rfkmc(self.kb_const_ev * float(key))
            elif self.draw_crit == "rkmc":
                draw_crit_obj = rkmc(self.kb_const_ev * float(key))
            else:
                raise ValueError("draw_crit must be rkmc or rfkmc")

            self.step_count = 0
            # check if all trajectories have reached time_stop
            last_time_arr = np.array([time_stop for i in self.trajectories])

            while not all([i > time_stop for i in last_time_arr]):
                lowest_time = np.min(last_time_arr)

                if self.step_count % 100 == 0:
                    print(
                        "Lowest time at step {}: {:.5e}\n".format(
                            self.step_count, lowest_time
                        )
                    )

                
                self.step_count = self.step_count + 1
                self.step(draw_crit_obj, time_stop)
                last_time_arr = np.array([time_stop for i in self.trajectories])


                if self.checkpoint:
                    if lowest_time > self.time_stop * self.save_ind / 2:
                        print("saving checkpoint at step {}".format(self.step_count))

                        time_save = self.time_stop * self.save_ind / 2
                        save_step = time_save / 500

                        self.save_as_matrix(
                            file="{}{}_trajectories_ckpt_{}".format(
                                self.checkpoint_dir, self.save_prefix, dict_ind
                            ),
                            start_time=0,
                            end_time=time_save,
                            step=save_step,
                            append=True,
                        )

                        self.save_ind = self.save_ind + 1

                if self.memory_friendly:
                    if lowest_time > time_stop * self.save_ind / 2:
                        print(
                            "coarsening trajectories at step {}\n".format(
                                self.step_count
                            )
                        )
                        time_save = time_stop * self.save_ind / 10
                        # time_save = np.min(self.time_stop * (self.save_ind - 1) / 10, 0)
                        save_step = time_save / (10 * self.save_ind)

                        traj_lists = []

                        for i in self.trajectories:
                            traj_lists.append(
                                sample_trajectory(i, 0, time_save, save_step)
                            )
                        traj_new = [
                            trajectory_from_list(i[0], 0, time_save) for i in traj_lists
                        ]
                        self.trajectories = traj_new
                        self.save_ind = self.save_ind + 1

            
            # save run
            start_time = 0
            end_time = self.time_stop
            step = float((end_time - start_time) / 1000)

            self.save_as_matrix(
                file="{}{}_trajectories_{}_ckpt".format(
                    self.checkpoint_dir, self.final_save_prefix, self.ramp_ind
                ),
                start_time=start_time,
                end_time=end_time,
                step=step,
                append=False,
            )
            self.ramp_ind = self.ramp_ind + 1

    def load(self):
        # TODO
        #retrieve all files with save_prefix in checkpoint_dir
        if self.checkpoint:
            files_checkpoint = glob(
                self.checkpoint_dir + self.save_prefix + "*"
            )
            ind_save = [i.split("/")[-1].split(".")[-2].split("_")[-1] for i in files_checkpoint]
            # get index of last save
            if len(ind_save) > 0:
                self.last_save_ind = max([int(i) for i in ind_save])

            #load trajectories from last save
            #self.





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
        memory_friendly=True,
        checkpoint=True,
        time_stop=time_stop,
        trajectories=trajectories_loaded,
        final_save_prefix="saved_data",
        save_freq=1000,
    )

    return kmc_obj
