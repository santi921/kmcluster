import json
import pandas as pd
import numpy as np
from kmcluster.core.transition_conditions import rfkmc, rkmc
from kmcluster.core.intialize import random_init, boltz, global_minimum_only, selected


class energy_data:
    def __init__(self, energies=None, rates=None, file_energies=None, file_rates=None):
        # assert that either energies or file_energies is not None
        assert (energies is not None) or (
            file_energies is not None
        ), "must provide energies list or filename for energies"
        # assert that either rates or file_rates is not None
        assert (rates is not None) or (
            file_rates is not None
        ), "must provide rate matrix or filename for rates"

        if rates is None:
            rates = read_rates(file_rates)
        if energies is None:
            energies = read_energies(file_energies)

        self.energies = energies
        self.rates = rates

    def lowest_energy_state(self):
        """
        return index of lowest energy state
        """
        return np.argmin(self.energies)

    def in_states(self, ind):
        """
        return index of states that have positive values from state of interest
        Takes:
            ind(int): index of state of interest
        Returns:
            index of states with positive rates to ind
        """
        return np.where(self.rates[:, ind] > 0)[0]

    def out_states(self, ind):
        """
        return index of states that have positive values from state of interest
        Takes:
            ind(int): index of state of interest
        Returns:
            index of states with positive rates from ind
        """
        return np.where(self.rates[ind, :] > 0)[0]

    def sum_rates_out(self, ind):
        """
        sum rates out of state
        Takes:
            ind(ind): state of interest
        Returns:
            (float) sum of rates out of state of interest
        """
        return np.sum(self.rates[ind, :])

    def sum_rates_in(self, ind):
        """
        sum rates into state
        Takes:
            ind(ind): state of interest
        Returns:
            (float) sum of rates into state of interest
        """
        return np.sum(self.rates[:, ind])


def read_energies(filename):
    # read csv in filename
    df = pd.read_csv(filename, header=None, index_col=None)
    energies = df.to_numpy()
    return energies


def read_rates(filename):
    # read matrix csv in filename
    df = pd.read_csv(filename, header=None, index_col=None)
    mat_ret = df.to_numpy()
    return mat_ret


def sparse_to_mat(sparse_mat, num_states=None):
    """
    sparse encoding to rate matrix
    Takes
        sparse_mat(list of lists) - transitions in format [ind_state_from, ind_state_to, rate]
    Returns
        rate_mat(numpy array) - square rate matrix with row start, column end state
    """
    # init square zero matrix
    # create a list of all states before creating a mat
    list_of_states = []

    for i in sparse_mat:
        state_a = i[0]
        state_b = i[1]
        if state_a not in list_of_states:
            list_of_states.append(state_a)
        if state_b not in list_of_states:
            list_of_states.append(state_b)
    list_of_states.sort()
    if num_states is None:
        num_states = len(list_of_states)
    else:
        num_states = num_states

    rate_mat = np.zeros((num_states, num_states))
    for row in sparse_mat:
        rate_mat[row[0], row[1]] = row[2]

    return rate_mat


def mat_to_sparse(matrix):
    """
    rate matrix to sparse matrix
    Takes:
        rate_mat(numpy array) - square mat w/ rates
    Returns:
        ret_list(list of lists) - rates in format [ind_state_from, ind_state_end, rate]
    """
    ret_list = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > 0:
                ret_list.append([i, j, matrix[i, j]])

    return ret_list


def pull_json(filename):
    """
    pull json file
    Takes:
        filename(str) - name of json file
    Returns:
        dictionary from json
    """
    with open(filename) as f:
        data = json.load(f)

    data["energies"] = np.array(data["energies"])
    data["rates"] = np.array(data["rates"])

    if data["draw_method"] == "rkmc":
        if type(data["rates"]) == list or type(data["rates"]) == np.ndarray:
            data["draw_method_obj"] = rkmc(r_0=r_0)
        else:
            rates_list = read_rates(data["rates"])
            data["rates"] = rates_list
        assert "r_0" in data, "must provide r_0 for rkmc"
        r_0 = data["r_0"]
        data["draw_method_obj"] = rkmc(r_0=r_0)

    else:
        if type(data["rates"]) == list or type(data["rates"]) == np.ndarray:
            data["draw_method_obj"] = rfkmc()
        else:
            rates_list = read_rates(data["rates"])
            data["rates"] = rates_list
            data["draw_method_obj"] = rfkmc()

    if data["init_method"] == "random":
        data["init_obj"] = random_init(size=data["size"], n_states=data["n_states"])

    elif data["init_method"] == "boltz":
        data["init_obj"] = boltz(
            energies=data["energies"], T=data["t_boltz"], size=data["size"]
        )
    elif data["init_method"] == "global_minimum_only":
        data["init_obj"] = global_minimum_only(
            energies=data["energies"], size=data["size"]
        )
    elif data["init_method"] == "selected":
        data["init_obj"] = selected(
            init_states=data["init_state_proportion"],
            size=data["size"],
            n_states=data["n_states"],
        )
    else:
        print("invalid init method")
        return {}

    return data
