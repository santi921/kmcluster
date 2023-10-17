import json
import pandas as pd
import numpy as np
from kmcluster.core.transition_conditions import rfkmc, rkmc
from kmcluster.core.intialize import random_init, boltz, global_minimum_only, selected

kb = 8.617 * 10 ** (-5)
import statistics as st


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


def test_dataset_1():
    """
    mess with this dataset to input your dataset
    return:
        Pt_H1_all: list of lists, each sublist is a barrier in the form [i,j,barrier]
    """

    Pt_H1_links = [
        [0, 1, 0.15],
        [0, 2, 0.61],
        [0, 3, 0.39],
        [2, 4, 0.27],
        [2, 6, 0.50],
        [2, 8, 0.66],
        [3, 8, 0.50],
        [5, 7, 0.52],
        [5, 9, 0.66],
        [5, 6, 0.66],
    ]

    Pt4H1_rawE = [
        -17.71720725,
        -17.68531409,
        -17.57336808,
        -17.50640668,
        -17.50097929,
        -17.50887522,
        -17.38155630,
        -17.25580137,
        -17.15164472,
        -17.13649884,
    ]

    H1_E = [Pt4H1_rawE[x] - Pt4H1_rawE[0] for x in range(0, len(Pt4H1_rawE))]

    Pt4H1_rev = []
    for i in range(0, len(Pt_H1_links)):
        Pt4H1_rev.append(
            [
                Pt_H1_links[i][1],
                Pt_H1_links[i][0],
                round(
                    (Pt_H1_links[i][2] + H1_E[Pt_H1_links[i][0]])
                    - H1_E[Pt_H1_links[i][1]],
                    2,
                ),
            ]
        )

    # relative energies
    H1_E = [Pt4H1_rawE[x] - Pt4H1_rawE[0] for x in range(0, len(Pt4H1_rawE))]
    # all the forward barriers
    Pt_H1_links = [
        [0, 1, 0.15],
        [0, 2, 0.61],
        [0, 3, 0.39],
        [2, 4, 0.27],
        [2, 6, 0.50],
        [2, 8, 0.66],
        [3, 8, 0.55],
        [5, 7, 0.52],
        [5, 9, 0.66],
        [5, 6, 0.66],
        [0, 6, 0.52],
        [1, 2, 0.28],
        [2, 5, 0.22],
        [3, 5, 0.22],
        [7, 8, 0.15],
        [8, 9, 0.14],
    ]
    # calculating the reverse barriers
    Pt4H1_rev = []
    for i in range(0, len(Pt_H1_links)):
        Pt4H1_rev.append(
            [
                Pt_H1_links[i][1],
                Pt_H1_links[i][0],
                round(
                    (Pt_H1_links[i][2] + H1_E[Pt_H1_links[i][0]])
                    - H1_E[Pt_H1_links[i][1]],
                    2,
                ),
            ]
        )
    # all barriers
    Pt_H1_all = Pt_H1_links + Pt4H1_rev
    return Pt_H1_all, H1_E


def test_dataset_2():
    Pt4H3_relE = [
        0.0,
        0.05265172000000007,
        0.08150556000000009,
        0.08234174999999766,
        0.10898511999999982,
        0.13161836000000093,
        0.15918047999999985,
        0.16211587000000094,
        0.17019705999999957,
        0.19589320000000043,
        0.2124428199999997,
        0.21772393999999906,
        0.23981415000000084,
        0.2480630899999987,
        0.24823677999999916,
        0.2504169300000001,
        0.26229379999999836,
        0.2676832299999994,
        0.2842788299999981,
        0.29314600000000013,
        0.2996583599999987,
        0.3080355399999988,
        0.3220917700000001,
        0.3320207499999981,
        0.3357182899999991,
        0.34816155999999765,
    ]
    Pt4H3_links = [
        [0, 3, 0.39],
        [1, 5, 0.27],
        [1, 16, 0.39],
        [2, 16, 0.55],
        [2, 19, 0.55],
        [3, 19, 0.23],
        [4, 18, 0.22],
        [6, 14, 0.17],
        [6, 8, 0.17],
        [7, 14, 0.14],
        [7, 24, 0.37],
        [12, 16, 0.23],
        [16, 19, 0.35],
        [20, 23, 0.45],
        [0, 10, 0.27],
        [0, 18, 0.74],
        [0, 22, 0.43],
        [1, 9, 0.17],
        [2, 13, 0.36],
        [3, 10, 0.34],
        [5, 14, 0.17],
        [5, 15, 0.31],
        [5, 25, 0.67],
        [12, 22, 0.61],
        [11, 18, 0.27],
        [15, 17, 0.16],
        [7, 17, 0.36],
        [20, 21, 0.26],
    ]
    # calculating the reverse barriers
    Pt4H3_rev = []
    for i in range(0, len(Pt4H3_links)):
        Pt4H3_rev.append(
            [
                Pt4H3_links[i][1],
                Pt4H3_links[i][0],
                round(
                    (Pt4H3_links[i][2] + Pt4H3_relE[Pt4H3_links[i][0]])
                    - Pt4H3_relE[Pt4H3_links[i][1]],
                    2,
                ),
            ]
        )
    # all barriers
    Pt4H3_all = Pt4H3_links + Pt4H3_rev
    return Pt4H3_all, Pt4H3_relE


def nice_print_clusters(ap, energies_list):
    for i in range(len(ap.cluster_centers_indices_)):
        print("Cluster %d" % i)
        for j in range(len(ap.labels_)):
            if ap.labels_[j] == i:
                print("State {} w energy: {:.3f}".format(j, energies_list[j]))
        print("")


def energy_to_rates(energies, temp, scale=1000):
    """
    convert energies to rates
    """
    rates = np.zeros(energies.shape)
    for i in range(len(energies)):
        for j in range(len(energies)):
            if energies[i][j] != 0:
                # rates[i][j] = scale * energies[i][j] / (kb * temp)
                rates[i][j] = (
                    scale
                    * (temp * kb / (4.1357 * 10**-15))
                    * np.exp(-energies[i][j] / (kb * temp))
                )

    return rates
