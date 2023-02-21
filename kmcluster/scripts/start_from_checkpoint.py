from kmcluster.core.kmc import kmc, load_kmc_from_matrix
from kmcluster.core.transition_conditions import rfkmc
from kmcluster.core.intialize import random_init, boltz, selected
from kmcluster.core.data import sparse_to_mat

from kmcluster.core.viz import (
    plot_top_n_states,
    plot_states,
    graph_trajectories_static,
    communities_static,
    compute_state_counts,
)


def get_merge_data():
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

    # raw energies
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


# main function
if __name__ == "__main__":
    # mess with this data to input yours, ill eventually make a read file
    Pt_H1_all, H1_E = get_merge_data()
    energies_mat = sparse_to_mat(Pt_H1_all)

    # initalize all the normal values
    temp_boltz = 100 * 8.617 * 10 ** (-5)
    k_b_ev = 8.614 * 10**-5
    rfkmc_obj = rfkmc(k_b_t=temp_boltz)
    init_boltz = boltz(energies=H1_E, T=temp_boltz, size=10000)
    init_random = random_init(10000, energies_mat.shape[0])
    # file to read from

    time_stop = 0.0002
    file = "./checkpoints/saved_data_trajectories_10_ckpt_start_0_end_1.00000e-04_step_1.00000e-07.npy"
    # initialize kmc object
    kmc_boltz = load_kmc_from_matrix(file, energies_mat, rfkmc_obj, time_stop)

    # run calcs
    trajectories = kmc_boltz.trajectories

    plot_top_n_states(
        trajectories,
        n_show=5,
        total_states=len(H1_E),
        resolution=0.000001,
        max_time=0.0001,
    )

    # plot_states(
    #    trajectories,
    #    states_to_plot=[0,1,2,3, 4, 5, 6, 7, 8,9],
    #    resolution = 0.0001,
    #    max_time = 0.001,
    #    title = "Test Run - Pt4H1",
    #    xlabel = "Time",
    #    ylabel = "State Counts",
    #    save = True,
    #    save_name = "../../reporting/plots/test_run_Pt4H1.png"
    # )

    # time_slice = 0
    # pos = graph_trajectories_static(
    #    trajectories=trajectories,
    #    energies = energies_mat,
    #    time=0.001,
    #    ret_pos = True
    # )
