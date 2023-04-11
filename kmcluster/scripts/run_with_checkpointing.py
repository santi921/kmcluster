from kmcluster.core.kmc_minimum import kmc 
from kmcluster.core.transition_conditions import rfkmc
from kmcluster.core.intialize import random_init, boltz
from kmcluster.core.data import sparse_to_mat

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


def get_merge_data_new(): 
    Pt4H3_relE=[0.0,
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
        0.34816155999999765]
    Pt4H3_links=[[0, 3, 0.39],
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
     [20, 21, 0.26]]
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


# main function
if __name__ == "__main__":
    # mess with this data to input yours, ill eventually make a read file
    T_kelvin = 100
    Pt_H2_all, Pt4H2_relE = get_merge_data_new()
    energies_mat = sparse_to_mat(Pt_H2_all)
    energies_nonzero = energies_mat[energies_mat != 0]
    print("smallest barrier: ", min(energies_nonzero))
    print("running at T = ", T_kelvin, "K")
    temp_boltz = T_kelvin * 8.617 * 10 ** (-5)


    rfkmc_obj = rfkmc(k_b_t=temp_boltz, energy_mat=energies_mat)
    init_boltz = boltz(energies=Pt4H2_relE, T=temp_boltz, size=10000)
    init_random = random_init(10000, energies_mat.shape[0])
    time_stop = 0.0001

    # initialize kmc object
    kmc_boltz = kmc(
        time_stop=time_stop,
        energies=energies_mat,
        draw_crit=rfkmc_obj,
        initialization=init_random,
        checkpoint=True,
        sample_frequency=1000,
        ckptprefix="Pt4H2_g_{}_".format(T_kelvin),
        checkpoint_dir="./checkpoints/", # change this to organize runs
        batch_size=1000, 
    )
    
    print("running kmc")
    kmc_boltz.run(n_steps=-1)
    print("kmc done")
    
    n_show = -1
    print("mpl plot - top states")
    
    # plot line - select
    kmc_boltz.plot_select_states(
        states_to_plot=[0, 1, 2, 5, 6],
        max_time=0.0001,
        title="State Distribution, {}K".format(T_kelvin),
        xlabel="Time (s)",
        ylabel="Population Proportion",
        save=True,
	    save_name="./plots/Pt4H2_g_{}_top{}.png".format(T_kelvin, n_show)
    )
    
    
    # stacked - top n
    
    kmc_boltz.plot_top_n_states_stacked(
    	n_show = -1,
    	max_time=0.0001, 
    	title="State Distribution, {}K".format(T_kelvin),
        xlabel="Time (s)",
        ylabel="Population Proportion", 
    	save=True, 
        show=True,
    	save_name="./plots/Pt4H2_stacked_{}_top{}.png".format(T_kelvin, n_show)
    )

    # stacked - select

    kmc_boltz.plot_select_states_stacked(
        states_to_plot=[0, 1, 2, 5, 6],
        max_time=0.0001, 
    	title="State Distribution, {}K".format(T_kelvin),
        xlabel="Time (s)",
        ylabel="Population Proportion",     
     	save=True, 
        show=True,
        save_name="./plots/Pt4H2_g_stacked_select_{}.png".format(T_kelvin)
    )
    
    # plot line - top n

    kmc_boltz.plot_top_n_states(
    	n_show = -1,
    	max_time=0.0001, 
    	title="State Distribution, {}K".format(T_kelvin),
        xlabel="Time (s)",
        ylabel="Population Proportion", 
    	save=True, 
        show=True,
    	save_name="./plots/Pt4H2_stacked_{}_top{}.png".format(T_kelvin, n_show)
    )
