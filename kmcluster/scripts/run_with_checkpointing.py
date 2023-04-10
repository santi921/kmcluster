import numpy as np
from kmcluster.core.kmc import kmc
from kmcluster.core.transition_conditions import rfkmc
from kmcluster.core.intialize import random_init, boltz, selected
from kmcluster.core.data import sparse_to_mat



def get_merge_data():
    """
    mess with this dataset to input your dataset
    return:
        Pt_H1_all: list of lists, each sublist is a barrier in the form [i,j,barrier]
    """


    # relative energies
    Pt4H2_relE = [0.0,
        0.12968483999999947,
        0.15544876999999957,
        0.2135138399999974,
        0.24601814000000033,
        0.25703292000000033,
        0.2939078699999982,
        0.3176236199999991,
        0.3213298899999977,
        0.32533018999999896,
        0.3709225499999995,
        0.3738283999999972,
        0.3905175899999982,
        0.39488384999999937,
        0.4040561700000005,
        0.4339835000000001,
        0.46826942999999943,
        0.4787926299999974,
        0.5032997299999984,
        0.5751533699999989,
        0.6028533899999999,
        0.7178215399999992]


    # all the forward barriers
    PtH2_links=[[0, 9, 0.76],
     	[1, 9, 0.34],
     	[5, 10, 0.21],
     	[4, 11, 0.38],
     	[6, 19, 0.33],
     	[17, 19, 0.21],
     	[1, 7, 0.39],
     	[9, 15, 0.16],
     	[9, 13, 0.12],
     	[10, 13, 0.18],
     	[14, 21, 0.57],
     	[11, 17, 0.22],
     	[7, 20, 0.48],
     	[0, 4, 0.31],
     	[0, 6, 0.63],
     	[1, 2, 0.28],
     	[3, 8, 0.39],
     	[1, 18, 0.5],
     	[16, 17, 0.41],
     	[2, 12, 0.43],
     	[7, 9, 0.25]
	]

    # calculating the reverse barriers
    Pt4H2_rev = []
    for i in range(0, len(PtH2_links)):
        Pt4H2_rev.append(
            [
                PtH2_links[i][1],
                PtH2_links[i][0],
                round(
                    (PtH2_links[i][2] + Pt4H2_relE[PtH2_links[i][0]])
                    - Pt4H2_relE[PtH2_links[i][1]],
                    2,
                ),
            ]
        )
    # all barriers
    Pt_H2_all = PtH2_links + Pt4H2_rev
    return Pt_H2_all, Pt4H2_relE

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
    temp_boltz = T_kelvin * 8.617 * 10 ** (-5)
    k_b_ev = 8.614 * 10**-5

    rfkmc_obj = rfkmc(k_b_t=temp_boltz)
    init_boltz = boltz(energies=Pt4H2_relE, T=temp_boltz, size=10000)
    init_random = random_init(10000, energies_mat.shape[0])
    

    # initialize kmc object
    kmc_boltz = kmc(
        time_stop=0.0001,
        energies=energies_mat,
        draw_crit=rfkmc_obj,
        initialization=init_random,
        memory_friendly=True,
        checkpoint=False,
        checkpoint_dir="./checkpoints/",
        final_save_prefix="saved_data",
        coarsening_mesh=100 # this is the mesh for checkpointing/saving + memory friendly
    )

    print("running kmc")
    kmc_boltz.run(n_steps=-1)
    print("kmc done")
    kmc_boltz.save_as_dict("Pt4H2_g_150K_0.0001s.json",start_time=0,end_time=0.0001,step=0.000001)
    print("saving done")
    
    n_show = 20
    print("mpl plot - top states")
    kmc_boltz.plot_top_n_states(
        n_show=n_show,
        #total_states=len(Pt4H2_relE),
        resolution=0.000001,
        max_time=0.0001,
        title="State Distribution, {}K".format(T_kelvin),
        xlabel="Time (s)",
        ylabel="Population Proportion",
        save=True,
	    save_name="Pt4H2_g_150K_top{}.png".format(n_show)
    )
    
    """
    print("plotly plot - top states")
    
    kmc_boltz.plot_top_n_states_stacked(
    	n_show = 5,
    	resolution=0.000001, 
    	max_time=None, 
    	title="State Distribution, {}K".format(T_kelvin),
        xlabel="Time (s)",
        ylabel="Population Proportion", 
    	save=True, 
        show=True,
    	save_name="Pt4H2_g_stacked_150K.png")

    print("plotly plot - select states")

    kmc_boltz.plot_select_states_stacked(
        states_to_plot=[0, 1, 2, 5, 6],
        resolution=0.000001, 
        max_time=None, 
    	title="State Distribution, {}K".format(T_kelvin),
        xlabel="Time (s)",
        ylabel="Population Proportion",     
     	save=True, 
        show=True,
        save_name="Pt4H2_g_stacked_select_150K.png"
    )
    """