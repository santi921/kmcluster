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


# main function
if __name__ == "__main__":
    # mess with this data to input yours, ill eventually make a read file
    T_kelvin = 150
    Pt_H2_all, Pt4H2_relE = get_merge_data()
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