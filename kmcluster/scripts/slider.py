from kmcluster.core.data import sparse_to_rate_mat, pull_json
from kmcluster.core.kmc import kmc
from kmcluster.core.transition_conditions import rfkmc
from kmcluster.core.intialize import boltz
from kmcluster.core.viz import graph_slider


def main():

    Pt_H1_links=[[0,1,0.15],[0,2,0.61],[0,3,0.39],[2,4,0.27],[2,6,0.50],[2,8,0.66],[3,8,0.50],[5,7,0.52],[5,9,0.66],[5,6,0.66]]
    H_H1_links=[[0,6,0.52],[1,2,0.28],[2,5,0.22],[3,5,0.22],[7,8,0.15],[8,9,0.14]]
    Pt4H1_rawE=[-17.71720725 ,-17.68531409 ,-17.57336808 ,-17.50640668,-17.50097929,-17.50887522,-17.38155630,-17.25580137,-17.15164472,-17.13649884]
    H1_E=[Pt4H1_rawE[x]-Pt4H1_rawE[0] for x in range(0,len(Pt4H1_rawE))]

    rates_mat = sparse_to_rate_mat(Pt_H1_links, len(Pt4H1_rawE))

    rfkmc_obj=rfkmc()

    init_boltz = boltz(
        energies = H1_E, 
        T=10, 
        size=10000)

    # initialize kmc object
    kmc_boltz = kmc(
        steps=1000,
        time_stop = 1000, 
        pop_size=1000,
        rates=rates_mat,
        draw_crit=rfkmc_obj,
        initialization=init_boltz,
    )

    # run calcs
    kmc_boltz.run(n_steps=1000)
    trajectories = kmc_boltz.trajectories


    trajectories=trajectories
    rates=rates_mat
    time_max=1000
    n_states=len(H1_E)
    file_name='test.html'

    graph_slider(
        trajectories=trajectories, 
        rates=rates_mat, 
        time_max=1000, 
        n_states=len(H1_E), 
        file_name='test.html')

main()