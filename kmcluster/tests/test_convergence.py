from kmcluster.core.kmc import kmc
from kmcluster.core.transition_conditions import rfkmc
from kmcluster.core.intialize import boltz
from kmcluster.core.data import sparse_to_mat

def test_convergence():
    temp_boltz = 200 * 8.617 * 10**(-5) 
    rfkmc_obj=rfkmc(k_b_t=temp_boltz)
    energies = [-1, 0]
    energies_mat = sparse_to_mat([[0, 1, 0.01], [1, 0, 0.01]])
    #print(energies_mat)

    init_boltz = boltz(
        energies = energies, 
        T=temp_boltz, 
        size=1000)
    #print(init_boltz.get_init_populations())
    
    # initialize kmc object
    kmc_boltz = kmc(
        time_stop = 10, 
        energies=energies_mat,
        draw_crit=rfkmc_obj,
        initialization=init_boltz,
        memory_friendly=False,
        check_converged=True
    )


    kmc_boltz.run(n_steps=-1)
    trajectories = kmc_boltz.trajectories




def main():  
    test_convergence()
    print("all tests pass!")


main()