from kmcluster.core.kmc import load_kmc_from_matrix
from kmcluster.core.transition_conditions import rfkmc
from kmcluster.core.intialize import boltz
from kmcluster.core.data import sparse_to_mat

def test_reload():
    temp_boltz = 7 * 8.617 * 10**(-5) 
    rfkmc_obj=rfkmc(k_b_t=temp_boltz)
    energies = [-1, 0, 0.1]
    energies_mat = sparse_to_mat([[0, 1, 0.01], [1, 0, 0.01]], num_states=3)
    print(energies_mat)

    file = "test_start_0_end_1_step_0.1.npy"

    
    # initialize kmc object
    kmc_loaded = load_kmc_from_matrix(
        file, 
        energies_mat=energies_mat,
        draw_crit=rfkmc_obj,
        time_stop=2
    )

    kmc_loaded.run(n_steps=-1)


def main():  
    test_reload()
    print("all tests pass!")


main()