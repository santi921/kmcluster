import numpy as np 

from kmcluster.core.intialize import random_init, boltz, global_minimum_only, selected
from kmcluster.core.transition_conditions import rfkmc, rkmc
from kmcluster.core.data import energy_data, rate_mat_to_sparse, sparse_to_rate_mat

def main():

    steps = 100 # number of steps
    size = 100 # number of independet trajectories 
    n_states = 10 # number of states
    energies = [0, 1.0, 1.0, -1.0] # list of energies 
    init_dict = {0:0.5, 1:0.5} # dictionary of custom proportions to start with ({state:proportion,...})
    temp_boltz = 1.0  # temperature for boltzmann distribution + kmc
    
    rates_two_states = np.array([[0.0,0.1],[0.01, 0.0]])
    rates_four_states_sink = np.array([[0.0,0.0,0.0,0.1],[0.0,0.0,0.0,0.1],[0.0,0.0,0.0,0.1],[0.0,0.0,0.0,0.0]])
    rates_four_sym = np.array([[0.0,0.1,0.0,0.0],[0.0,0.0,0.1,0.0],[0.0,0.0,0.0,0.1],[0.1,0.0,0.0,0.0]])
    
    rates_as_sparse = [[0,1,0.1], [0,2,2.0]]


    parameters = {
        'steps': steps,
        'size': size,
        'n_states': n_states,
        'energies': energies,
        'draw_method': 'rfkmc',
        'init_method': 'random_init',
        'draw_obj': rfkmc(),
        'init_obj': random_init(size, n_states),
    }
    
    # convert between mat and back 
    sparse_test = rate_mat_to_sparse(rates_two_states)
    mat_test = sparse_to_rate_mat(sparse_test, 2)
    # compare mats
    assert np.array_equal(rates_two_states, mat_test), 'mat conversion failed'

    # convert between sparse and back
    mat_test_2 = sparse_to_rate_mat(rates_as_sparse, 3)
    sparse_test_2 = rate_mat_to_sparse(mat_test_2)
    # compare sparse
    assert rates_as_sparse == sparse_test_2, 'sparse conversion failed'

    # test all initializers 
    init_boltz = boltz(energies = energies, T=temp_boltz, size = size)
    init_global_min = global_minimum_only(energies=energies, size=size, n_states=n_states)
    init_select = selected(size=size, selected_proportions=init_dict, n_states=n_states)
    init_random = random_init(size, n_states)
    
    init_boltz.get_init_populations()
    init_global_min.get_init_populations()
    init_select.get_init_populations()
    init_random.get_init_populations()

    assert len(init_boltz.population) == (4), 'boltz init failed'
    assert len(init_global_min.population) == (4), 'global min init failed'
    assert len(init_select.population) == (n_states), 'selected init failed'
    assert len(init_random.population) == (n_states), 'random init failed'
    
    init_select_lazy = selected(selected_proportions={0:0.5, 1:0.5}, size=2, n_states=2)
    assert init_select_lazy.get_init_populations() == [1,1], 'selected init lazy failed'

    # test all draw conditions 
    draw_rfkmc = rfkmc()
    draw_rkmc = rkmc(r_0=1.0)
    print(draw_rfkmc.call(rates_four_states_sink[0]))
    print(draw_rkmc.call(rates_four_states_sink[0]))

    # test data + data utils
    e_data = energy_data(
        energies=energies, 
        rates=rates_four_sym
    )

    assert e_data.in_states(3) == 2, 'init state failed'
    assert e_data.out_states(3) == 0, 'out states failed'
    assert e_data.sum_rates_in(3) == 0.1, 'sum rates in failed'
    assert e_data.sum_rates_out(3) == 0.1, 'sum rates out failed'
    assert e_data.lowest_energy_state() == 3, 'lowest energy state failed'
    
    print("all tests passed!")
main()


