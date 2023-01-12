from kmcluster.core.kmc import kmc
from kmcluster.core.transition_conditions import rfkmc
from kmcluster.core.data import pull_json
from kmcluster.core.trajectory import trajectory


def test_kmc():
    parameters = pull_json('./parameters.json')

    # initialize kmc class
    kmc_boltz = kmc(
        steps=parameters['steps'],
        pop_size=parameters['size'],
        rates=parameters['rates'],
        draw_crit=parameters['draw_method_obj'],
        initialization=parameters['init_obj'],
    )
    # run calcs
    kmc_boltz.run(n_steps=10)


def test_kmc_stop():
    parameters = pull_json('./parameters.json')

    # initialize kmc class
    kmc_boltz = kmc(
        steps=parameters['steps'],
        pop_size=parameters['size'],
        rates=parameters['rates'],
        draw_crit=parameters['draw_method_obj'],
        initialization=parameters['init_obj'],
        time_stop=parameters['time_stop']
    )
    # run calcs
    kmc_boltz.run(n_steps=100)

    for i in kmc_boltz.trajectories: 
        assert i.transition_times[-2] < parameters['time_stop'], 'time stop failed'


def test_traj():
    traj_test = trajectory(0)
    crit_test = rfkmc()
    
    assert traj_test.states == [0], 'traj init failed'
    assert traj_test.transition_times == [0], 'traj init failed'
    
    traj_test.step([0.0,0.1], crit_test, time_stop=-1)
    print(traj_test.last_state())
    print(traj_test.get_history_as_dict())
    assert traj_test.last_state() == 1, 'traj step failed'
    assert len(traj_test.transition_times) == 2, 'traj step failed'
    
    dict_states = traj_test.get_history_as_dict()
    states, times = traj_test.get_history()
    assert states == [0, 1], 'traj get history failed'


def main():  
    test_traj()
    test_kmc()
    test_kmc_stop()
    print("all tests pass!")


main()