
import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


def compute_state_counts(trajectories, resolution, max_time):
    """
    bin and count what states are in what bins
    Takes: 
        list of trajectories objects
        resolution(float): bin size
        time_stop(float): time upper bound on counting
    """

    states = []
    for i in np.arange(0, max_time, resolution): 
        for traj in trajectories: 
            states.append(traj.state_at_time(i))
    
    states_active = list(set(states))
    states_np = np.array(states)
    count_dict = {state:  np.count_nonzero(states_np == state) for state in states_active}
    return count_dict
    

def plot_top_n_states(trajectories, total_states, n_show = 5, max_time = 100, resolution = 0.1): 
    """
    given a list of trajectory objects and n plot the dynamics of the top n states
    
    Takes: 
        trajectories: list of trajectory objects
        n(int): number of states to plot
        time_stop(float): time upper bound on plotting
    """
    resolution = 0.1
    count_dict = compute_state_counts(trajectories, resolution, max_time)   
    
    keys_top_n = sorted(count_dict, key=count_dict.get, reverse=True)[:n_show]
    
    x_axis = np.arange(0, max_time, resolution)
    counts_per_state = np.zeros((n_show, len(x_axis)))

    for traj in trajectories:
        for ind, i in enumerate(x_axis): 
            state = traj.state_at_time(i)
            if state in keys_top_n: 
                counts_per_state[keys_top_n.index(state), ind] += 1

    for i in range(n_show):
        #cubic_interpolation_model = make_interp_spline(x_axis, counts_per_state[i,:]/total_states)
        #plt.plot(x_axis, cubic_interpolation_model(x_axis), label = "spline")
        plt.plot(x_axis, counts_per_state[i,:]/total_states, label = keys_top_n[i])

        #plt.plot(x_axis, counts_per_state[i,:]/total_states, label = keys_top_n[i])
    
    plt.legend()
    plt.show()


def graph_trajectories_static(trajectories, time_to_plot): 
    """
    given a list of trajectories, plot the state of at time_to_plot as a graph
    """
    pass


def graph_trajectories_dynamic(trajectories, time_to_plot): 
    """
    given a list of trajectories, plot as an animated graph from 0 to time_to_plot
    """
    pass


def communities_static(trajectories, time_to_plot): 
    """
    given a list of trajectories, plot the state of at time_to_plot as a graph w/ labelled communities
    """
    pass


