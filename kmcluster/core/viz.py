
import copy
import sys
import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import networkx as nx
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button


def set_node_facecolor(G, **kwargs):
    """
    Changes node's facecolor
    Parameters
    ----------
    G     : Graph whose node's facecolor needs to be changed
    kwargs
    ------
    node  : The node whose facecolor needs to be changed. If node not specified
            and only one color is specified changes all the nodes to that
            facecolor. If node not specified and color is a list or tuple of
            colors, each node will be changed to its corresponding color.
    color : Can be a single value of a list of colors.
    The following color abbreviations are supported:
    ==========  ========
    character   color
    ==========  ========
    'b'         blue
    'g'         green
    'r'         red
    'c'         cyan
    'm'         magenta
    'y'         yellow
    'k'         black
    'w'         white
    ==========  ========
    Examples:
    set_node_facecolor(G, node=1, color='b')
    set_node_facecolor(G, node=1, color=['b'])
    set_node_facecolor(G, color=['r', 'b', 'g', 'y', 'm', 'k', 'c'])
    set_node_facecolor(G, color='b')
    set_node_facecolor(G, color=('b'))
    """
    try:
        node = kwargs['node']
    except KeyError:
        node = None
    try:
        color = kwargs['color']
    except KeyError:
        print("color argument required")
        sys.exit(0)

    if (isinstance(color, list) or isinstance(color, tuple)) and \
            len(color) == 1:
        color = color[0]

    same_colors = {'blue': 'b',
                   'green': 'g',
                   'red': 'r',
                   'cyan': 'c',
                   'magenta': 'm',
                   'yellow': 'y',
                   'black': 'k',
                   'white': 'w'}

    if isinstance(color, str) and len(color) != 1:
        color = same_colors[color]

    colors_dict = {'b': (0.,    0.,    1.,  1.),
                   'g': (0.,    0.5,   0.,  1.),
                   'r': (1.,    0.,    0.,  1.),
                   'c': (0.,    0.75,  0.75,  1.),
                   'm': (0.75,  0.,    0.75,  1.),
                   'y': (0.75,  0.75,  0.,  1.),
                   'k': (0.,    0.,    0.,  1.),
                   'w': (1.,    1.,    1.,  1.)}

    fig = plt.gcf()
    axes = plt.gca()
    no_of_nodes = G.number_of_nodes()
    nodes_collection = axes.get_children()[no_of_nodes + 2]

    # if node is specified manually changing the facecolor array
    # so that the colors for other nodes are retained
    if node:
        node_index = G.nodes().index(node)
        facecolor_array = nodes_collection.get_facecolor().tolist()
        facecolor_array = [tuple(x) for x in facecolor_array]
        if len(facecolor_array) == 1:
            facecolor_array = [copy.deepcopy(facecolor_array[0])
                               for i in range(no_of_nodes)]
        facecolor_array[node_index] = colors_dict[color]
        nodes_collection.set_facecolor(facecolor_array)

    # if node not specified call the matplotlib's set_facecolor function
    else:
        nodes_collection.set_facecolor(color)

    plt.draw()


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


def graph_trajectories_static(frame, trajectories, rates, ret_pos = False, pos=None, ax=None): 
    """
    given a list of trajectories, plot the state of at time_to_plot as a graph
    """
    # make matrix whether there is a rate connecting two states
    G = nx.DiGraph()

    rates_binary = np.zeros((len(rates), len(rates)))
    
    for i in range(len(rates)):
        for j in range(len(rates)): 
            if rates[i][j] > 0: 
                G.add_weighted_edges_from(
                    [(i, j, rates[i][j])], 
                    label=round(rates[i][j], 2)
                    )
                rates_binary[i][j] = 1
    
    counts = {}
    for traj in trajectories:
        state = traj.state_at_time(frame)
        if state in counts:
            counts[state] += 1
        else:
            counts[state] = 1
    
    
    # sort counts by key 
    counts = dict(sorted(counts.items()))
    counts_list = list(counts.values())
    counts_transformed = [400+(i/10) for i in counts_list]
    
    if pos is None:
        pos = nx.spring_layout(G)
    
    if ax != None:
        nodes = nx.draw_networkx_nodes(
            G, 
            pos, 
            node_color=counts_transformed, 
            node_size=counts_transformed,
            cmap=plt.cm.YlOrRd, 
            ax=ax)
        # Set edge color to red
        nodes.set_edgecolor('r')
        nx.draw_networkx_edges(
            G, 
            pos,
            arrowstyle='-|>',
            arrowsize=20, 
            ax=ax)
        # Uncomment this if you want your labels
        nx.draw_networkx_labels(G, pos, ax=ax)
        # edge labels
        nx.draw_networkx_edge_labels(
            G, 
            pos, 
            edge_labels=nx.get_edge_attributes(G, 'label'),
            ax=ax)
    else: 
        nodes = nx.draw_networkx_nodes(
            G, 
            pos, 
            node_color=counts_transformed, 
            node_size=counts_transformed,
            cmap=plt.cm.YlOrRd, 
            )
        # Set edge color to red
        nodes.set_edgecolor('r')
        nx.draw_networkx_edges(
            G, 
            pos,
            arrowstyle='-|>',
            arrowsize=20, 
            )
        # Uncomment this if you want your labels
        nx.draw_networkx_labels(G, pos, ax=ax)
        # edge labels
        nx.draw_networkx_edge_labels(
            G, 
            pos, 
            edge_labels=nx.get_edge_attributes(G, 'label'),
            )

    plt.show()
    if ret_pos: 
        return pos


def single_frame(frame, trajectories, rates, pos, n_states, ax=None): 
    """
    given a list of trajectories, plot the state of at time_to_plot as a graph
    """
    # make matrix whether there is a rate connecting two states
    axis = plt.gca()
    axis.clear()
    #print(frame)
    G = nx.DiGraph()

    rates_binary = np.zeros((len(rates), len(rates)))
    for i in range(len(rates)):
        for j in range(len(rates)): 
            if rates[i][j] > 0: 
                G.add_weighted_edges_from(
                    [(i, j, rates[i][j])], 
                    label=round(rates[i][j], 2)
                    )
                rates_binary[i][j] = 1
    
    counts = {i:0 for i in range(n_states)}
    for traj in trajectories:
        state = traj.state_at_time(frame)
        if state in counts:
            counts[state] += 1
        else:
            counts[state] = 1
    
    
    # sort counts by key 
    counts = dict(sorted(counts.items()))
    counts_list = list(counts.values())
    counts_transformed = [400+(i) for i in counts_list]
    
    if ax != None:
        nodes = nx.draw_networkx_nodes(
            G, 
            pos, 
            node_color=counts_transformed, 
            node_size=counts_transformed,
            cmap=plt.cm.YlOrRd, 
            ax=ax)
        # Set edge color to red
        nodes.set_edgecolor('r')
        e = nx.draw_networkx_edges(
            G, 
            pos,
            arrowstyle='-|>',
            arrowsize=20, 
            ax=ax)
        # Uncomment this if you want your labels
        n_labels = nx.draw_networkx_labels(G, pos, ax=ax)
        # edge labels
        e_labels = nx.draw_networkx_edge_labels(
            G, 
            pos, 
            edge_labels=nx.get_edge_attributes(G, 'label'),
            ax=ax)
    else: 
        nodes = nx.draw_networkx_nodes(
            G, 
            pos, 
            node_color=counts_transformed, 
            node_size=counts_transformed,
            cmap=plt.cm.YlOrRd, 
            )
        # Set edge color to red
        nodes.set_edgecolor('r')
        e = nx.draw_networkx_edges(
            G, 
            pos,
            arrowstyle='-|>',
            arrowsize=20, 
            )
        # Uncomment this if you want your labels
        n_labels = nx.draw_networkx_labels(G, pos, ax=ax)
        # edge labels
        e_labels = nx.draw_networkx_edge_labels(
            G, 
            pos, 
            edge_labels=nx.get_edge_attributes(G, 'label'),
            )

    return nodes, e, n_labels, e_labels,


def single_frame_slider(frame, trajectories, rates, pos, n_states, ax=None): 
    """
    given a list of trajectories, plot the state of at time_to_plot as a graph
    """
    
    axis = plt.gca()
    axis.clear()
    
        
    
    #print(frame)
    G = nx.DiGraph()

    rates_binary = np.zeros((len(rates), len(rates)))
    for i in range(len(rates)):
        for j in range(len(rates)): 
            if rates[i][j] > 0: 
                G.add_weighted_edges_from(
                    [(i, j, rates[i][j])], 
                    label=round(rates[i][j], 2)
                    )
                rates_binary[i][j] = 1
    
    counts = {i:0 for i in range(n_states)}
    for traj in trajectories:
        state = traj.state_at_time(frame)
        if state in counts:
            counts[state] += 1
        else:
            counts[state] = 1
    
    
    # sort counts by key 
    counts = dict(sorted(counts.items()))
    counts_list = list(counts.values())
    counts_transformed = [400+(i) for i in counts_list]
    
    if ax != None:
        nodes = nx.draw_networkx_nodes(
            G, 
            pos, 
            node_color=counts_transformed, 
            node_size=counts_transformed,
            cmap=plt.cm.YlOrRd, 
            ax=ax)
        # Set edge color to red
        nodes.set_edgecolor('r')
        e = nx.draw_networkx_edges(
            G, 
            pos,
            arrowstyle='-|>',
            arrowsize=20, 
            ax=ax)
        # Uncomment this if you want your labels
        n_labels = nx.draw_networkx_labels(G, pos, ax=ax)
        # edge labels
        e_labels = nx.draw_networkx_edge_labels(
            G, 
            pos, 
            edge_labels=nx.get_edge_attributes(G, 'label'),
            ax=ax)
    else: 
        nodes = nx.draw_networkx_nodes(
            G, 
            pos, 
            node_color=counts_transformed, 
            node_size=counts_transformed,
            cmap=plt.cm.YlOrRd, 
            )
        # Set edge color to red
        nodes.set_edgecolor('r')
        e = nx.draw_networkx_edges(
            G, 
            pos,
            arrowstyle='-|>',
            arrowsize=20, 
            )
        # Uncomment this if you want your labels
        n_labels = nx.draw_networkx_labels(G, pos, ax=ax)
        # edge labels
        e_labels = nx.draw_networkx_edge_labels(
            G, 
            pos, 
            edge_labels=nx.get_edge_attributes(G, 'label'),
            )

    #return nodes, e, n_labels, e_labels,


def graph_pos(rates): 
    """
    given a list of trajectories, plot the state of at time_to_plot as a graph
    """
    # make matrix whether there is a rate connecting two states
    G = nx.DiGraph()

    rates_binary = np.zeros((len(rates), len(rates)))
    
    for i in range(len(rates)):
        for j in range(len(rates)): 
            if rates[i][j] > 0: 
                G.add_weighted_edges_from(
                    [(i, j, rates[i][j])], 
                    label=round(rates[i][j], 2)
                    )
                rates_binary[i][j] = 1
    
    pos = nx.spring_layout(G)

    return pos 


def get_node_info_at_time(trajectories, time, shift = 400, scale = 10): 
    counts = {}
    for traj in trajectories:
        state = traj.state_at_time(time)
        if state in counts:
            counts[state] += 1
        else:
            counts[state] = 1
    
    
    # sort counts by key 
    counts = dict(sorted(counts.items()))
    counts_list = list(counts.values())
    counts_transformed = [shift+(i/scale) for i in counts_list]
    return counts_transformed


def graph_trajectories_dynamic(trajectories, rates, time_max, n_states, file_name): 
    """
    given a list of trajectories, plot as an animated graph from 0 to time_to_plot
    """

    pos = graph_pos(rates)

    fig = plt.gcf()
    ani = animation.FuncAnimation(fig, 
        single_frame, 
        fargs=(trajectories, rates, pos, n_states),
        blit=False, frames = np.arange(0,time_max,1), interval=10, repeat=True)
    file_name_25 = file_name + '_25.gif'
    file_name_10 = file_name + '_10.gif'
    ani.save(file_name_10, writer='imagemagick', fps=10)
    ani.save(file_name_25, writer='imagemagick', fps=25)
        

def graph_traj_plotly(trajectories, rates, time_max, file_name):
    import plotly.express as px
    df = px.data.gapminder()
    px.scatter(
        df, 
        x="gdpPercap", 
        y="lifeExp", 
        animation_frame="year", 
        animation_group="country",
        size="pop", 
        color="continent", 
        hover_name="country",
        log_x=True, 
        size_max=55, 
        range_x=[100,100000], 
        range_y=[25,90])
    

def communities_static(trajectories, time_to_plot): 
    """
    given a list of trajectories, plot the state of at time_to_plot as a graph w/ labelled communities
    """
    pass


def graph_slider(trajectories, rates, time_max, n_states, file_name):
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.1, bottom=0.3)

    # Make a horizontal slider to control the frequency.
    axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])

    freq_slider = Slider(
        axfreq,
        'Frame',
        0.0,
        time_max,
        1.0,
        handle_style = {'facecolor':'red'}, 
        track_color="lightgrey", 
        facecolor="lightgrey",
        initcolor='red'
    )

    pos = graph_pos(rates)

    def helper_plot(init_frame): 
        #reset ax 
        ax.clear()
        single_frame_slider(init_frame, trajectories, rates, pos, n_states, ax)

    helper_plot(init_frame=0)
    freq_slider.on_changed(helper_plot)
    
    fig.canvas.draw_idle()
    plt.show()

