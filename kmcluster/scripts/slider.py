from kmcluster.core.data import sparse_to_rate_mat, pull_json
import numpy as np 
from kmcluster.core.kmc import kmc
from kmcluster.core.transition_conditions import rfkmc
from kmcluster.core.intialize import boltz
from kmcluster.core.viz import single_frame_slider, graph_pos
from matplotlib.widgets import Slider, Button
import matplotlib.pyplot as plt
import networkx as nx 




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

"""graph_slider(
    trajectories=trajectories, 
    rates=rates_mat, 
    time_max=1000, 
    n_states=len(H1_E), 
    file_name='test.html')"""

#def graph_slider(trajectories, rates, time_max, n_states, file_name):
# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
#ax.margins(x=0)
fig.subplots_adjust(left=0.1, bottom=0.3)

# Make a horizontal slider to control the frequency.
axcolor = 'lightgoldenrodyellow'
axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])

freq_slider = Slider(
    axfreq,
    'Frame',
    0.0,
    1000.0,
    1.0,
    #initcolor='none',
    handle_style = {'facecolor':'red'}, 
    track_color="lightgrey", 
    facecolor="lightgrey",
    initcolor='red'
    #valstep=[i for i in range(0, 1000, 100)]
    #valfmt="%0.1f",
    #valstep=1,
    #slidermin=0,
    #slidermax=1000,
)

pos = graph_pos(rates)

def helper_plot(init_frame): 
    #reset ax 
    ax.clear()
    single_frame_slider(init_frame, trajectories, rates, pos, n_states, ax)
    
    #axfreq.add_artist(axfreq.xaxis)
    #sl_xticks = np.arange(0, 1000, 100)
    #sl_xticks.xaxis.set_visible(True)
    #axfreq.set_xticks(sl_xticks)    

helper_plot(init_frame=0)
freq_slider.on_changed(helper_plot)
#axfreq.set_xticks(np.arange(0, time_max, 100))

fig.canvas.draw_idle()
plt.show()