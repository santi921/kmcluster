from kmcluster.core.intialize import population_ind_to_trajectories
from tqdm import tqdm

class kmc():
    def __init__(self, steps, pop_size, draw_crit, initialization, rates, time_stop=-1):
        self.steps = steps
        self.size = pop_size
        self.draw_crit = draw_crit
        self.rates = rates
        self.time_stop = time_stop
        self.initialization = initialization
        self.pop_init = initialization.get_init_populations()
        self.trajectories = population_ind_to_trajectories(self.pop_init, draw_crit)
    

    def step(self):

        for traj in self.trajectories:
            # get last state in traj 
            traj_last_ind = traj.last_state()
            # get row from numpy array
            # get row of transitions from state traj_last_ind
            rates_from_i = self.rates[traj_last_ind]
            traj.step(rates_from_i, self.draw_crit, time_stop=self.time_stop)
            


    def run(self, n_steps=10):
        for _ in tqdm(range(n_steps)):
            self.step()
        
        