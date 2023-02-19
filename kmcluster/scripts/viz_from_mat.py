from kmcluster.core.viz import (
    plot_top_n_states,
    plot_states,
    graph_trajectories_static,
    graph_trajectories_dynamic,
    communities_static,
    compute_state_counts,
)


if __name__ == "__main__":
    kmc_boltz = kmc(
        self,
        draw_crit,
        energies,
        memory_friendly,
        check_converged,
        initialization=None,
        checkpoint=False,
        checkpoint_dir="./checkpoints/",
        final_save_prefix="saved_data",
        time_stop=-1,
        trajectories=None,
    )

    # kmc_boltz.run(n_steps=100)
    trajectories = kmc_boltz.trajectories

    plot_top_n_states(
        trajectories,
        n_show=5,
        total_states=len(H1_E),
        resolution=0.0001,
        max_time=0.001,
    )

    # plot_states(
    #    trajectories,
    #    states_to_plot=[0,1,2,3, 4, 5, 6, 7, 8,9],
    #    resolution = 0.0001,
    #    max_time = 0.001,
    #    title = "Test Run - Pt4H1",
    #    xlabel = "Time",
    #    ylabel = "State Counts",
    #    save = True,
    #    save_name = "../../reporting/plots/test_run_Pt4H1.png"
    # )

    # time_slice = 0
    # pos = graph_trajectories_static(
    #    trajectories=trajectories,
    #    energies = energies_mat,
    #    time=0.001,
    #    ret_pos = True
    # )
