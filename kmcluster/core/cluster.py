import networkx as nx
from sklearn.cluster import AffinityPropagation
from kmcluster.core.data import nice_print_clusters, energy_to_rates
import matplotlib.pyplot as plt
import numpy as np


def affinity_at_temp(energies_mat, energy_list, temp=100, pref=None, verbose=True):
    rate_mat = energy_to_rates(energies_mat, temp, scale=1)
    ap = AffinityPropagation(
        affinity="precomputed",
        max_iter=10000,
        preference=pref,
        damping=0.5,
    ).fit(rate_mat)
    if verbose:
        nice_print_clusters(ap, energy_list)
    return ap


def plot_affinity_at_temp(
    G, energies_mat, energy_list, temperature, weightage=None, verbose=False
):
    ap = affinity_at_temp(
        energies_mat,
        energy_list=energy_list,
        temp=temperature,
        pref=weightage,
        verbose=False,
    )

    dict_labels = {i: [] for i in np.unique(ap.labels_)}
    for i, label in enumerate(ap.labels_):
        dict_labels[label].append(i + 1)

    pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
    # Draw the graph, but don't color the nodes
    nx.draw(
        G,
        pos,
        edge_color="k",
        with_labels=True,
        font_weight="light",
        node_size=280,
        width=0.9,
    )

    # Now, color the nodes
    for i, label in enumerate(ap.labels_):
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=dict_labels[label],
            node_color=plt.cm.tab20(i / len(np.unique(ap.labels_))),
            node_size=280,
        )


def plot_coms_cdlib(G, com_list):
    dict_labels = {i: list_com for i, list_com in enumerate(com_list)}

    pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
    # Draw the graph, but don't color the nodes
    nx.draw(
        G,
        pos,
        edge_color="k",
        with_labels=True,
        font_weight="light",
        node_size=280,
        width=0.9,
    )

    # Now, color the nodes
    for i, label in enumerate(dict_labels):
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=dict_labels[label],
            node_color=plt.cm.tab20(i / len(dict_labels)),
            node_size=280,
        )
