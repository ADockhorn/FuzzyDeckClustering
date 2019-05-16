import matplotlib.pyplot as plt
from typing import List
import numpy as np
import matplotlib.lines as mlines
from scipy.spatial.distance import squareform, pdist
from sklearn.manifold import MDS
from matplotlib.cm import get_cmap

from deck_cluster import DeckCluster, DeckClustering
from fuzzy_deck_cluster import FuzzyDeckCluster, FuzzyDeckClustering


def plot_distance_matrix(sdist):
    """ Create a heatmap plot of distance matrix in square-form.

    :param sdist: distance matrix in square-form
    :return: axes object for further processing
    """
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 3.5), dpi=80)
    plt.imshow(sdist)
    plt.colorbar()
    return axes


def plot_cluster_result(labels_true, clustering_data: List, sdist):
    """ Plots the true labels and the clustering result side-by-side using a 2D projection created
    by MultiDimensionalScaling.

    :param labels_true: true labels
    :param clustering_data: list of clustering_data dicts  which should be plotted.
    :param sdist: distance matrix in square-form
    """

    plt.subplots(1, len(clustering_data)+1)

    # true labels
    from sklearn.manifold import MDS
    embedding = MDS(n_components=2, dissimilarity="precomputed")
    deck_transformed = embedding.fit_transform(sdist)

    # plot true labels
    plt.subplot(int("{}{}{}".format(1, len(clustering_data)+1, 1)))
    plt.scatter(deck_transformed[:, 0], deck_transformed[:, 1], c=labels_true, cmap=get_cmap("tab10"))
    plt.title("true labels")

    # plot clustering result
    for i, (alg_name, alg_data) in enumerate(clustering_data):
        plt.subplot(int("{}{}{}".format(1, len(clustering_data) + 1, i+2)))

        plt.scatter(deck_transformed[:, 0], deck_transformed[:, 1], c=alg_data["labels"], cmap=get_cmap("tab10"))
        plt.title(alg_name)
    plt.show()


def plot_eval_values(clustering_data, target, max_n, simple_plot: bool):
    """ Plots a comparison of homogeneity, completeness, or v-measure.
    This plot is very specific to present the results in the paper in a clean way. Please adapt for further use.

    Clustering_alg is a list of dicts of which each dict contains information on:
        "name": the name of the algorithm
        "distance": the distance measure to be used (either euclidean or jaccard)
        "alg": any sklearn cluster algorithm, e.g. AgglomerativeClustering
        "n": number of clusters
        "group": Int, in case this clustering belongs to a group of clustering

    :param clustering_data: list of dicts as described above
    :param target: "homogeneity", "completeness", or "v-measure"
    :param max_n: the maximal number of clusters
    :param simple_plot: simplify plot
    :return:
    """
    # load values
    values = np.zeros((4, max_n-2))
    for i, alg_data in enumerate(clustering_data):
        values[alg_data["group"], alg_data["n"]-2] = alg_data[target]

    if simple_plot:
        cmap = [get_cmap("tab20b").colors[12], get_cmap("tab20b").colors[14]]
        labels = ['single; $d_{jaccard/euclid}$', 'complete; $d_{jaccard/euclid}$']
        style = ["dotted", "solid"]
        markers = ['.', 'x']

    else:
        cmap = [get_cmap("tab20b").colors[12], get_cmap("tab20b").colors[13],
                get_cmap("tab20b").colors[14], get_cmap("tab20b").colors[15]]
        labels = ['single; $d_{jaccard}$', 'complete; $d_{jaccard}$', 'single; $d_{euclid}$', 'complete; $d_{euclid}$']
        style = ["dotted", "solid", "solid", "dotted"]
        markers = ['.', 'x', ".", '+']

    # plot target values
    steps = 4
    plt.figure(num=None, figsize=(5, 3.5), dpi=200, facecolor='w', edgecolor='k')
    for i in range(len(labels)):
        plt.plot(range(2, max_n, steps), values[i, 0:max_n:steps], c=cmap[i], marker=markers[i], linestyle=style[i])

    plt.xlabel("number of clusters")
    plt.title("{} per number of clusters".format(target))
    plt.tight_layout()

    # add legend
    handles = list()
    handles.append(mlines.Line2D([], [], marker=markers[1], color=cmap[1], linestyle=style[1], label=labels[1]))
    handles.append(mlines.Line2D([], [], marker=markers[0], color=cmap[0], linestyle=style[0], label=labels[0]))
    if not simple_plot:
        handles.append(mlines.Line2D([], [], marker=markers[2], color=cmap[2], linestyle=style[2], label=labels[2]))
        handles.append(mlines.Line2D([], [], marker=markers[3], color=cmap[3], linestyle=style[3], label=labels[3]))
    plt.legend(handles=handles)

    plt.gca().set_ylim([0.0, 1.1])

    plt.show()


def plot_sse_comparison(clustering_data, max_n):
    """ Plots a comparison of homogeneity, completeness, or v-measure.
    This plot is very specific to present the results in the paper in a clean way. Please adapt for further use.

    Clustering_alg is a list of dicts of which each dict contains information on:
        "name": the name of the algorithm
        "distance": the distance measure to be used (either euclidean or jaccard)
        "alg": any sklearn cluster algorithm, e.g. AgglomerativeClustering
        "n": number of clusters
        "group": Int, in case this clustering belongs to a group of clustering

    :param clustering_data: list of dicts as described above
    :param max_n:
    :return:
    """
    # load values
    values = np.zeros((2, max_n - 2))
    for i, alg_data in enumerate(clustering_data):
        values[0, alg_data["n"] - 2] = alg_data["sse_centroid"]
        values[1, alg_data["n"] - 2] = alg_data["sse_core"]

    # plot SSE and to core and centroid
    cmap = [get_cmap("tab20b").colors[12], get_cmap("tab20b").colors[13],
            get_cmap("tab20b").colors[14], get_cmap("tab20b").colors[15]]
    cmapoffset = 0
    markers = ['.', "x", 'x', '+']
    style = ["solid", "dotted", "solid", "dotted"]
    steps = 4
    plt.figure(num=None, figsize=(5, 3.5), dpi=200, facecolor='w', edgecolor='k')
    for i in range(2):
        plt.plot(range(2, max_n, steps), values[i, 0:max_n:steps], c=cmap[i + cmapoffset], marker=markers[i],
                 linestyle=style[i])

    plt.xlabel("number of clusters")
    plt.title("SSE to centroid and core per number of clusters".format())
    plt.tight_layout()

    # set legend
    handles = list()
    handles.append(mlines.Line2D([], [], marker=markers[0], color=cmap[cmapoffset + 0], linestyle=style[0],
                                 label='SSE to centroid'))
    handles.append(mlines.Line2D([], [], marker=markers[1], color=cmap[cmapoffset + 1], linestyle=style[1],
                                 label='SSE to core'))
    plt.legend(handles=handles)

    plt.show()
    pass


def plot_cluster_comparison(played_decks, clustering_data: List, labels_true, fuzzy: bool,
                            debug: bool = False, seed=10):
    """ Plots the true labels and the clustering result side-by-side using a 2D projection created
    by MultiDimensionalScaling.
    This plot is very specific to present the results in the paper in a clean way. Please adapt for further use.

    Clustering_alg is a list of dicts of which each dict contains information on:
        "name": the name of the algorithm
        "distance": the distance measure to be used (either euclidean or jaccard)
        "alg": any sklearn cluster algorithm, e.g. AgglomerativeClustering
        "n": number of clusters
        "group": Int, in case this clustering belongs to a group of clustering

    :param played_decks: the original data points
    :param clustering_data: list of dicts as described above
    :param labels_true: true labels of the data points
    :param fuzzy: use of Decks or FuzzyDecks?
    :param debug: show debug statements
    :param seed: random seed of the MultiDimensionalScaling
    """
    n_datapoints = len(played_decks)
    markers = ['o'] * len(played_decks)

    for alg_dict in clustering_data:
        decks = np.array(played_decks)
        clusters = []
        for label in set(alg_dict["labels"]):
            indices = np.where(alg_dict["labels"] == label)
            if fuzzy:
                clusters.append(FuzzyDeckCluster(decks[indices]))
            else:
                clusters.append(DeckCluster(decks[indices]))

        if fuzzy:
            clustering = FuzzyDeckClustering(clusters)
        else:
            clustering = DeckClustering(clusters)

        alg_dict["indices"] = list(range(n_datapoints))

        for cluster in clustering.deck_clusters:
            played_decks.append(cluster.centroid())
            if debug:
                print("Centroid: " + str(cluster.centroid()))
            markers.append('^')
            alg_dict["indices"].append(len(markers)-1)

        for cluster in clustering.deck_clusters:
            played_decks.append(cluster.core())
            if debug:
                print("Core: " + str(cluster.core()))
            markers.append('s')
            alg_dict["indices"].append(len(markers) - 1)

    # recalculate distance matrix
    deckdata = np.array(played_decks).reshape(len(played_decks), 1)
    dist = pdist(deckdata, lambda u, v: u[0].jaccard_distance(v[0]))
    sdist = squareform(dist)

    # calculate 2D projection
    embedding = MDS(n_components=2, n_init=4, dissimilarity="precomputed", random_state=seed)
    deck_transformed = embedding.fit_transform(sdist)

    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(8, 3.5), dpi=200, facecolor='w', edgecolor='k')

    # plot true labels with colors comparable to the true labeling. Note that only the colors where defined here
    colormap = {0: 0, 1: 1, 2: 11, 3: 6, 4: 9, 5: 8, 6: 5, 7: 13}
    for m, c, _x, _y in zip(['o']*len(deck_transformed), [get_cmap("tab20").colors[colormap[c]] for c in labels_true],
                            deck_transformed[:n_datapoints, 0],
                            deck_transformed[:n_datapoints, 1]):
        ax1.plot(_x, _y, 'o', markersize=9, markerfacecolor=c, marker=m,
                 markeredgewidth=0.5, markeredgecolor=(0, 0, 0, 1))

    ax1.set_ylim([-0.65, 0.65])
    ax1.set_title("Druid Deck Archetypes")

    # plot the cluster result
    markers = np.array(markers)
    for i, alg_dict in enumerate(clustering_data):
        for m, c, _x, _y in zip(markers[alg_dict["indices"]],
                                list(alg_dict["labels"]) + list(range(13)) + list(range(13)),
                                deck_transformed[alg_dict["indices"], 0],
                                deck_transformed[alg_dict["indices"], 1]):
            ax2.plot(_x, _y, 's', markersize=9, markerfacecolor=get_cmap("tab20").colors[c], marker=m,
                     markeredgewidth=0.7 if m == '^' else 0.5, markeredgecolor=(0, 0, 0, 1))
        ax2.set_title("Complete Linkage Clustering")
        ax2.set_ylim([-0.65, 0.65])
    plt.tight_layout()
    plt.show()
