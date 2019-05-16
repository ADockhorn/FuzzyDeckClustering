from sklearn.cluster import AgglomerativeClustering
from analysis_tools import load_data_set, normalize_labels, calculate_distance_matrix, \
    eval_v_measure_homogeneity_completeness, eval_mean_distance
from plot_utils import plot_distance_matrix, plot_sse_comparison, plot_eval_values, plot_cluster_comparison
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # set plot parameters
    HERO_CLASS = "DRUID"
    DEBUG = False
    FUZZY = True
    SIMPLE_PLOTS = True

    # load data set
    playedDecks = load_data_set(HERO_CLASS, FUZZY, "data/Decks.json")
    id_to_index = {deck_id: i for i, deck_id in enumerate([p.deck_id for p in playedDecks])}
    MAX_N = len(playedDecks)
    archetype_label_dict, labels_true = normalize_labels(labels=[d.archetype[0] for d in playedDecks])

    # calculate distance matrices
    dist_jaccard = calculate_distance_matrix(playedDecks, measure="jaccard")
    sdist_jaccard = squareform(dist_jaccard)
    dist_euclidean = calculate_distance_matrix(playedDecks, measure="euclidean")
    sdist_euclidean = squareform(dist_euclidean)

    # generate Figure 1a
    ax = plot_distance_matrix(sdist_jaccard)
    ax.set_title("pairwise Jaccard distance")
    ax.set_xlabel(str(HERO_CLASS).lower() + " decks")
    plt.show()

    # create a list of cluster algorithms to evaluate
    clustering_data = []
    for i in range(2, MAX_N):
        clustering_data.append({"name": "Single({})".format(i), "distance": "jaccard", "n": i, "group": 0,
                                "alg": AgglomerativeClustering(n_clusters=i, affinity='precomputed', linkage="single")})
    for i in range(2, MAX_N):
        clustering_data.append({"name": "Complete({})".format(i), "distance": "jaccard", "n": i, "group": 1,
                                "alg": AgglomerativeClustering(n_clusters=i, affinity='precomputed',
                                                               linkage="complete")})

    # Clustering using Jaccard and Euclidean distance return the same clustering results,
    # but Euclidean distance is omitted for simplicity
    if not SIMPLE_PLOTS:
        for i in range(2, MAX_N):
            clustering_data.append({"name": "Single({})".format(i), "distance": "euclidean", "n": i, "group": 2,
                                    "alg": AgglomerativeClustering(n_clusters=i, affinity='precomputed',
                                                                   linkage="single")})
        for i in range(2, MAX_N):
            clustering_data.append({"name": "Complete({})".format(i), "distance": "euclidean", "n": i, "group": 3,
                                    "alg": AgglomerativeClustering(n_clusters=i, affinity='precomputed',
                                                                   linkage="complete")})

    # calculate cluster validation measures
    eval_v_measure_homogeneity_completeness(clustering_data, sdist_euclidean, sdist_jaccard, labels_true)
    eval_mean_distance(playedDecks, clustering_data, FUZZY)

    # generate Figure 1b
    a = [(i, d["v-measure"]) for i, d in enumerate(clustering_data)]
    c = clustering_data[sorted(a, key=lambda u: u[1], reverse=True)[0][0]]
    plot_cluster_comparison(playedDecks, [c], labels_true, FUZZY, seed=11)

    # generate Figure 1c (this plot is very specific to assure comparability of the clustering and the true labels
    plot_sse_comparison(clustering_data[-(MAX_N-2):], MAX_N)

    # generate Figure 2a
    plot_eval_values(clustering_data, "homogeneity", MAX_N, simple_plot=SIMPLE_PLOTS)

    # generate Figure 2b
    plot_eval_values(clustering_data, "completeness", MAX_N, simple_plot=SIMPLE_PLOTS)

    # generate Figure 2c
    plot_eval_values(clustering_data, "v-measure", MAX_N, simple_plot=SIMPLE_PLOTS)
