import json
from scipy.spatial.distance import pdist
import numpy as np
from deck_cluster import Deck, DeckCluster, DeckClustering
from fuzzy_deck_cluster import FuzzyDeck, FuzzyDeckCluster, FuzzyDeckClustering
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.metrics.cluster import contingency_matrix
from typing import List, Union


def load_data_set(hero_class: str, fuzzy: bool, filename: str = "data/Decks.json", debug: bool = False) \
        -> Union[List[Deck], List[FuzzyDeck]]:
    """ Loads the data set provided in this repository and returns a list of Decks or FuzzyDecks. The deck list
    is sorted by archetype so the distance matrix is easier to visualize.

    :param hero_class: which hero class: ALL, HUNTER, ROGUE, PALADIN, MAGE, PRIEST, DRUID, WARLOCK, WARRIOR, SHAMAN
    :param fuzzy: load as Deck or FuzzyDeck
    :param filename: the json file
    :param debug: show debug statements
    :return: returns a list of Decks or FuzzyDecks
    """
    if debug:
        print("### loading dataset...")
    with open(filename) as f:
        data = json.load(f)

    hero_classes = list(data["series"]["metadata"].keys())
    if hero_class not in hero_classes and hero_class != "ALL":
        raise Exception("hero class <" + hero_class + "> not available. "
                        "Consider using one class out of: " + ", ".join(hero_classes))

    if debug:
        for cl in hero_classes:
            print("" + str(len(data["series"]["data"][cl])) + " played decks for hero class " + cl)

    played_decks = []
    if hero_class == "ALL":
        for hero_class in hero_classes:
            for i, deck_data in enumerate(data["series"]["data"][hero_class]):

                if fuzzy:
                    played_decks.append(FuzzyDeck(deck_data))
                else:
                    played_decks.append(Deck(deck_data))
    else:
        for i, deck_data in enumerate(data["series"]["data"][hero_class]):

            if fuzzy:
                played_decks.append(FuzzyDeck(deck_data))
            else:
                played_decks.append(Deck(deck_data))

    # sort by cluster label for easier visualization of distance matrix
    played_decks = sorted(played_decks, key=lambda x: x.archetype[0])
    return played_decks


def archetype_ranges(played_decks: List[Deck], debug=False):
    """

    :param played_decks:
    :param debug:
    :return:
    """
    last_id = [0, played_decks[0].archetype[0]]
    for i, deck in enumerate(played_decks):
        if deck.archetype[0] != last_id[1]:
            if debug:
                print("Archetype: " + str(last_id[1]) + " range = \t " + str(last_id[0]) + " - " + str(i - 1))
            last_id = [i, deck.archetype[0]]
        if debug:
            print("Archetype: " + str(last_id[1]) + " range = \t " + str(last_id[0]) + " - " + str(i))


def calculate_distance_matrix(played_decks: Union[List[FuzzyDeck], List[Deck]], measure: str):
    """ Calculates the distance matrix of a list of Deck or FuzzyDeck objects.
    Returns the vector-form distance vector.

    :param played_decks: list of Deck or FuzzyDeck objects
    :param measure: "jaccard" or "euclidean"
    :return: distance matrix
    """
    deck_data = np.array(played_decks).reshape(len(played_decks), 1)
    if measure == "jaccard":
        dist = pdist(deck_data, lambda u, v: u[0].jaccard_distance(v[0]))
    elif measure == "euclidean":
        dist = pdist(deck_data, lambda u, v: u[0].euclidean_distance(v[0]))
    else:
        raise ValueError("Unknown distance measure {}. ".format(measure) +
                         "Please choose one of the following distance measures ['euclidean','jaccard']")

    return dist


def eval_v_measure_homogeneity_completeness(clustering_alg: List, sdist_euclidean, sdist_jaccard,
                                            labels_true, debug: bool = False):
    """ Calculates v-measure, homogeneity, and completeness for each clustering algorithm stored in clustering_alg
    and adds it to each algorithms dictionary.

    Clustering_alg is a list of dicts of which each dict contains information on:
        "name": the name of the algorithm
        "distance": the distance measure to be used (either euclidean or jaccard)
        "alg": any sklearn cluster algorithm, e.g. AgglomerativeClustering
        "n": number of clusters
        "group": Int, in case this clustering belongs to a group of clustering

    :param clustering_alg: list of dicts as described above
    :param sdist_euclidean: Euclidean distance matrix in square-form
    :param sdist_jaccard: Jaccard distance matrix in square-form
    :param labels_true: the true labels per Deck
    :param debug: show debug statements?
    """
    for i, alg_dict in enumerate(clustering_alg):
        if "alg" in alg_dict:
            if alg_dict["distance"] == "euclidean":
                clustering = alg_dict["alg"].fit(sdist_euclidean)
            elif alg_dict["distance"] == "jaccard":
                clustering = alg_dict["alg"].fit(sdist_jaccard)
            else:
                raise ValueError("Unknown distance measure {}. ".format(alg_dict["distance"]) +
                                 "Please choose one of the following distance measures ['euclidean','jaccard']")
            labels_predicted = clustering.labels_
            alg_dict["labels"] = labels_predicted
        else:
            labels_predicted = alg_dict["labels"]

        alg_dict["homogeneity"], alg_dict["completeness"], alg_dict["v-measure"] = \
            homogeneity_completeness_v_measure(labels_true, labels_predicted)

        if debug:
            print("Alg: " + alg_dict["name"] + "; \t v-measure = " + str(alg_dict["v-measure"]))


def eval_cluster_contingency(clustering_alg: List, labels_true, sdist):
    """ Calculates a clustering's contingency matrix for each clustering algorithm stored in the list clustering_alg
    and adds it to the dict.

    Clustering_alg is a list of dicts of which each dict contains information on:
        "name": the name of the algorithm
        "distance": the distance measure to be used (either euclidean or jaccard)
        "alg": any sklearn cluster algorithm, e.g. AgglomerativeClustering
        "n": number of clusters
        "group": Int, in case this clustering belongs to a group of clustering

    :param clustering_alg: list of dicts as described above
    :param labels_true: the true labels per Deck
    :param sdist: the distance matrix in square-form to be used
    """
    for (alg_name, alg_dict) in clustering_alg:
        if "alg" in alg_dict:
            clustering = alg_dict["alg"].fit(sdist)
            labels_pred = clustering.labels_
            alg_dict["labels"] = labels_pred
        else:
            labels_pred = alg_dict["labels"]

        pred_label_dict, new_labels = normalize_labels(labels_pred)

        alg_dict["cm"] = contingency_matrix(labels_true, new_labels)


def eval_mean_distance(played_decks, clustering_data: List, fuzzy: bool, debug: bool = False):
    """ Calculates the mean distance and the sum of squared errors for each cluster and its related core and centroid.
    Always uses Jaccard distance.

    :param played_decks: the data points
    :param clustering_data: list of dicts as described above
    :param fuzzy: load as Deck or FuzzyDeck
    :param debug: show debug statements
    :return:
    """

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

        sum_of_squared_distances_centroid = 0
        sum_of_squared_distances_core = 0

        for cluster in clustering.deck_clusters:
            centroid = cluster.centroid()
            core = cluster.core()
            for deck in cluster.decks:
                sum_of_squared_distances_centroid += (deck.jaccard_distance(centroid))**2
                sum_of_squared_distances_core += (deck.jaccard_distance(core))**2
        alg_dict["sse_centroid"] = sum_of_squared_distances_centroid
        alg_dict["sse_core"] = sum_of_squared_distances_core

        if debug:
            print("Alg: " + alg_dict["name"] + "; \t sse = " + str(alg_dict["sse_centroid"]))
            print("Alg: " + alg_dict["name"] + "; \t sse = " + str(alg_dict["sse_core"]))


def normalize_labels(labels):
    """ Change the labels from arbitrary numbers to the range [0, len(set(labels))].
    Points that are in the same cluster will stay in the same cluster.
    Points from different clusters will remain in different clusters.

    :param labels: labels before
    :return: dict of {new_label: old_label}, list of new labels
    """
    new_labels = np.array([-1] * len(labels))
    labels = np.array(labels)
    label_dict = dict()
    for i, label in enumerate(set(labels)):
        new_labels[np.where(labels == label)] = i
        label_dict[i] = label
    return label_dict, new_labels
