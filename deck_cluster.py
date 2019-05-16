from typing import List, Dict
import numpy as np
import math


class Deck:

    def __init__(self, deck_data: Dict = None):
        """ Creates a Deck object based on a deck_data Dict containing information on
        'archetype_id',     Int,                                  default = -1
        'total_games',      Int,                                  default =  1
        'deck_id',          Int,                                  default = -1
        'card_multiset',    Dict(card, card_frequency) or str,    default = dict()

        :param deck_data: Dict containing 'archetype_id', 'total_games', 'deck_id', and/or 'card_multiset'
        """
        self.card_multiset = {}
        self.archetype = [-1]
        self.total_games = [1]
        self.deck_id = [-1]

        if deck_data is not None:
            if "archetype_id" in deck_data:
                self.archetype = [deck_data["archetype_id"]]

            if "total_games" in deck_data:
                self.total_games = [deck_data["total_games"]]

            if "deck_id" in deck_data:
                self.deck_id = deck_data["deck_id"]

            if "deck_list" in deck_data:
                if type(deck_data["deck_list"]) is str:
                    for deckentry in deck_data["deck_list"][2:-2].split("],["):
                        [card_id, count] = deckentry.split(",")
                        self.card_multiset[card_id] = float(count)
                else:
                    for [card_id, count] in deck_data["deck_list"]:
                        self.card_multiset[card_id] = float(count)

    def union(self, deck2: 'Deck') -> 'Deck':
        """ Creates a new Deck object that represents the union of self and deck2.
        The archetype information, deck_id's as well as the total number of plays per deck are preserved.

        :param deck2: a Deck object
        :return: union of self and deck2
        """
        d = Deck()
        d.archetype = self.archetype.copy()
        d.archetype.extend(deck2.archetype)

        d.total_games = self.total_games.copy()
        d.total_games.extend(deck2.total_games)

        d.deck_id = self.deck_id.copy()
        d.deck_id.extend(deck2.deck_id)

        cards = set(list(self.card_multiset.keys()) + list(deck2.card_multiset.keys()))
        for card in cards:
            occ1 = self.card_multiset.get(card, 0)
            occ2 = deck2.card_multiset.get(card, 0)
            d.card_multiset[card] = max(occ1, occ2)
        return d

    def intersection(self, deck2: 'Deck') -> 'Deck':
        """ Creates a new Deck object that represents the intersection of self and deck2.
        The archetype information, deck_id's as well as the total number of plays per deck are preserved.

        :param deck2: a Deck object
        :return: intersection of self and deck2
        """
        d = Deck()
        d.archetype = self.archetype.copy()
        d.archetype.extend(deck2.archetype)

        d.total_games = self.total_games.copy()
        d.total_games.extend(deck2.total_games)

        d.deck_id = self.deck_id.copy()
        d.deck_id.extend(deck2.deck_id)

        cards = set(list(self.card_multiset.keys()) + list(deck2.card_multiset.keys()))
        for card in cards:
            occ1 = self.card_multiset.get(card, 0)
            occ2 = deck2.card_multiset.get(card, 0)
            if min(occ1, occ2) > 0:
                d.card_multiset[card] = min(occ1, occ2)
        return d

    def subtract(self, deck2: 'Deck') -> 'Deck':
        """ Creates a new Deck object that represents the subtraction of deck2 from self.
        The resulting object has neither archetype nor total_games count.

        :param deck2: a Deck object
        :return: subtraction of deck2 from self
        """
        d = Deck()

        cards = set(list(self.card_multiset.keys()))
        for card in cards:
            occ1 = self.card_multiset.get(card, 0)
            occ2 = deck2.card_multiset.get(card, 0)

            if (occ1 - occ2) > 0:
                d.card_multiset[card] = occ1 - occ2

        return d

    def jaccard_distance(self, deck2: 'Deck') -> float:
        """ Calculates the Jaccard distance of self and deck2.
        Returns a distance of 1 in case both decks share not even a single card.

        :param deck2: a Deck object
        :return: Jaccard distance of self and deck2
        """
        cards = set(list(self.card_multiset.keys()) + list(deck2.card_multiset.keys()))

        j_nominator = 0
        j_denominator = 0

        for card in cards:
            occ1 = self.card_multiset.get(card, 0)
            occ2 = deck2.card_multiset.get(card, 0)
            j_nominator += min(occ1, occ2)
            j_denominator += max(occ1, occ2)

        if j_denominator == 0:
            return 1
        return 1 - (j_nominator / j_denominator)

    def euclidean_distance(self, deck2: 'Deck') -> float:
        """ Calculates the Euclidean distance of self and deck2.

        :param deck2: a Deck object
        :return: Euclidean distance of self and deck2
        """
        cards = set(list(self.card_multiset.keys()) + list(deck2.card_multiset.keys()))

        euclidean_distance = 0

        for card in cards:
            occ1 = self.card_multiset.get(card, 0)
            occ2 = deck2.card_multiset.get(card, 0)
            euclidean_distance += (occ1-occ2)**2
        return math.sqrt(euclidean_distance)

    def __str__(self):
        return str(self.card_multiset)

    def __repr__(self):
        return self.__str__()


class DeckCluster:
    def __init__(self, decks: List[Deck]):
        """ A DeckCluster represents a List of Decks

        :param decks: a list of Deck objects
        """
        self.decks = decks

    def core(self) -> Deck:
        """ The core represents the intersection of all decks contained in the DeckCluster object.

        :return: core of the DeckCluster
        """
        core = Deck()
        for card in self.decks[0].card_multiset:
            core.card_multiset[card] = self.decks[0].card_multiset[card]

        for deck in self.decks:
            core = core.intersection(deck)
        return core

    def variants(self) -> Deck:
        """ Variants are all cards occurring in contained decks but not occurring the DeckCluster's core.

        :return: core of the DeckCluster
        """
        contained_cards = Deck()
        for card in self.decks[0].card_multiset:
            contained_cards.card_multiset[card] = self.decks[0].card_multiset[card]

        for deck in self.decks:
            contained_cards = contained_cards.union(deck)
        variants = contained_cards.subtract(self.core())
        return variants

    def centroid(self) -> Deck:
        """ Calculates the centroid of a DeckCluster.
        Alternatively weighted_centroid could be used to take the total_games per deck into account.

        :return: centroid of the DeckCluster
        """
        cards = set()
        for deck in self.decks:
            cards = cards.union(set(deck.card_multiset.keys()))

        c = Deck()
        for card in cards:
            card_sum = 0
            for deck in self.decks:
                card_sum += deck.card_multiset.get(card, 0)
            c.card_multiset[card] = card_sum / len(self.decks)
        return c

    def weighted_centroid(self) -> Deck:
        """ Calculates the weighted centroid of a DeckCluster.
        Alternatively centroid could be used to not take the total_games per deck into account.

        :return: weighted centroid of the DeckCluster
        """
        cards = set()
        for deck in self.decks:
            cards = cards.union(set(deck.card_multiset.keys()))

        c = Deck()
        for card in cards:
            card_sum = 0
            totalgames_sum = 0
            for deck in self.decks:
                card_sum += deck.card_multiset.get(card, 0)*deck.total_games[0]
                totalgames_sum += sum(deck.total_games)
            c.card_multiset[card] = card_sum / totalgames_sum

        return c


class DeckClustering:

    def __init__(self, deck_clusters: List[DeckCluster]):
        """ A DeckClustering is initialized using a list of DeckClusters

        :param deck_clusters:
        """
        self.deck_clusters = deck_clusters

    def get_centroids(self) -> List[Deck]:
        """ Calculates the centroids of all included decks

        :return: centroids of all included decks in the DeckClustering
        """
        centroids = []
        for cluster in self.deck_clusters:
            centroids.append(cluster.centroid())
        return centroids

    def get_prediction(self, previous_cards: Deck) -> List:
        """ [Work in Progress]
        return the probability of observing upcoming cards given a list of previous cards encoded as Deck

        :param previous_cards: Deck of previously seen cards
        :return:
        """
        centroids = self.get_centroids()
        dist = [centroid.jaccard_distance(previous_cards) for centroid in centroids]
        closest_centroid = centroids[np.argmin(dist)[0]]

        predicted_multiset = closest_centroid.subtract(previous_cards)
        predicted_count_sum = sum([y for (x, y) in predicted_multiset.card_multiset.items()])
        predicted_prob = [(x, y/predicted_count_sum) for (x, y) in predicted_multiset.card_multiset.items()]

        return predicted_prob


if __name__ == "__main__":
    print("Definition of Decks")
    D_1 = Deck({"deck_list": [["a", 1], ["b", 1], ["c", 2], ["d", 1], ["e", 0], ["f", 2]], "total_games": 1})
    D_2 = Deck({"deck_list": [["a", 1], ["b", 1], ["c", 2], ["d", 0], ["e", 2], ["f", 1]], "total_games": 2})
    print("D1: " + str(D_1))
    print("D2: " + str(D_2))
    print()

    print("Deck operations")
    D_intersection = D_1.intersection(D_2)
    print("intersection: " + str(D_intersection))

    D_union = D_1.union(D_2)
    print("union: " + str(D_union))

    D_subtract = D_1.subtract(D_2)
    print("subtract: " + str(D_union))

    jaccard = D_1.jaccard_distance(D_2)
    print("Jaccard distance: " + str(jaccard))

    euclidean = D_1.euclidean_distance(D_2)
    print("Euclidean distance: " + str(euclidean))
    print()

    C = DeckCluster([D_1, D_2])
    print("DeckCluster operations")
    print("core: " + str(C.core()))
    print("variants: " + str(C.variants()))
    print("centroid: " + str(C.centroid()))
    print("weighted centroid: " + str(C.weighted_centroid()))
