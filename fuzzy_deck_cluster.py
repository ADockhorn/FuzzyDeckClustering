from typing import List, Dict
import math


class FuzzyDeck:
    def __init__(self, deck_data=None):
        """ Creates a Deck object based on a deck_data Dict containing information on
        'archetype_id',     Int,                                  default = -1
        'total_games',      Int,                                  default =  1
        'deck_id',          Int,                                  default = -1
        'card_multiset',    Dict(card, card_frequency) or str,    default = dict()

        :param deck_data: Dict containing 'archetype_id', 'total_games', 'deck_id', and/or 'card_multiset'
        """
        self.card_multiset = {}
        self.archetype = [-1]
        self.total_games = [0]
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
                        self.card_multiset[card_id] = [1.0]*int(count)
                else:
                    for [card_id, count] in deck_data["deck_list"]:
                        self.card_multiset[card_id] = [1.0]*int(count)

    def union(self, deck2: 'FuzzyDeck') -> 'FuzzyDeck':
        """ Creates a new FuzzyDeck object that represents the union of self and deck2.
        The archetype information, deck_id's as well as the total number of plays per deck are preserved.

        :param deck2: a FuzzyDeck object
        :return: union of self and deck2
        """
        d = FuzzyDeck()
        d.archetype = self.archetype.copy()
        d.archetype.extend(deck2.archetype)

        d.total_games = self.total_games.copy()
        d.total_games.extend(deck2.total_games)

        cards = set(list(self.card_multiset.keys()) + list(deck2.card_multiset.keys()))
        for card in cards:
            length = max(len(self.card_multiset.get(card, [])), len(deck2.card_multiset.get(card, [])))
            d.card_multiset[card] = []
            for i in range(length):
                occ1 = self.card_multiset.get(card, 0)[i] if i < len(self.card_multiset.get(card, [])) else 0
                occ2 = deck2.card_multiset.get(card, 0)[i] if i < len(deck2.card_multiset.get(card, [])) else 0
                d.card_multiset[card].append(max(occ1, occ2))
        return d

    def intersection(self, deck2: 'FuzzyDeck') -> 'FuzzyDeck':
        """ Creates a new FuzzyDeck object that represents the intersection of self and deck2.
        The archetype information, deck_id's as well as the total number of plays per deck are preserved.

        :param deck2: a FuzzyDeck object
        :return: intersection of self and deck2
        """
        d = FuzzyDeck()
        d.archetype = self.archetype.copy()
        d.archetype.extend(deck2.archetype)

        d.total_games = self.total_games.copy()
        d.total_games.extend(deck2.total_games)

        cards = set(list(self.card_multiset.keys()) + list(deck2.card_multiset.keys()))
        for card in cards:
            length = max(len(self.card_multiset.get(card, [])), len(deck2.card_multiset.get(card, [])))
            d.card_multiset[card] = []
            for i in range(length):
                occ1 = self.card_multiset.get(card, 0)[i] if i < len(self.card_multiset.get(card, [])) else 0
                occ2 = deck2.card_multiset.get(card, 0)[i] if i < len(deck2.card_multiset.get(card, [])) else 0
                if min(occ1, occ2) > 0:
                    d.card_multiset[card].append(min(occ1, occ2))
        return d

    def subtract(self, deck2: 'FuzzyDeck') -> 'FuzzyDeck':
        """ Creates a new FuzzyDeck object that represents the subtraction of deck2 from self.
        The resulting object has neither archetype nor total_games count.

        :param deck2: a FuzzyDeck object
        :return: subtraction of deck2 from self
        """
        d = FuzzyDeck()

        cards = set(list(self.card_multiset.keys()))
        for card in cards:
            length = len(self.card_multiset.get(card, []))
            for i in range(length):
                occ1 = self.card_multiset.get(card, 0)[i] if i < len(self.card_multiset.get(card, 0)) else 0
                occ2 = deck2.card_multiset.get(card, 0)[i] if i < len(deck2.card_multiset.get(card, 0)) else 0

                if (occ1 - occ2) > 0:
                    if card not in d.card_multiset:
                        d.card_multiset[card] = []
                    d.card_multiset[card].append(occ1 - occ2)

        return d

    def jaccard_distance(self, deck2: 'FuzzyDeck') -> float:
        """ Calculates the Jaccard distance of self and deck2.
        Returns a distance of 1 in case both decks share not even a single card.

        :param deck2: a FuzzyDeck object
        :return: Jaccard distance of self and deck2
        """
        cards = set(list(self.card_multiset.keys()) + list(deck2.card_multiset.keys()))

        j_nominator = 0
        j_denominator = 0

        for card in cards:
            length = max(len(self.card_multiset.get(card, [])), len(deck2.card_multiset.get(card, [])))
            for i in range(length):
                occ1 = self.card_multiset.get(card, 0)[i] if i < len(self.card_multiset.get(card, [])) else 0
                occ2 = deck2.card_multiset.get(card, 0)[i] if i < len(deck2.card_multiset.get(card, [])) else 0

                j_nominator += min(occ1, occ2)
                j_denominator += max(occ1, occ2)

        if j_denominator == 0:
            return 1
        return 1 - (j_nominator / j_denominator)

    def euclidean_distance(self, deck2: 'FuzzyDeck') -> float:
        """ Calculates the Euclidean distance of self and deck2.

        :param deck2: a FuzzyDeck object
        :return: Euclidean distance of self and deck2
        """
        cards = set(list(self.card_multiset.keys()) + list(deck2.card_multiset.keys()))

        euclidean_distance = 0
        for card in cards:
            length = max(len(self.card_multiset.get(card, [])), len(deck2.card_multiset.get(card, [])))
            for i in range(length):
                occ1 = self.card_multiset.get(card, 0)[i] if i < len(self.card_multiset.get(card, [])) else 0
                occ2 = deck2.card_multiset.get(card, 0)[i] if i < len(deck2.card_multiset.get(card, [])) else 0

                euclidean_distance += (occ1-occ2)**2

        return math.sqrt(euclidean_distance)

    def __str__(self):
        return str(self.card_multiset)

    def __repr__(self):
        return self.__str__()


class FuzzyDeckCluster:
    def __init__(self, decks: List[FuzzyDeck]):
        self.decks = decks

    def core(self) -> FuzzyDeck:
        """ The core represents the intersection of all decks contained in the FuzzyDeckCluster object.

        :return: core of the FuzzyDeckCluster
        """
        core = FuzzyDeck()
        for card in self.decks[0].card_multiset:
            core.card_multiset[card] = self.decks[0].card_multiset[card].copy()

        for deck in self.decks:
            core = core.intersection(deck)
        return core

    def variants(self) -> FuzzyDeck:
        """ Variants are all cards occurring in contained decks but not occurring the FuzzyDeckCluster's core.

        :return: core of the FuzzyDeckCluster
        """
        core = self.core()
        allcards = FuzzyDeck()
        for card in self.decks[0].card_multiset:
            allcards.card_multiset[card] = self.decks[0].card_multiset[card].copy()

        for deck in self.decks:
            allcards = allcards.union(deck)
        variants = allcards.subtract(core)
        return variants

    def centroid(self):
        """ Calculates the centroid of a FuzzyDeckCluster.
        Alternatively weighted_centroid could be used to take the total_games per deck into account.

        :return: centroid of the FuzzyDeckCluster
        """
        cards = set()
        for deck in self.decks:
            cards = cards.union(set(deck.card_multiset.keys()))

        c = FuzzyDeck()
        for card in cards:
            length = len(self.decks[0].card_multiset.get(card, []))
            for deck in self.decks:
                length = max(length, len(deck.card_multiset.get(card, [])))

            c.card_multiset[card] = []
            for i in range(length):
                membership_sum = 0
                for deck in self.decks:
                    membership_sum += deck.card_multiset.get(card, 0)[i] \
                        if i < len(deck.card_multiset.get(card, [])) else 0

                c.card_multiset[card].append(membership_sum / len(self.decks))

        return c

    def weighted_centroid(self):
        """ Calculates the weighted centroid of a FuzzyDeckCluster.
        Alternatively centroid could be used to not take the total_games per deck into account.

        :return: weighted centroid of the FuzzyDeckCluster
        """
        cards = set()
        for deck in self.decks:
            cards = cards.union(set(deck.card_multiset.keys()))

        c = FuzzyDeck()
        for card in cards:
            length = len(self.decks[0].card_multiset.get(card, []))
            for deck in self.decks:
                length = max(length, len(deck.card_multiset.get(card, [])))

            c.card_multiset[card] = []
            for i in range(length):
                membership_sum = 0
                total_games_sum = 0
                for deck in self.decks:
                    membership_sum += (deck.card_multiset.get(card, 0)[i]
                                       if i < len(deck.card_multiset.get(card, [])) else 0) * deck.total_games[0]
                    total_games_sum += deck.total_games[0]

                c.card_multiset[card].append(membership_sum / total_games_sum)

        return c


class FuzzyDeckClustering:

    def __init__(self, deck_clusters: List[FuzzyDeckCluster]):
        """ A DeckClustering is initialized using a list of DeckClusters

        :param deck_clusters:
        """
        self.deck_clusters = deck_clusters

    def get_centroids(self) -> List[FuzzyDeck]:
        """ Calculates the centroids of all included decks

        :return: centroids of all included decks in the FuzzyDeckClustering
        """
        centroids = []
        for cluster in self.deck_clusters:
            centroids.append(cluster.centroid())
        return centroids

    def get_prediction(self, previous_cards: FuzzyDeck) -> Dict:
        raise NotImplementedError()


if __name__ == "__main__":
    print("Definition of FuzzyDecks")
    D_1 = FuzzyDeck({"deck_list": [["a", 1], ["b", 1], ["c", 2], ["d", 1], ["e", 0], ["f", 2]], "total_games": 1})
    D_2 = FuzzyDeck({"deck_list": [["a", 1], ["b", 1], ["c", 2], ["d", 0], ["e", 2], ["f", 1]], "total_games": 2})
    print("D1: " + str(D_1))
    print("D2: " + str(D_2))
    print()

    print("FuzzyDeck operations")
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

    C = FuzzyDeckCluster([D_1, D_2])
    print("FuzzyDeckCluster operations")
    print("core: " + str(C.core()))
    print("variants: " + str(C.variants()))
    print("centroid: " + str(C.centroid()))
    print("weighted centroid: " + str(C.weighted_centroid()))
