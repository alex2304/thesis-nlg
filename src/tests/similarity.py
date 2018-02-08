from itertools import combinations

from nltk.corpus import wordnet as wn
from nltk.corpus.reader import itemgetter, pprint

from src.nltk_utils import Lemmatizer
# noinspection PyUnresolvedReferences
from src.vocabulary import get_vocabulary, Vocabulary, WordInfo

if __name__ == '__main__':
    words = ['clauses', 'possessive', 'purge', 'verbose', 'frontier']

    lemmatizer = Lemmatizer()

    lemmas = [lemmatizer.lemmatize(word) for word in words]

    print(lemmas)

    synsets = dict()

    for l in lemmas:
        word_synsets = wn.synsets(l)

        for s in word_synsets:
            if l == s.name().split('.')[0]:
                synsets[l] = s
                break

        else:
            if word_synsets:
                synsets[l] = word_synsets[0]
                print('Default:', l, word_synsets)

            else:
                print('Unknown:', l)

    print(synsets)

    words_combinations = list(combinations(synsets, 2))

    similarities = []

    # ic = wordnet_ic.ic('ic-brown.dat')
    # ic = wn.ic(genesis, False, 0.0)
    sim_cluser = set()

    for w1, w2 in words_combinations:
        s1, s2 = synsets[w1], synsets[w2]

        sim = s1.path_similarity(s2) or 0

        if sim:
            sim_cluser.add(w1)
            sim_cluser.add(w2)

            similarities.append(((w1, w2), sim))

    pprint(sorted(similarities, key=itemgetter(1), reverse=True))

    not_sim_cluster = set(lemmas).difference(sim_cluser)

    print(sim_cluser, not_sim_cluster)

    voc = get_vocabulary()

    for w in not_sim_cluster:
        pprint(voc.get_phrases_containing(w))