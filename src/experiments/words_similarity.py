from collections import defaultdict
from itertools import combinations

from nltk import pos_tag, flatten
from nltk.corpus import wordnet as wn
from nltk.corpus.reader import itemgetter, pprint

from src.grammars import parse_phrases
from src.ngrams import n_grams
from src.nltk_utils import Lemmatizer, Tokenizer, Parser
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

    tokenizer = Tokenizer()
    parser = Parser()

    for w in lemmas:
        phrases = voc.get_phrases_containing(w)

        if not phrases:
            hypernyms = synsets[w].hypernyms() or []

            for h in hypernyms:
                h_word = h.name().split('.')[0]
                phrases = voc.get_phrases_containing(w)
                if phrases:
                    break

            else:
                phrases = defaultdict(set)

        if not phrases:
            examples = [e + '.' for e in synsets[w].examples()]

            if examples:
                sents_tokens = tokenizer.tokenize_sents(examples)

                tokens = list(flatten(sents_tokens))

                ordered_tokens = list(flatten(sents_tokens))
                ordered_ttokens = pos_tag(ordered_tokens)

                tokens_indexes = parser.parse_tokens(ordered_tokens)

                for i in range(2, 5):
                    tt_ngrams = list(n_grams(ordered_ttokens, i, tokens_indexes, pad_left=False))

                    ngrams_containing_word = tt_ngrams# []

                    # for ngram in tt_ngrams:
                    #     ngram_tokens = [token for token, _ in ngram]
                    #     if w in ngram_tokens:
                    #         ngrams_containing_word.append(ngram)

                    types, types_phrases = parse_phrases(ngrams_containing_word, i)

                    for t, p in zip(types, types_phrases):
                        phrases[t].add(p)

        print(w, phrases)
