import traceback
from collections import Counter
from typing import List, Any, Dict, Tuple

import nltk


def n_grams(elements: List[Any], n=1, indexes=None, pad_left=True) -> List[Tuple[Any]]:
    """
    Examples of input/output:
        [0, 1, 2], n=1 --> [(0,), (1,), (2,)]
        [0, 1, 2, 3, 4], n=2, pad_left=True --> [(None, 0), (0, 1), (1, 2), (2, 3), (3, 4)]

    :param elements: the source data to be converted into ngrams
    :param n: the degree of the ngrams
    :param pad_left: whether the ngrams should be left-padded
    :param indexes: list of tuples, where each tuple contains indexes of elements; n-grams will be created within tuples

    Example of indexes list:
        [(0, 1), (2, 3, 4), (5), ...]
        [(0, 1, ..., n-1)] (by default)

    :return: [(ngram_1), (ngram_2), ...]
    """
    if indexes is None:
        indexes = [list(range(len(elements)))]

    if not elements:
        return []

    grams = []

    try:
        for phrase_indexes in indexes:
            phrases = elements[phrase_indexes[0]:phrase_indexes[-1]+1]

            grams.extend(nltk.ngrams(phrases, n=n, pad_left=pad_left))

            # old version (without using nltk.ngrams)
            # if n > 1:
            #     end_of_range = len(phrase_indexes) - n + 1
            #
            #     for i in range(end_of_range):
            #         ind_begin, ind_end = phrase_indexes[i], phrase_indexes[i+n-1]
            #         grams.append(tuple(elements[ind_begin:ind_end+1]))
            # else:
            #     grams.extend((e,)
            #                  for e in elements[phrase_indexes[0]:phrase_indexes[-1]+1])

    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return []

    for g in grams:
        yield g


def n_grams_model(elements: List[Any], n=1, indexes=None, pad_left=True) -> Dict[Any, int]:
    assert n > 0

    grams = list(n_grams(elements, n, indexes, pad_left))

    counted = Counter(grams)
    total_count = sum(counted.values())

    for k in counted.keys():
        counted[k] /= total_count

    return counted


def get_ngrams_containing(ngrams: List[Tuple], gram: str) -> List[Tuple]:
    ngrams_containing = []

    for ngram in ngrams:
        if gram in ngram:
            ngrams_containing.append(ngram)

    return ngrams_containing
