import traceback
from collections import Counter
from typing import List, Any, Dict, Tuple, Sized, Union

from src.skipgrams import skipgrams

ngrams_separator = '_'


def ngram2str(ngram: tuple, sep: str = ngrams_separator):
    if isinstance(ngram, str):
        return ngram

    return sep.join([str(e) for e in ngram])


def str2ngram(text: str, sep: str = ngrams_separator) -> tuple:
    if isinstance(text, tuple):
        return text

    return tuple(text.split(sep))


class NGramsParser:
    _delimiters = [',',
                   ':',
                   ';',
                   '(',
                   ')',
                   '"',
                   '`',
                   '\'',
                   '.',
                   '?',
                   '!',
                   '-',
                   '—',
                   '\n',
                   '\t',
                   ' ']

    _paired_delimiters = [('(', ')'),
                          ('\'', '\''),
                          ('"', '"'),
                          ('`', '`'),
                          ('«', '»')]

    # ``, ''

    @classmethod
    def parse_tokens(cls, tokens):
        indexes = []
        last_index = 0

        for i, t in enumerate(tokens):
            if t in cls._delimiters:
                if last_index < i:
                    indexes.append(list(range(last_index, i)))

                last_index = i + 1

        if last_index < len(tokens):
            indexes.append(list(range(last_index, len(tokens))))

        return indexes

    @classmethod
    def parse_sents_tokens(cls, sents_tokens):
        return [cls.parse_tokens(s) for s in sents_tokens]


def n_grams(elements: Sized, n=1, indexes=None, pad_left=False, skip_grams=0, ngram_type='tuple') -> List[
    Union[Tuple, str]]:
    """
    Examples of input/output:
        [0, 1, 2], n=1 --> [(0,), (1,), (2,)]
        [0, 1, 2, 3, 4], n=2, pad_left=True --> [(None, 0), (0, 1), (1, 2), (2, 3), (3, 4)]

    :param elements: the source data to be converted into ngrams
    :param n: the degree of the ngrams
    :param pad_left: whether the ngrams should be left-padded
    :param indexes: list of tuples, where each tuple contains indexes of elements; n-grams will be created within tuples
    :param skip_grams:

    Example of indexes list:
        [(0, 1), (2, 3, 4), (5), ...]
        [(0, 1, ..., n-1)] (by default)

    :return: [(ngram_1), (ngram_2), ...]
    """
    if indexes is None:
        indexes = [list(range(len(elements)))]

    if not elements:
        return []

    ngrams = []

    try:
        for phrase_indexes in indexes:
            phrases = elements[phrase_indexes[0]:phrase_indexes[-1] + 1]

            phrase_ngrams = skipgrams(phrases, n=n, k=skip_grams, pad_left=pad_left)
            # phrase_ngrams = nltk.ngrams(phrases, n=n, pad_left=pad_left)

            ngrams.extend(phrase_ngrams)

    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return []

    if ngram_type == 'str':
        return [ngram2str(ngram)
                for ngram in ngrams]

    return ngrams


def n_grams_model(elements: List[Any], n=1, indexes=None, pad_left=True, skip_grams=0) -> Dict[Any, int]:
    assert n > 0

    ngrams = n_grams(elements, n, indexes, pad_left, skip_grams)

    counted = Counter(ngrams)
    total_count = sum(counted.values())

    for k in counted.keys():
        counted[k] /= total_count

    return counted
