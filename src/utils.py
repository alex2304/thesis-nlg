import os
import pickle
from collections import Counter
from typing import List, Any, Dict, Tuple

from nltk import str2tuple

corpora_root_folder = os.path.join(os.path.dirname(__file__), '../corpora/')
data_root = os.path.join(os.path.dirname(__file__), '../data/')

voc_file_path = os.path.join(data_root, 'voc')


def get_grams(elements: List[Any], n=1):
    if n > 1:
        end_range = len(elements) - n + 1

        return [' '.join(elements[i:i + n])
                for i in range(end_range)]
    else:
        return list(elements)


def get_counted(grams: List[Any]):
    counted = Counter(grams)
    total_count = sum(counted.values())

    for k in counted.keys():
        counted[k] /= total_count

    return counted


def n_gram(elements: List[Any], n=1) -> Dict[Any, int]:
    assert n > 0

    grams = get_grams(elements, n)

    return get_counted(grams)


def get_ngrams_containing(ngrams: Dict, text: str) -> List[Tuple[str, Any]]:
    text = str(text).lower()
    matches = []

    for e, p in ngrams.items():
        words = e.lower().split(' ')
        if text in words:
            matches.append((e, p))

    return matches


def load_corpora(root_folder=corpora_root_folder, files_delimiter=' '):
    corpora = []

    for f_name in os.listdir(root_folder):
        with open(os.path.join(root_folder, f_name), mode='r', encoding='utf-8') as f:
            corpora.append(f.read())

    return files_delimiter.join(corpora)


def save_voc(voc, overwrite=False):
    if os.path.exists(voc_file_path) and not overwrite:
        return False

    with open(os.path.join(data_root, 'voc'), mode='wb') as f:
        pickle.dump(voc, f)

    return True


def load_voc():
    if not os.path.exists(voc_file_path):
        return None

    with open(voc_file_path, mode='rb') as f:
        return pickle.load(f)
