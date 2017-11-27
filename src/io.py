import json
import os
import pickle

corpora_root_folder = os.path.join(os.path.dirname(__file__), '../corpora/')
data_root = os.path.join(os.path.dirname(__file__), '../data/')

voc_file_path = os.path.join(data_root, 'voc')
phrases_file_path = os.path.join(data_root, 'phrases')


class MultiDimensionalArrayEncoder(json.JSONEncoder):
    def encode(self, obj):
        def hint_tuples(item):
            if isinstance(item, tuple):
                return {'__tuple__': True, 'items': [hint_tuples(i) for i in item]}
            if isinstance(item, list):
                return [hint_tuples(e) for e in item]
            if isinstance(item, dict):
                return {k: hint_tuples(v) for k, v in item.items()}
            if isinstance(item, set):
                return {'__set__': True, 'items': [hint_tuples(i) for i in item]}
            else:
                return item

        return super(MultiDimensionalArrayEncoder, self).encode(hint_tuples(obj))


def hinted_tuple_hook(obj):
    if '__tuple__' in obj:
        return tuple(obj['items'])
    if '__set__' in obj:
        return set(obj['items'])
    else:
        return obj


def load_corpora(root_folder=corpora_root_folder, files_delimiter=' '):
    corpora = []

    for f_name in os.listdir(root_folder):
        with open(os.path.join(root_folder, f_name), mode='r', encoding='utf-8') as f:
            corpora.append(f.read())

    return files_delimiter.join(corpora)


def save_voc(voc, overwrite=False):
    if os.path.exists(voc_file_path) and not overwrite:
        return False

    with open(voc_file_path, mode='wb') as f:
        pickle.dump(voc, f)

    return True


def save_phrases(phrases, overwrite=False):
    if os.path.exists(phrases_file_path) and not overwrite:
        return False

    with open(phrases_file_path, mode='wb') as f:
        pickle.dump(phrases, f)

    return True


def load_voc():
    if not os.path.exists(voc_file_path):
        return None

    with open(voc_file_path, mode='rb') as f:
        return pickle.load(f)


def load_phrases():
    if not os.path.exists(phrases_file_path):
        return None

    with open(phrases_file_path, mode='rb') as f:
        result = pickle.load(f)

        return result

if __name__ == '__main__':
    enc = MultiDimensionalArrayEncoder()
    jsonstring = enc.encode({"NP": {(1, (2, 3)), (4, 5, 6)}})

    print(jsonstring)

    print(json.loads(jsonstring, object_hook=hinted_tuple_hook))