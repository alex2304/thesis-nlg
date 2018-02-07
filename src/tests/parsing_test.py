import pickle
import traceback
from math import ceil

from os.path import join, dirname

import os
from typing import Union

from nltk import pos_tag_sents, word_tokenize, sent_tokenize, defaultdict, Counter, pprint
from nltk.parse.stanford import StanfordParser
from tqdm import tqdm

from src.io import load_corpora
from src.ngrams import ngram2str, n_grams


def build_parser():
    parser_folder_path = join(dirname(__file__), './standford_parser')

    parser = StanfordParser(path_to_jar=join(parser_folder_path, 'stanford-parser.jar'),
                            path_to_models_jar=join(parser_folder_path, 'stanford-parser-3.4.1-models.jar'))

    return parser


parser = build_parser()

target_labels = ('NP', 'VP', 'PP', 'ADJP', 'ADVP')

terminals = ['CC',
             'CD',
             'DT',
             'EX',
             'FW',
             'IN',
             'JJ',
             'JJR',
             'JJS',
             'LS',
             'MD',
             'NN',
             'NNS',
             'NNP',
             'NNPS',
             'PDT',
             'POS',
             'PRP',
             'PRP$',
             'RB',
             'RBR',
             'RBS',
             'RP',
             'SYM',
             'TO',
             'UH',
             'VB',
             'VBD',
             'VBG',
             'VBN',
             'VBP',
             'VBZ',
             'WDT',
             'WP',
             'WP$',
             'WRB']

accepted_tags = terminals + list(target_labels)

replacements = {
    'PRP$': 'PRPS',
    'WP$': 'WPS'
}

prods_file_path = os.path.join(os.path.dirname(__file__), 'productions')


def empty_prods():
    # {
    #     'NP': {
    #         'NN NN': 1,
    #         'NN': 2
    #     },
    #     'VP': {
    #         'VBS': 10,
    #         'VP VBS': 5
    #     },
    #     # ...
    # }
    return {
        label: defaultdict(int)
        for label in target_labels
    }


def get_tagged_sents(text):
    if isinstance(text, str):
        sents = sent_tokenize(text)

        sents_tokens = [word_tokenize(s)
                        for s in sents]

    else:
        sents_tokens = text

    tagged_sentences = pos_tag_sents(sents_tokens)

    return tagged_sentences


def validate_tags(sequence: Union[tuple, list]) -> tuple:
    validated = []

    for tag in sequence:
        if tag not in accepted_tags:
            return tuple()

        validated.append(replacements.get(tag) or tag)

    return tuple(validated)


def load_prods():
    if os.path.exists(prods_file_path):
        with open(prods_file_path, mode='rb') as f:
            try:
                productions = pickle.load(f)

            except:
                productions = None

    else:
        productions = None

    if productions is None:
        productions = empty_prods()

    return productions


def save_prods(productions):
    with open(prods_file_path, mode='wb') as f:
        pickle.dump(productions, f)


def extract_productions(text):
    tagged_sents = get_tagged_sents(text)

    productions = load_prods()

    # TODO: find n-grams which were already parsed?
    # tags_ngrams = []
    #
    # for sent in tagged_sents:
    #     sent_tags = [tag for _, tag in sent]
    #
    #     for n in range(1, 6 + 1):
    #         tags_ngrams.extend(n_grams(sent_tags, n))

    # build sentences trees
    trees_iters = list(parser.tagged_parse_sents(tagged_sents))

    for tree_iter in trees_iters:
        try:
            tree = next(tree_iter)

            for production in tree.productions():
                if production.is_nonlexical():
                    lhs, rhs = production.lhs(), production.rhs()

                    label = lhs.symbol()
                    tags = validate_tags([n.symbol() for n in rhs])

                    if label in target_labels and tags:
                        tags_str = ngram2str(tags, sep=' ')

                        productions[label][tags_str] += 1

        except StopIteration:
            pass

    save_prods(productions)

    return productions


def display_prods(label=None):
    prods = load_prods()

    if label:
        pprint(prods.get(label, {}))

    for p in prods:
        print(p, len(prods[p]))

    print()


def demo():
    while True:
        input_text = input('> ')

        prods = extract_productions(input_text)

        print(prods)


if __name__ == '__main__':
    # demo()

    corpora = load_corpora(test=False)

    sents_per_time = 20
    iters = int(ceil(len(corpora) / sents_per_time))

    for i in tqdm(range(iters)):
        corpus = corpora[sents_per_time * i: sents_per_time * (i + 1)]

        try:
            extract_productions(corpus)
        except:
            traceback.print_exc()

    display_prods()
