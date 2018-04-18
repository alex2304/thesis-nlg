import os
import pickle
import traceback
from math import ceil
from os.path import join, dirname

from nltk import pos_tag_sents, word_tokenize, sent_tokenize, defaultdict, pprint
from nltk.parse.stanford import StanfordParser
from tqdm import tqdm

from src.grammar.utils import tags_seq_to_symbols
from src.io import load_corpora
from src.ngrams import ngram2str
from src.grammar.settings import target_labels, prods_file_path


def build_parser():
    parser_folder_path = join(dirname(__file__), './standford_parser')

    parser = StanfordParser(path_to_jar=join(parser_folder_path, 'stanford-parser.jar'),
                            path_to_models_jar=join(parser_folder_path, 'stanford-parser-3.4.1-models.jar'))

    return parser


parser = build_parser()


def empty_prods():
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


def load_prods(file_path):
    if os.path.exists(file_path):
        with open(file_path, mode='rb') as f:
            try:
                productions = pickle.load(f)

            except:
                productions = None

    else:
        productions = None

    if productions is None:
        productions = empty_prods()

    return productions


def save_prods(productions, file_path):
    with open(file_path, mode='wb') as f:
        pickle.dump(productions, f)


def extract_productions(text, max_deep=None):
    tagged_sents = get_tagged_sents(text)

    productions = load_prods(prods_file_path)

    # build sentences trees
    trees_iters = list(parser.tagged_parse_sents(tagged_sents))

    for tree_iter in trees_iters:
        try:
            tree = next(tree_iter)

            for production in tree.productions():
                if production.is_nonlexical():
                    lhs, rhs = production.lhs(), production.rhs()

                    label = lhs.symbol()
                    tags_seq = [n.symbol() for n in rhs]

                    symbols = tags_seq_to_symbols(tags_seq)

                    if label in target_labels and symbols:
                        symbols_str = ngram2str(symbols, sep=' ')

                        productions[label][symbols_str] += 1

        except StopIteration:
            pass

    save_prods(productions, prods_file_path)

    return productions


def display_prods(label=None):
    prods = load_prods(prods_file_path)

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


def extract_productions_run():
    corpora = load_corpora(test=False)

    sents_per_time = 15
    iters = int(ceil(len(corpora) / sents_per_time))

    for i in tqdm(range(iters)):
        corpus = corpora[sents_per_time * i: sents_per_time * (i + 1)]

        try:
            extract_productions(corpus)

        except:
            traceback.print_exc()

    display_prods()


if __name__ == '__main__':
    pass
    # demo()

    # exctract_productions_run()
