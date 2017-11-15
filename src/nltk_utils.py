from os import environ
from os.path import join, dirname
from typing import List, Union

from nltk import str2tuple, sent_tokenize, word_tokenize, pos_tag_sents, WordNetLemmatizer, SnowballStemmer, \
    RegexpTokenizer
from nltk.corpus import wordnet as wn, stopwords
from nltk.corpus.reader import NOUN
from nltk.parse.stanford import StanfordParser
from tqdm import tqdm


def is_wordnet_tag(tag):
    return tag in [wn.ADJ, wn.ADV, wn.VERB, wn.NOUN]


def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


def penn_to_wn(tag: str):
    if is_wordnet_tag(tag):
        return tag
    elif is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB

    # noun by default
    return wn.NOUN


class Lemmatizer(WordNetLemmatizer):
    def __init__(self):
        super().__init__()

    def lemmatize(self, word, pos=NOUN):
        return super().lemmatize(word, penn_to_wn(pos))


# TreebankWordTokenizer
class Tokenizer(RegexpTokenizer):
    _regexp = '\w+|\S|[!?.]+|\'[dst]'

    def span_tokenize(self, s):
        raise NotImplementedError()

    def __init__(self):
        super().__init__(self._regexp)

    def tokenize(self, text):
        return super().tokenize(text)

    def tokenize_sents(self, strings):
        return super().tokenize_sents(strings)


class Stemmer(SnowballStemmer):
    def __init__(self):
        super().__init__('english', ignore_stopwords=True)

    def stem(self, token):
        return super().stem(token)


class Parser:
    _delimiters = [',', ':', ';', '(', ')', '"', '\'', '.', '?', '!', '-']
    _paired_delimiters = [('(', ')'), ('\'', '\''), ('"', '"')]

    def parse_tokens(self, tokens):
        indexes = []
        last_index = 0

        for i, t in enumerate(tokens):
            if t in self._delimiters:
                if last_index < i:
                    indexes.append(list(range(last_index, i)))
                last_index = i + 1

        if last_index < len(tokens):
            indexes.append(list(range(last_index, len(tokens))))

        return indexes

    def parse_sents_tokens(self, sents_tokens):
        return [self.parse_tokens(s) for s in sents_tokens]


def split_tokens_tags(tagged_tokens: List[str]):
    tags, tokens = [], []

    for tt_str in tagged_tokens:
        tt_tuple = str2tuple(tt_str)

        tokens.append(tt_tuple[0])
        tags.append(tt_tuple[1])

    return tokens, tags


def content_fraction(tokens: List[str]):
    stop_words = stopwords.words('english')

    content = [t
               for t in tokens
               if t.lower() not in stop_words]

    return len(content) / len(tokens)


def get_lexical_units(text_or_tagged_sents: Union[List[str], str], tagged_sents=False) -> List[List[str]]:
    environ['JAVA_HOME'] = 'C:\Program Files\Java\jdk1.8.0_91'
    assert isinstance(text_or_tagged_sents, str) and not tagged_sents or isinstance(text_or_tagged_sents,
                                                                                    list) and tagged_sents
    _path = join(dirname(__file__), './standford_parser')

    parser = StanfordParser(join(_path, 'stanford-parser.jar'), join(_path, 'stanford-parser-3.4.1-models.jar'))

    # tag sentences if plain text is passed
    if not tagged_sents:
        sents = sent_tokenize(text_or_tagged_sents)
        sents_tokens = [word_tokenize(s) for s in sents]
        tagged_sentences = pos_tag_sents(sents_tokens)
    else:
        tagged_sentences = text_or_tagged_sents

    sents_lexical_units = []

    def lu_subtree_filter(t):
        h = t.height()
        l = len(t.leaves())

        if h <= 2:
            return True
        elif 2 <= l <= 3:
            return True
        else:
            return False

    for s in tqdm(tagged_sentences):
        tree_it = parser.tagged_parse(s)
        try:
            s_tree = next(tree_it)
        except StopIteration:
            continue

        lu_trees = s_tree.subtrees(lu_subtree_filter)

        sents_lexical_units.append([' '.join(t.leaves()) for t in lu_trees])

    return sents_lexical_units

if __name__ == '__main__':
    print(
    get_lexical_units('King has been gathered the data from Alice, and then went to the bed with a telescope.',
                      tagged_sents=False)
    )