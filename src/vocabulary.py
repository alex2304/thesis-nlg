from math import log
from typing import Tuple, List

from nltk import sent_tokenize, flatten, defaultdict, pos_tag

from src.grammars import parse_phrases
from src.io import load_corpora, load_voc, save_voc, load_phrases, save_phrases
from src.nltk_utils import Lemmatizer, Tokenizer, Stemmer, Parser
from src.ngrams import n_grams, n_grams_model, get_ngrams_containing


class WordInfo:
    def __init__(self, token, tag, stem, lemma):
        self.stem = stem
        self.lemma = lemma
        self.tag = tag
        self.token = token

    def ttoken(self):
        return self.token, self.tag

    def __eq__(self, other):
        return other.token == self.token

    def __hash__(self):
        return hash(self.token)


class Vocabulary:
    _default_prob = 1e-07

    def __init__(self, corpora: str, n=1, test=False, t_phrases=None):
        self.n = n
        self.test = test

        # create utils and save as members of the class
        tokenizer = Tokenizer()
        stemmer = Stemmer()
        lemmatizer = Lemmatizer()
        parser = Parser()

        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.lemmatizer = lemmatizer
        self.parser = parser

        # segment sentences
        if isinstance(corpora, str):
            sents = sent_tokenize(corpora)

            if test:
                sents = sents[:10]

            sents_tokens = tokenizer.tokenize_sents(sents)

        else:
            sents_tokens = corpora

            if test:
                sents_tokens = sents_tokens[:10]

        # tokenizing and POS tagging
        tokens = list(flatten(sents_tokens))
        ttokens = pos_tag(tokens)

        # stemming
        stems = [stemmer.stem(t)
                 for t in tokens]

        # lemmatizing
        lemmas = [lemmatizer.lemmatize(token, pos=tag)
                  for token, tag in ttokens]

        # create list of words with their info
        assert len(ttokens) == len(stems) == len(lemmas)
        words = [WordInfo(token=token, tag=tag, stem=s, lemma=l)
                 for (token, tag), s, l in zip(ttokens, stems, lemmas)]

        # create indexes
        self._t_words = {}
        self._tag_words = defaultdict(set)
        self._s_words = defaultdict(set)
        self._l_words = defaultdict(set)

        for w in words:
            self._t_words[w.token] = w
            self._tag_words[w.tag].add(w)
            self._s_words[w.stem].add(w)
            self._l_words[w.lemma].add(w)

        # create n-gram models
        ordered_tokens = list(flatten(sents_tokens))
        ordered_ttokens = pos_tag(ordered_tokens)

        tokens_indexes = parser.parse_tokens(ordered_tokens)

        self.ngram_tokens_model = n_grams_model(ordered_tokens, n=n, indexes=tokens_indexes)
        self.ngram_tags_model = n_grams_model([tag for _, tag in ordered_ttokens], n=n, indexes=tokens_indexes)

        # parse phrasal units from tokens
        self._s_phrases = defaultdict(set)
        self._t_phrases = defaultdict(set)

        if t_phrases is not None:
            self._t_phrases = t_phrases

            for t, phrases in t_phrases.items():
                for p in phrases:
                    self._s_phrases[len(p)].add(p)

            print('phrases loaded from file')
            return

        for i in range(2, 5):
            tt_ngrams = list(n_grams(ordered_ttokens, i, tokens_indexes, pad_left=False))

            types, phrases = parse_phrases(tt_ngrams, i)

            for t, p in zip(types, phrases):
                self._s_phrases[len(p)].add(p)
                self._t_phrases[t].add(p)

    # def spec_n_gram(self, ngram_tokens: Dict, ngram_lemmas: Dict) -> Dict:
    #     combined_ngram = dict()
    #     missed_lemmas = set()
    #
    #     for tagged_token_gram, tt_prob in ngram_tokens.items():
    #         gram_tokens = [nltk.str2tuple(tt)
    #                        for tt in tagged_token_gram.split(' ')]
    #
    #         gram_lemmas = [self.lemmatizer.lemmatize(word=token, pos=tag)
    #                        for token, tag in gram_tokens]
    #
    #         lemma_gram = ' '.join(gram_lemmas)
    #
    #         lemma_gram_prob = ngram_lemmas.get(lemma_gram)
    #         if lemma_gram_prob is None:
    #             missed_lemmas.add(lemma_gram)
    #             lemma_gram_prob = self._default_prob
    #
    #         combined_ngram[lemma_gram] = {
    #             'lemma': (lemma_gram, lemma_gram_prob),
    #             'ttoken': (tagged_token_gram, tt_prob)
    #         }
    #
    #     if missed_lemmas:
    #         print('Missed lemmas: %s' % str(missed_lemmas))
    #
    #     return combined_ngram

    # def _create_voc(self, stems, lemmas, ttokens):
    #     tokens, tags = split_tokens_tags(ttokens)
    #
    #     tokens_indexes = Parser().parse_tokens(tokens)
    #
    #     # create tokens and tags n-gram
    #     self._tokens = n_gram(tokens, n=self.n, indexes=tokens_indexes)
    #     self._tags = n_gram(tags, n=self.n, indexes=tokens_indexes)
    #     self._tagged_tokens = n_gram(ttokens, n=self.n, indexes=tokens_indexes)
    #
    #     # create lemmas n-gram
    #     self._lemmas = n_gram(lemmas, n=self.n, indexes=tokens_indexes)
    #
    #     # create lemmas and tokens
    #     self._lemmas_tokens = self.spec_n_gram(self._tagged_tokens, self._lemmas)
    #
    #     # create stems unigram
    #     self._stems = n_gram(stems, n=1, indexes=tokens_indexes)

    ######################################################################
    # Getting elements containing text
    ######################################################################

    def lemma_words(self, l) -> List[WordInfo]:
        return self._l_words.get(l) or set()

    def get_phrases_containing(self, lemma: str, size=None):
        # if size:
        #     indexed_phrases =
        #
        # if tag.startswith('N'):
        #     phrase_type = 'NP'
        # elif tag.startswith('V'):
        #     phrase_type = 'VP'
        # elif tag in ['IN', 'TO']:
        #     phrase_type = 'PP'
        # elif tag in ['JJ', 'RB']:
        #     phrase_type = 'ADJP'
        # else:
        #     raise NotImplementedError(tag)

        # TODO: options which phrases to return

        ttokens = set(w.ttoken() for w in self.lemma_words(lemma))

        result = defaultdict(list)
        if ttokens:
            for phr_type, phrases in self._t_phrases.items():
                result[phr_type].extend([phrase
                                         for phrase in list(phrases)
                                         if ttokens.intersection(phrase)])

        return result

    ######################################################################
    # Probabilities
    ######################################################################

    def sent_model(self, log_prob=True, *ttokens) -> Tuple[float, float]:
        """
        Example of input:
            ttokens = (('alice', 'NN'), ('left', 'VBD'), ('the', 'DT'), ('court', 'NN'))
        :return: (n-gram language model, n-gram morpheme model)
        """
        tokens, tags = zip(*ttokens)

        # TODO: smoothing (penalize short sentences)
        # TODO: add dependency model (NP, VP, ...)
        # TODO: fix morpheme model as in paper
        # P('There was heavy rain') ~ P('There')P('was'|'There')P('heavy'|'was')P('rain'|'heavy')

        # language model
        tokens_ngrams = n_grams(tokens, n=self.n)
        language_model = sum(
            [log(self.ngram_tokens_model.get(t) or self._default_prob)
             for t in tokens_ngrams]
        )

        # morpheme model
        tags_ngrams = n_grams(tags, n=self.n)
        morpheme_model = sum(
            [log(self.ngram_tags_model.get(t) or self._default_prob)
             for t in tags_ngrams]
        )

        return language_model, morpheme_model

    def kw_log_prob(self, ttokens: List[Tuple], kws: List[Tuple]) -> float:
        """
        Example of input:
        :param ttokens: (('alice', 'NN'), ('left', 'VBD'), ('the', 'DT'), ('court', 'NN'))
        :param kws: (('left', 'VBD'), ('court', 'NN'))
        :return: (n-gram keyword-production model)
        """
        tokens, tags = zip(*ttokens)

        # keywords-production model
        tokens_ngrams = n_grams(tokens, n=self.n)

        ngrams_with_kws = set()

        for kw in kws:
            ngrams_with_kws.update(get_ngrams_containing(tokens_ngrams, kw[0]))

        kw_prod_model = sum(
            [log(self.ngram_tokens_model.get(ngram) or self._default_prob)
             for ngram in ngrams_with_kws]
        )

        return kw_prod_model


def get_vocabulary(corpora=None, reload_corpora=False, reload_phrases=False, n=1, test=False) -> Vocabulary:
    voc = load_voc()

    if not voc or reload_corpora:
        print('re')
        if not reload_phrases:
            phrases = load_phrases()
        else:
            phrases = None

        voc = Vocabulary(corpora, n=n, test=test, t_phrases=phrases)
        save_voc(voc, overwrite=True)

        if reload_phrases:
            save_phrases(voc._t_phrases)

    return voc


if __name__ == '__main__':
    corpora = load_corpora()

    vocabulary = get_vocabulary(corpora, reload_corpora=True, n=3, reload_phrases=True, test=False)
