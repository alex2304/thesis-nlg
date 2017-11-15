from typing import Dict

import nltk
from nltk import sent_tokenize, pos_tag_sents, flatten

from src.nltk_utils import split_tokens_tags, Lemmatizer, Tokenizer, Stemmer, Parser
from src.utils import n_gram, load_corpora, load_voc, save_voc, get_ngrams_containing


class Vocabulary:
    _default_prob = 1e-07

    def __init__(self, corpora: str, n=1, test=False):
        self.n = n
        self.test = test

        # segment sentences
        self.sents = sent_tokenize(corpora)
        if test:
            self.sents = self.sents[:5]

        # tokenize
        sents_tokens = Tokenizer().tokenize_sents(self.sents)

        # tokens pos tagging
        self.sents_tokens_tagged = pos_tag_sents(sents_tokens)
        tagged_tokens = [[nltk.tuple2str(t) for t in sent_tokens]
                         for sent_tokens in self.sents_tokens_tagged]

        # stem
        stemmer = Stemmer()
        sents_stems = [[stemmer.stem(token) for token in s_tokens]
                       for s_tokens in sents_tokens]

        # lemmatize
        # TODO: works too long and sometimes wrong: 'as' => 'a'
        self.lemmatizer = Lemmatizer()
        sents_lemmas = [[self.lemmatizer.lemmatize(tagged_token[0], pos=tagged_token[1])
                         for tagged_token in sent_tokens]
                        for sent_tokens in self.sents_tokens_tagged]

        # find lexical units
        # sents_lu = get_lexical_units(self.sents_tokens_tagged, tagged_sents=True)
        sents_lu = []

        self._create_voc(flatten(sents_stems), flatten(sents_lemmas),
                         flatten(tagged_tokens), flatten(sents_lu))

    def spec_n_gram(self, ngram_tokens: Dict, ngram_lemmas: Dict) -> Dict:
        combined_ngram = dict()
        missed_lemmas = set()

        for tagged_token_gram, tt_prob in ngram_tokens.items():
            gram_tokens = [nltk.str2tuple(tt)
                           for tt in tagged_token_gram.split(' ')]

            gram_lemmas = [self.lemmatizer.lemmatize(word=token, pos=tag)
                           for token, tag in gram_tokens]

            lemma_gram = ' '.join(gram_lemmas)

            lemma_gram_prob = ngram_lemmas.get(lemma_gram)
            if lemma_gram_prob is None:
                missed_lemmas.add(lemma_gram)
                lemma_gram_prob = self._default_prob

            combined_ngram[lemma_gram] = {
                'lemma': (lemma_gram, lemma_gram_prob),
                'ttoken': (tagged_token_gram, tt_prob)
            }

        if missed_lemmas:
            print('Missed lemmas: %s' % str(missed_lemmas))

        return combined_ngram

    def _create_voc(self, stems, lemmas, ttokens, lexical_units):
        # TODO: filter phrases by POS-templates
        tokens, tags = split_tokens_tags(ttokens)

        tokens_indexes = Parser().parse_tokens(tokens)

        # create tokens and tags n-gram
        self._tokens = n_gram(tokens, n=self.n, indexes=tokens_indexes)
        self._tags = n_gram(tags, n=self.n, indexes=tokens_indexes)
        self._tagged_tokens = n_gram(ttokens, n=self.n, indexes=tokens_indexes)

        # create lemmas n-gram
        self._lemmas = n_gram(lemmas, n=self.n, indexes=tokens_indexes)

        # create lemmas and tokens
        self._lemmas_tokens = self.spec_n_gram(self._tagged_tokens, self._lemmas)

        # create stems unigram
        self._stems = n_gram(stems, n=1, indexes=tokens_indexes)

        # create lexical units unigram
        # self._lexical_units = n_gram(lexical_units, n=1)

    ######################################################################
    # Getting elements containing text
    ######################################################################

    def get_tokens_containing(self, token: str, word_tag=None):
        return get_ngrams_containing(self._tokens, token)

    def get_lemmas_containing(self, lemma: str, word_tag=None):
        return get_ngrams_containing(self._lemmas, lemma)

    def get_lu_containing(self, lu: str):
        return get_ngrams_containing(self._lexical_units, lu)

    def get_ttokens_containing(self, token, tag, lemma):
        lemmas_tokens_ngram = get_ngrams_containing(self._lemmas_tokens, lemma)

        ttokens = []
        for lemma, gram in lemmas_tokens_ngram:
            lemma, lemma_prob = gram['lemma']
            ttoken, ttoken_prob = gram['ttoken']

            ttokens.append(ttoken)

        return ttokens

    ######################################################################
    # Probabilities
    ######################################################################

    def prob_token(self, token):
        return self._tokens.get(token) or self._default_prob

    def prob_lemma(self, lemma):
        return self._lemmas.get(lemma) or self._default_prob

    def prob_tag(self, tag):
        return self._tags.get(tag) or self._default_prob


def get_vocabulary(corpora, reload_corpora=False, n=1, test=False) -> Vocabulary:
    voc = load_voc()

    if not voc or reload_corpora or (voc.n != n or voc.test != test):
        voc = Vocabulary(corpora, n=n, test=test)
        save_voc(voc, overwrite=True)

    return voc


if __name__ == '__main__':
    vocabulary = get_vocabulary(load_corpora(), reload_corpora=True, n=3, test=False)
