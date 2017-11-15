from itertools import product
from math import log
from operator import itemgetter
from typing import List

from nltk import pos_tag, TreebankWordTokenizer, SnowballStemmer, tuple2str, str2tuple, flatten

from src.nltk_utils import Lemmatizer, split_tokens_tags, Parser
from src.utils import load_corpora, n_gram
# noinspection PyUnresolvedReferences
from src.vocabulary import get_vocabulary, Vocabulary


class TextGenerator:
    def __init__(self, corpora: str, n=1, test=False):
        self._corp_voc = get_vocabulary(corpora, n=n, test=test)
        self._lemmatizer = Lemmatizer()
        self._tokenizer = TreebankWordTokenizer()
        self._stemmer = SnowballStemmer('english', ignore_stopwords=True)
        self._parser = Parser()

    @staticmethod
    def combine_sentences(collocations: List[List[str]]) -> List[str]:
        # TODO: apply grammar restrictions
        return list(product(*collocations, repeat=1))

    def acquire_rules(self, kws: List[str]):
        pass

    def generate(self, keywords: List[str], limit=5):
        tokens = self._tokenizer.tokenize(' '.join(keywords))
        tagged_tokens = pos_tag(tokens)

        rules = []
        for t_t in tagged_tokens:
            token, tag = t_t

            lemma = self._lemmatizer.lemmatize(token, pos=tag)
            ttokens_of_kw = self._corp_voc.get_ttokens_containing(token, tag, lemma)

            # rules.append(' '.join([' '.join(str2tuple(_tt)[0] for _tt in tt)
            #                        for tt in ttokens_of_kw]))
            rules.append(ttokens_of_kw)

        sents_candidates = [' '.join(flatten(s))
                            for s in self.combine_sentences(rules)]

        # TODO: temporary
        sents_candidates = [' '.join(str2tuple(tt)[0] for tt in s.split(' '))
                            for s in sents_candidates]

        return self.evaluate(sents_candidates)[:limit]

    def process_sentence(self, s):
        # tokenize
        tokens = self._tokenizer.tokenize(s)

        # tokens pos tagging
        # ttokens = [tuple2str(tt) for tt in pos_tag(tokens)]
        ttokens = pos_tag(tokens)

        # stem
        stems = [self._stemmer.stem(token) for token in tokens]

        # lemmatize
        lemmas = [self._lemmatizer.lemmatize(tt[0], pos=tt[1]) for tt in ttokens]

        return [tuple2str(tt) for tt in ttokens], lemmas, stems

    def evaluate(self, sents):
        sents_probs = []
        # TODO: filter phrases by POS-templates

        for s in sents:
            # TODO: what to choose?
            ttokens, lemmas, stems = self.process_sentence(s)
            tokens, tags = split_tokens_tags(ttokens)

            n = self._corp_voc.n
            tokens_indexes = self._parser.parse_tokens(tokens)

            # language model
            lemmas_ngram = n_gram(lemmas, n=n, indexes=tokens_indexes)
            language_model = sum(
                [log(self._corp_voc.prob_lemma(l))
                 for l in lemmas_ngram]
            )

            # morpheme model
            tags_ngram = n_gram(tags, n=n, indexes=tokens_indexes)
            morpheme_model = sum(
                [log(self._corp_voc.prob_tag(t))
                 for t in tags_ngram]
            )

            # TODO: add for keywords model
            keywords_model = 0

            sents_probs.append(sum([language_model, morpheme_model, keywords_model]))

        return sorted(zip(sents, sents_probs), key=itemgetter(1), reverse=True)


def main():
    gen = TextGenerator(corpora=load_corpora(), n=3, test=False)

    while True:
        keywords = input('keywords: ').split(' ')

        print(gen.generate(keywords, limit=3))

if __name__ == '__main__':
    main()