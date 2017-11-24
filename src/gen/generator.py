from itertools import product
from operator import itemgetter
from typing import List

from nltk import pos_tag, pprint
from tqdm import tqdm

from src.utils import load_corpora
# noinspection PyUnresolvedReferences
from src.vocabulary import get_vocabulary, Vocabulary, WordInfo


class TextGenerator:
    def __init__(self, corpora: str, n=1, test=False):
        voc = get_vocabulary(corpora, n=n, test=test)
        self.voc = voc

        self.lemmatizer = voc.lemmatizer
        self.tokenizer = voc.tokenizer
        self.stemmer = voc.stemmer
        self.parser = voc.parser

        print('Generator created')

    @staticmethod
    def combine_elements(*args) -> List[str]:
        return list(product(*args, repeat=1))

    def generate(self, keywords: List[str], limit=5):
        tokens = self.tokenizer.tokenize(' '.join(keywords))
        ttokens = pos_tag(tokens)

        lemmas = [self.lemmatizer.lemmatize(token, pos=tag)
                  for token, tag in ttokens]

        stems = [self.stemmer.stem(t)
                 for t in tokens]

        kws_phrases = [self.voc.get_phrases_containing(l)
                       for l in lemmas]

        # kws_phrases = dict(zip(ttokens, phrases))
        # pprint(kws_phrases)

        # TODO: create nps and vps
        nps, vps = set(), set()
        for tt, phrases in zip(ttokens, kws_phrases):
            token, tag = tt

            nps.update(phrases.get('NP', []))
            vps.update(phrases.get('VP', []))

        sents_candidates = self.combine_elements(nps, vps)
        print('%d sentences candidates' % len(sents_candidates))

        ranked_sents = self.rank_sents(sents_candidates, ttokens)

        return ranked_sents[:limit]

    def rank_sents(self, sents, kws):
        sents_probs = []

        for s in sents:
            # kws = ['rush', 'court']
            # s = (NP, VP)
            # NP = (('a', 'DT'), ('rush', 'NN'), ('at', 'IN'), ('alice', 'NN'))
            # VP = (('left', 'VBD'), ('the', 'DT'), ('court', 'NN'))
            # TODO: different types of sentence
            np, vp = s
            sent_ttokens = np + vp

            # skip sentence if not all keywords are presented
            if not set(sent_ttokens).issuperset(kws):
                continue

            language_model, morpheme_model = self.voc.sent_model(*sent_ttokens)
            kw_prod_model = self.voc.kw_log_prob(sent_ttokens, kws)

            sents_probs.append(sum([language_model, morpheme_model, kw_prod_model]))

        return sorted(zip(sents, sents_probs), key=itemgetter(1), reverse=True)


def main():
    gen = TextGenerator(corpora=load_corpora().lower(), n=3, test=False)

    while True:
        keywords = input('keywords: ').split(' ')

        print(gen.generate(keywords, limit=3))

if __name__ == '__main__':
    main()