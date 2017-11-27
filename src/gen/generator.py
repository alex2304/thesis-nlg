from itertools import product
from operator import itemgetter
from typing import List, Tuple

from nltk import pos_tag

from src.io import load_corpora
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
        tags = [tt[1] for tt in ttokens]

        lemmas = [self.lemmatizer.lemmatize(token, pos=tag)
                  for token, tag in ttokens]

        stems = [self.stemmer.stem(t)
                 for t in tokens]

        kws_phrases = [self.voc.get_phrases_containing(l)
                       for l in lemmas]

        # pprint(kws_phrases)

        def kws_with_tag(tag) -> List[Tuple]:
            _res = set()
            for tt in ttokens:
                _token, _tag = tt
                if _tag.startswith(tag):
                    _res.add(tt)
            return list(_res)

        # TODO: create nps and vps
        nps, vps = set(), set()
        for tt, phrases in zip(ttokens, kws_phrases):
            token, tag = tt

            if tag.startswith('N') or tag in ['JJ', 'PRP', 'PRP$', 'IN', 'TO', 'DT']:
                np_phrases = phrases.get('NP', [])

                if np_phrases:
                    nps.update(np_phrases)

                    advps, pps = kws_with_tag('ADVP'), kws_with_tag('PP')

                    for vb_tt in kws_with_tag('V'):
                        vps.update([(vb_tt,) + p for p in np_phrases])

                        if advps:
                            vps.update([(vb_tt,) + p + (advp_tt,) for p in np_phrases for advp_tt in advps])

                        if pps:
                            vps.update([(vb_tt,) + p + (pp_tt,) for p in np_phrases for pp_tt in pps])

            if tag.startswith('V') or tag.startswith('N') or tag in ['JJ', 'PRP', 'PRP$', 'IN', 'TO', 'DT', 'RB']:
                vp_phrases = phrases.get('VP', [])

                if vp_phrases:
                    vps.update(phrases.get('VP', []))

                    for to_tt in kws_with_tag('TO'):
                        vps.update([(to_tt,) + p for p in vp_phrases])

            if tag.startswith('N') or tag in ['JJ', 'PRP', 'PRP$', 'IN', 'TO', 'DT']:
                pp_phrases = phrases.get('PP', [])

                # if pp_phrases:
                #     dts, ns, vs = kws_with_tag('DT'), kws_with_tag('N'), kws_with_tag('V')
                #
                #     for
        sents_candidates = self.combine_elements(nps, vps)
        print('%d sentences candidates\n' % len(sents_candidates))

        ranked_sents = self.rank_sents(sents_candidates, ttokens)

        return ranked_sents[:limit]

    def rank_sents(self, sents, kws):
        sents_probs = []
        correct_sents = []

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
            model_sum = sum([language_model, morpheme_model, kw_prod_model])

            # print('%.2f' % model_sum, '%.2f' % language_model, '%.2f' % morpheme_model, '%.2f' % kw_prod_model, s)

            sents_probs.append(model_sum)
            correct_sents.append(s)

        return sorted(zip(correct_sents, sents_probs), key=itemgetter(1), reverse=True)


def pprint_sents(sents, show_details=False):
    print('%d ranked sentences:' % len(sents))

    for s in sents:
        if show_details:
            print(s)
        else:
            sent_parts = s[0]
            s_text = ' '.join(tt[0] for p in sent_parts for tt in p).capitalize() + '.'

            print(s_text)


def main():
    gen = TextGenerator(corpora=load_corpora().lower(), n=3, test=False)

    while True:
        keywords = input('keywords: ').lower().split(' ')

        pprint_sents(gen.generate(keywords, limit=5))
        print()

if __name__ == '__main__':
    main()