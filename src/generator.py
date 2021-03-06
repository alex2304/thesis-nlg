import random
from itertools import product
from operator import itemgetter
from typing import List, Tuple

from nltk import pos_tag, pprint, defaultdict

from src.io import load_terminal_rules_sents
from src.ngrams import str2ngram
# noinspection PyUnresolvedReferences
from src.vocabulary import get_vocabulary, Vocabulary


def kws_with_tag(ttokens, tag) -> List[Tuple]:
    _res = set()

    for tt in ttokens:
        _token, _tag = tt

        if _tag.startswith(tag):
            _res.add(tt)

    return list(_res)


class TextGenerator:
    def __init__(self, corpora: str = None, n=1, test=False):
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

    def gen_sents_candidates(self, kws_ttokens, kws_phrases):
        marks = {'COMMA': [((",", ','),)],
                 'COLON': [((":", ':'),)],
                 'SEMICOLON': [((";", ';'),)],
                 # 'DOT': [((".", '.'),)],
                 'QUESTION': [(("?", '?'),)],
                 'EXCLAM': [(("!", '!'),)],
                 'DASH': [(("-", '-'),)]}

        s_patterns = ['ADJP',
                      'ADVP NP VP',
                      'ADVP VP',
                      'CC NP VP',
                      'NP',
                      'NP ADJP',
                      'NP ADVP VP',
                      'NP NP',
                      'NP VP',
                      'PP COMMA NP VP',
                      'PP NP VP',
                      'VP']

        sents = set()
        # TODO: restrict for not-unique phrases

        sents_rules = load_terminal_rules_sents()

        phrases = defaultdict(dict)

        for tt, token_phrases in zip(kws_ttokens, kws_phrases):
            token, tag = tt

            for p_type in token_phrases:
                phrases[p_type][token] = token_phrases[p_type]

        for sents_rule in sents_rules:
            sent_symbols = str2ngram(sents_rule)

            sent = []
            used_kws = set()

            # TODO: different sent candidates with same rule
            for p_type in sent_symbols:
                if p_type in marks:
                    p_type_phrases = marks.get(p_type, [])

                else:
                    kws_p_type_phrases = phrases.get(p_type, {})

                    if kws_p_type_phrases:
                        possible_kws = set(kws_p_type_phrases.keys()).difference(used_kws)

                        if not possible_kws:
                            p_type_phrases = []

                        else:
                            kw = random.choice(list(possible_kws))

                            p_type_phrases = kws_p_type_phrases[kw]

                            used_kws.add(kw)

                    else:
                        p_type_phrases = []

                if not p_type_phrases:
                    break

                sent.append(random.choice(list(p_type_phrases)))
                # for phr_cand in list(p_type_phrases):
                #     if phr_cand not in sent:
                #         sent.append(phr_cand)
                #         break
                # else:
                #     break

            else:
                sents.add(tuple(sent))

            # sent_raw = [phrases.get(p_type, []) or marks.get(p_type, []) for p_type in sent_symbols]

            # print(sent_raw)

            # sents.update(self.combine_elements(*sent_raw))

        return sents
        # for p in s_patterns:
        #     p_types = p.split(' ')
        #
        #     sents.update(self.combine_elements(*[phrases.get(p_type, []) for p_type in p_types]))
        #
        # return sents

    def generate(self, keywords: List[str], limit=5):
        tokens = self.tokenizer.tokenize(' '.join(keywords))
        ttokens = pos_tag(tokens)

        lemmas = [self.lemmatizer.lemmatize(token, pos=tag)
                  for token, tag in ttokens]

        kws_phrases = [self.voc.get_phrases_containing(l)
                       for l in lemmas]

        pprint(kws_phrases)

        phrases = defaultdict(set)

        for tt, token_phrases in zip(ttokens, kws_phrases):
            token, tag = tt

            for p_type in token_phrases:
                phrases[p_type].update(token_phrases[p_type])

            # if tag.startswith('N') or tag in ['JJ', 'PRP', 'PRP$', 'IN', 'TO', 'DT']:
            #     np_phrases = token_phrases.get('NP', [])
            #
            #     if np_phrases:
            #         nps.update(np_phrases)
            #
            #         advps, pps = kws_with_tag(ttokens, 'ADVP'), kws_with_tag(ttokens, 'PP')
            #
            #         for vb_tt in kws_with_tag(ttokens, 'V'):
            #             vps.update([(vb_tt,) + p for p in np_phrases])
            #
            #             if advps:
            #                 vps.update([(vb_tt,) + p + (advp_tt,) for p in np_phrases for advp_tt in advps])
            #
            #             if pps:
            #                 vps.update([(vb_tt,) + p + (pp_tt,) for p in np_phrases for pp_tt in pps])
            #
            # if tag.startswith('V') or tag.startswith('N') or tag in ['JJ', 'PRP', 'PRP$', 'IN', 'TO', 'DT', 'RB']:
            #     vp_phrases = token_phrases.get('VP', [])
            #
            #     if vp_phrases:
            #         vps.update(token_phrases.get('VP', []))
            #
            #         for to_tt in kws_with_tag(ttokens, 'TO'):
            #             vps.update([(to_tt,) + p for p in vp_phrases])


            # if tag.startswith('N') or tag in ['JJ', 'PRP', 'PRP$', 'IN', 'TO', 'DT']:
            #     pp_phrases = token_phrases.get('PP', [])

            # if pp_phrases:
            #     dts, ns, vs = kws_with_tag('DT'), kws_with_tag('N'), kws_with_tag('V')
            #
            #     for

        # sents_candidates = self.combine_elements(nps, vps)

        # sents_candidates = self.gen_sents_candidates(phrases)

        sents_candidates = self.gen_sents_candidates(ttokens, kws_phrases)

        print('%d sentences candidates\n' % len(sents_candidates))

        return list(sents_candidates)[:100]

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
            # np, vp = s
            # sent_ttokens = np + vp

            sent_ttokens = []

            for phrase in s:
                sent_ttokens.extend(phrase)

            # skip sentence if not all keywords are presented
            # if not set(sent_ttokens).issuperset(kws):
            #     continue

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
            sent_parts = s  # TODO:
            s_text = ' '.join(tt[0] for p in sent_parts for tt in p).capitalize() + '.'

            print(s_text)


def main():
    # corpora = load_corpora().lower()
    corpora = None

    gen = TextGenerator(corpora=corpora, n=3, test=False)

    while True:
        keywords = input('keywords: ').lower().split(' ')

        pprint_sents(gen.generate(keywords, limit=5))
        print()


if __name__ == '__main__':
    main()
