from typing import List, Tuple

from nltk import CFG, RecursiveDescentParser, pos_tag, defaultdict
from tqdm import tqdm

from src.nltk_utils import Tokenizer, Lemmatizer, Stemmer

'''
NP* (IN, TO, DT, N, JJ, PRP, PRPS):
    1) NP <- NP*
    2) VP <- V NP* (+ V)
    3) VP <- V NP* ADVP (+ V, ADVP)
    4) VP <- V NP* PP (+ V, PP)

VP* (V, RB, IN, TO, DT, N, JJ, PRP, PRPS):
    1) VP <- VP*
    2) VP <- TO VP* (+ TO)
    
PP* (IN, TO, DT, N, JJ, PRP, PRPS):
    1) NP <- DT N PP* (+ DT, N)
    2) VP <- V NP PP* (+ V, NP)

ADJP* (DT, JJ):
    1) NP <- DT ADJP* N (+ DT, N)

ADVP* (RB):
    1) VP <- V NP ADVP* (+ V, NP)
    
:keyword_i:
1) RB => get ADVPs
2) DT, JJ => get ADJPs
                  
'''
s_g = '''
S -> NP VP

NP -> DT N 
NP -> DT N PP
NP -> DT DT N
NP -> DT ADJP N
NP -> JJ N
NP -> DT JJ NN
NP -> NNP
NP -> NNP NNP
NP -> N
NP -> PRP
NP -> PRPS N

VP -> V
VP -> V NP
VP -> V NP ADVP  
VP -> V NP PP
VP -> TO VP

PP -> IN NP | TO NP
ADJP -> DT JJ
ADJP -> JJ
ADVP -> RB

N -> NN | NNS
V -> VPG | VDN
VPG -> VB | VBP | VBZ | VBG
VDN -> VBD | VBN

NN -> "NN"
NNS -> "NNS"
NNP -> "NNP"
PRP -> "PRP"
PRPS -> "PRP$"

VB -> "VB"
VBP -> "VBP"
VBZ -> "VBZ"
VBG -> "VBG"
VBD -> "VBD"
VBN -> "VBN"

RB -> "RB"
JJ -> "JJ"
DT -> "DT"
IN -> "IN"
TO -> "TO"
'''


def get_grammar_for(tag):
    g = str(s_g).lower()
    tag = str(tag).lower()

    include_tags = {tag}
    rules = defaultdict(list)
    ordered = []

    for line in g.split('\n'):
        if not line:
            continue

        line_splt = line.split(' ')

        from_tag = line_splt[0]

        if from_tag in include_tags:
            if from_tag not in ordered:
                ordered.append(from_tag)

            to_tags = line_splt[2:]
            include_tags.update(to_tags)

            rules[from_tag].append(to_tags)


np_g = '''
S -> NP

NP -> DT N 
NP -> DT N PP
NP -> DT DT N
NP -> DT ADJP N
NP -> JJ N
NP -> DT JJ NN
NP -> NNP
NP -> NNP NNP
NP -> N
NP -> PRP
NP -> PRPS N

PP -> IN NP | TO NP
ADJP -> DT JJ
ADJP -> JJ

N -> NN | NNS

NN -> "NN"
NNS -> "NNS"
NNP -> "NNP"
PRP -> "PRP"
PRPS -> "PRP$"

JJ -> "JJ"
DT -> "DT"
IN -> "IN"
TO -> "TO"
'''

vp_g = '''
S -> VP

NP -> DT N 
NP -> DT N PP
NP -> DT DT N
NP -> DT ADJP N
NP -> JJ N
NP -> DT JJ NN
NP -> NNP
NP -> NNP NNP
NP -> N
NP -> PRP
NP -> PRPS N

VP -> V
VP -> V NP
VP -> V NP ADVP  
VP -> V NP PP
VP -> TO VP

PP -> IN NP | TO NP
ADJP -> DT JJ
ADJP -> JJ
ADVP -> RB

N -> NN | NNS
V -> VPG | VDN
VPG -> VB | VBP | VBZ | VBG
VDN -> VBD | VBN

NN -> "NN"
NNS -> "NNS"
NNP -> "NNP"
PRP -> "PRP"
PRPS -> "PRP$"

VB -> "VB"
VBP -> "VBP"
VBZ -> "VBZ"
VBG -> "VBG"
VBD -> "VBD"
VBN -> "VBN"

RB -> "RB"
JJ -> "JJ"
DT -> "DT"
IN -> "IN"
TO -> "TO"
'''

pp_g = '''
S -> PP

NP -> DT N 
NP -> DT N PP
NP -> DT DT N
NP -> DT ADJP N
NP -> JJ N
NP -> DT JJ NN
NP -> NNP
NP -> NNP NNP
NP -> N
NP -> PRP
NP -> PRPS N

PP -> IN NP | TO NP
ADJP -> DT JJ
ADJP -> JJ

N -> NN | NNS

NN -> "NN"
NNS -> "NNS"
NNP -> "NNP"
PRP -> "PRP"
PRPS -> "PRP$"

JJ -> "JJ"
DT -> "DT"
IN -> "IN"
TO -> "TO"
'''

adjp_g = '''
S -> ADJP

ADJP -> DT JJ
ADJP -> JJ

JJ -> "JJ"
DT -> "DT"
'''

advp_g = '''
S -> ADVP

ADVP -> RB

RB -> "RB"
'''

grammars = [(np_g, 'NP'), (vp_g, 'VP'), (pp_g, 'PP'), (adjp_g, 'ADJP'), (advp_g, 'ADVP')]
g_parsers = [(RecursiveDescentParser(CFG.fromstring(g)), g_type)
             for g, g_type in grammars]


def preparse_tags(parser_type, _tags):
    noun_first, noun_last = {'DT', 'JJ', 'PRP', 'PRP$'}, {'PRP'}
    verb_first, verb_last = {'TO'}, {'RB'}
    pp_first = {'IN', 'TO'}
    adjp_first = {'DT', 'JJ'}

    first, last = _tags[0], _tags[-1]

    if parser_type == 'NP':
        return (first.startswith('N') or first in noun_first) and (last.startswith('N') or last in noun_last)
    elif parser_type == 'VP':
        return (first.startswith('V') or _tags[0] in verb_first) and (
            last.startswith('V') or last.startswith('N') or last in noun_last or last in verb_last)
    elif parser_type == 'PP':
        return (first in pp_first) and (last.startswith('N') or last in noun_last)
    elif parser_type == 'ADJP':
        return first in adjp_first and last == 'JJ'
    elif parser_type == 'ADVP':
        return first == 'RB' and first == last
    else:
        raise NotImplementedError(parser_type)


def parse_phrases(tt_ngrams, n) -> Tuple[List, List[List[Tuple]]]:
    phrases = []
    phrases_types = []

    for tt_gram in tqdm(tt_ngrams, desc='parsing phrases from %d-grams' % n):
        tags = [tag for _, tag in tt_gram]

        for p, p_type in g_parsers:
            # TODO: check if tags have been already observed
            if preparse_tags(p_type, tags):
                try:
                    trees = list(p.parse(tags))

                    if trees:
                        phrase = tt_gram
                        phrases.append(phrase)
                        phrases_types.append(p_type)
                        break
                except:
                    pass

    assert len(phrases_types) == len(phrases)
    return phrases_types, phrases

if __name__ == '__main__':
    gramm = vp_g

    t = Tokenizer()
    l = Lemmatizer()
    s = Stemmer()

    g = CFG.fromstring(gramm)
    p = RecursiveDescentParser(g)

    while True:
        try:
            tokens = t.tokenize(input('> ').lower())
            ttokens = pos_tag(tokens)

            stems = [s.stem(t) for t in tokens]
            lemmas = [l.lemmatize(token, tag) for token, tag in ttokens]

            terms = [tag for token, tag in ttokens]

            print(terms)

            for tree in p.parse(terms):
                tree.draw()
                break
            else:
                print('no trees')
        except ValueError as e:
            print(e)
