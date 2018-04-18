import traceback
from typing import List, Tuple

from nltk import CFG, RecursiveDescentParser, pos_tag, defaultdict
from tqdm import tqdm

from src.io import load_observed_tags, save_observed_tags, load_terminal_rules
from src.ngrams import ngram2str
from src.nltk_utils import Tokenizer, Lemmatizer, Stemmer
from src.tests.settings import tags_seq_to_symbols

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

generated_grammar = '''
NP -> ADJP NN
NP -> ADJP NNS
NP -> DT ADJP
NP -> DT ADJP NN
NP -> NP ADJP
NP -> NP ADVP
NP -> NP CC NP
NP -> NP JJ NN
NP -> NP JJ NNS
NP -> NP NN
NP -> NP NN NN
NP -> NP NN NNS
NP -> NP NP
NP -> NP PP
NP -> NP PP PP
NP -> NP VP

NP -> CC
NP -> CD
NP -> CD CC
NP -> CD NN
NP -> CD NNP
NP -> CD NNP NN
NP -> CD NNP NN NN
NP -> CD NNS
NP -> DT
NP -> DT CD
NP -> DT CD NNS
NP -> DT JJ
NP -> DT JJ JJ NN
NP -> DT JJ NN
NP -> DT JJ NN NN
NP -> DT JJ NNS
NP -> DT JJR NN
NP -> DT JJS
NP -> DT JJS NN
NP -> DT NN
NP -> DT NN CC NN
NP -> DT NN NN
NP -> DT NN NNP
NP -> DT NN NNS
NP -> DT NN POS
NP -> DT NNP
NP -> DT NNP NN
NP -> DT NNP NNP
NP -> DT NNPS
NP -> DT NNS
NP -> DT PRPS NNS
NP -> DT VBG NN
NP -> EX
NP -> JJ
NP -> JJ DT NN
NP -> JJ JJ NN
NP -> JJ JJ NNS
NP -> JJ NN
NP -> JJ NN NN
NP -> JJ NNP
NP -> JJ NNS
NP -> JJR
NP -> JJR NN
NP -> JJS
NP -> NN
NP -> NN CC NN
NP -> NN NN
NP -> NN NNP
NP -> NN NNS
NP -> NN RB
NP -> NNP
NP -> NNP CC NNP
NP -> NNP DT NNP
NP -> NNP NN
NP -> NNP NNP
NP -> NNP NNP NNP
NP -> NNP NNP POS
NP -> NNP POS
NP -> NNS
NP -> NNS CC NNS
NP -> PDT
NP -> PDT DT
NP -> PDT DT NN
NP -> PDT DT NNS
NP -> PRP
NP -> PRPS
NP -> PRPS JJ
NP -> PRPS JJ NN
NP -> PRPS JJ NNS
NP -> PRPS NN
NP -> PRPS NN CC NN
NP -> PRPS NN NN
NP -> PRPS NN NNS
NP -> PRPS NNP
NP -> PRPS NNS
NP -> RB
NP -> RB DT NN
NP -> RB JJ
NP -> RB NN
NP -> UH
NP -> VB
NP -> VBG
NP -> WDT
NP -> WP

VP -> ADVP VBD NP
VP -> ADVP VBG NP
VP -> ADVP VBN
VP -> ADVP VBN PP
VP -> JJ NP
VP -> MD ADVP VP
VP -> MD RB VP
VP -> MD VP
VP -> NN NP
VP -> NN PP
VP -> NNP NP
VP -> NP
VP -> NP PP
VP -> TO VP
VP -> VB ADJP
VP -> VB ADJP PP
VP -> VB ADVP
VP -> VB ADVP NP
VP -> VB ADVP PP
VP -> VB NP
VP -> VB NP ADVP
VP -> VB NP ADVP PP
VP -> VB NP NP
VP -> VB NP PP
VP -> VB NP PP PP
VP -> VB NP VP
VP -> VB PP
VP -> VB PP ADVP
VP -> VB PP NP
VP -> VB PP PP
VP -> VB VP
VP -> VBD ADJP
VP -> VBD ADJP PP
VP -> VBD ADVP
VP -> VBD ADVP ADJP
VP -> VBD ADVP NP
VP -> VBD ADVP PP
VP -> VBD ADVP VP
VP -> VBD NP
VP -> VBD NP ADVP
VP -> VBD NP NP
VP -> VBD NP PP
VP -> VBD NP PP PP
VP -> VBD PP
VP -> VBD PP NP
VP -> VBD PP PP
VP -> VBD RB ADJP
VP -> VBD RB NP
VP -> VBD RB VP
VP -> VBD VP
VP -> VBG ADJP
VP -> VBG ADVP
VP -> VBG ADVP PP
VP -> VBG NP
VP -> VBG NP PP
VP -> VBG PP
VP -> VBG PP PP
VP -> VBG VP
VP -> VBN ADJP
VP -> VBN ADJP PP
VP -> VBN ADVP
VP -> VBN ADVP PP
VP -> VBN NP
VP -> VBN NP NP
VP -> VBN NP PP
VP -> VBN PP
VP -> VBN PP PP
VP -> VBN VP
VP -> VBP ADJP
VP -> VBP ADVP
VP -> VBP ADVP PP
VP -> VBP ADVP VP
VP -> VBP NP
VP -> VBP NP ADVP
VP -> VBP NP PP
VP -> VBP PP
VP -> VBP PP PP
VP -> VBP RB VP
VP -> VBP VP
VP -> VBZ ADJP
VP -> VBZ ADVP
VP -> VBZ ADVP NP
VP -> VBZ ADVP PP
VP -> VBZ ADVP VP
VP -> VBZ NP
VP -> VBZ NP PP
VP -> VBZ PP
VP -> VBZ RB NP
VP -> VBZ RB VP
VP -> VBZ VP
VP -> VP CC VP

VP -> MD
VP -> MD RB
VP -> NN
VP -> VB
VP -> VBD
VP -> VBG
VP -> VBN
VP -> VBP
VP -> VBZ

PP -> ADVP ADJP
PP -> ADVP IN IN NP
PP -> ADVP IN NP
PP -> ADVP RB NP
PP -> ADVP TO NP
PP -> CC NP
PP -> CC PP
PP -> DT IN NP
PP -> DT PP CC PP
PP -> IN ADJP
PP -> IN ADVP
PP -> IN IN NP
PP -> IN NP
PP -> IN NP ADVP
PP -> IN NP NP
PP -> IN NP PP
PP -> IN PP
PP -> JJ IN NP
PP -> JJ NP
PP -> JJR IN NP
PP -> NN NP
PP -> NN PP
PP -> NNP NP
PP -> NP IN
PP -> NP IN NP
PP -> NP PP
PP -> PDT NP
PP -> PP ADVP
PP -> PP CC PP
PP -> PP CC PP NP
PP -> PP PP
PP -> RB ADJP
PP -> RB ADVP
PP -> RB IN ADJP
PP -> RB IN IN NP
PP -> RB IN NP
PP -> RB IN PP
PP -> RB NP
PP -> RB PP
PP -> RB TO NP
PP -> RP NP
PP -> RP PP
PP -> TO ADJP
PP -> TO NP
PP -> VBG NP
PP -> VBG PP
PP -> VBN NP
PP -> VBN PP

PP -> IN
PP -> IN CC IN
PP -> IN PRPS NNS
PP -> IN RB
PP -> JJ
PP -> NN
PP -> RB
PP -> RP
PP -> TO

ADJP -> ADJP ADJP
ADJP -> ADJP CC ADJP
ADJP -> ADJP IN ADJP
ADJP -> ADJP JJ
ADJP -> ADJP PP
ADJP -> ADVP DT
ADJP -> ADVP JJ
ADJP -> ADVP JJ PP
ADJP -> ADVP NN
ADJP -> ADVP RB JJ
ADJP -> ADVP VBN
ADJP -> ADVP VBN PP
ADJP -> DT JJ PP
ADJP -> IN JJ PP
ADJP -> IN NP
ADJP -> IN PP
ADJP -> JJ ADVP
ADJP -> JJ NP
ADJP -> JJ PP
ADJP -> JJ PP PP
ADJP -> JJR PP
ADJP -> NN PP
ADJP -> NP JJ
ADJP -> NP JJR
ADJP -> NP NP
ADJP -> NP RB
ADJP -> NP RBR
ADJP -> RB ADVP VBN PP
ADJP -> RB JJ NP
ADJP -> RB JJ PP
ADJP -> RB NP
ADJP -> RB PP
ADJP -> RB RB JJ PP
ADJP -> RB RB PP
ADJP -> RB RB VBN PP
ADJP -> RB VBN PP
ADJP -> RBR JJ PP
ADJP -> VBN PP

ADJP -> DT JJ
ADJP -> DT JJ CC JJ
ADJP -> DT JJ JJ
ADJP -> DT JJR
ADJP -> DT RB JJ
ADJP -> DT RBR JJ
ADJP -> IN
ADJP -> IN JJ
ADJP -> IN NN
ADJP -> IN RB
ADJP -> JJ
ADJP -> JJ CC JJ
ADJP -> JJ CC NN
ADJP -> JJ CC RB
ADJP -> JJ IN
ADJP -> JJ JJ
ADJP -> JJ JJR
ADJP -> JJ JJS
ADJP -> JJ NN
ADJP -> JJ RB
ADJP -> JJR
ADJP -> JJR CC JJR
ADJP -> JJR JJ
ADJP -> JJS
ADJP -> JJS JJ
ADJP -> JJS NN
ADJP -> NN
ADJP -> NN CC JJ
ADJP -> NN CC NN
ADJP -> NN JJ
ADJP -> NN RB
ADJP -> NNP
ADJP -> NNP JJ
ADJP -> RB
ADJP -> RB DT
ADJP -> RB IN
ADJP -> RB JJ
ADJP -> RB JJ CC JJ
ADJP -> RB JJ NN
ADJP -> RB JJ RB
ADJP -> RB JJR
ADJP -> RB JJS
ADJP -> RB NN
ADJP -> RB RB
ADJP -> RB RB JJ
ADJP -> RB RB VBN
ADJP -> RB RBR
ADJP -> RB RBR JJ
ADJP -> RB VBG
ADJP -> RB VBN
ADJP -> RBR
ADJP -> RBR JJ
ADJP -> RBR RB
ADJP -> RBS JJ
ADJP -> RBS JJ CC JJ
ADJP -> RBS RB
ADJP -> RBS RB JJ
ADJP -> RP
ADJP -> VBG
ADJP -> VBN
ADJP -> VBN CC JJ
ADJP -> WRB JJ

ADVP -> ADVP
ADVP -> ADVP CC ADVP
ADVP -> ADVP IN
ADVP -> ADVP JJ
ADVP -> ADVP NP
ADVP -> ADVP PP
ADVP -> ADVP RB
ADVP -> ADVP RB RB
ADVP -> DT PP
ADVP -> IN ADJP
ADVP -> IN NP
ADVP -> IN NP PP
ADVP -> IN PP
ADVP -> JJ NN PP
ADVP -> JJ NP
ADVP -> JJ PP
ADVP -> JJS PP
ADVP -> NN PP
ADVP -> NNS PP
ADVP -> NP IN
ADVP -> NP JJR
ADVP -> NP PP
ADVP -> NP RB
ADVP -> NP RB PP
ADVP -> NP RBR
ADVP -> NP RP
ADVP -> RB ADJP
ADVP -> RB ADVP
ADVP -> RB ADVP PP
ADVP -> RB JJ PP
ADVP -> RB NP
ADVP -> RB NP PP
ADVP -> RB PP
ADVP -> RB RB PP
ADVP -> RB VP
ADVP -> RBR NP
ADVP -> RBR PP
ADVP -> RP ADJP
ADVP -> RP PP

ADVP -> CC
ADVP -> CC RB
ADVP -> CD
ADVP -> DT
ADVP -> DT IN
ADVP -> DT IN RB
ADVP -> DT JJR
ADVP -> DT NN
ADVP -> DT RB
ADVP -> DT RB RB
ADVP -> DT RBR
ADVP -> DT RBS
ADVP -> DT VBZ
ADVP -> EX
ADVP -> IN
ADVP -> IN CC IN
ADVP -> IN DT
ADVP -> IN JJ
ADVP -> IN JJS
ADVP -> IN NN
ADVP -> IN PDT
ADVP -> IN RB
ADVP -> JJ
ADVP -> JJ NN
ADVP -> JJ RB
ADVP -> JJR
ADVP -> JJR IN
ADVP -> JJS
ADVP -> NN
ADVP -> NN IN
ADVP -> NN RB RB
ADVP -> NNP
ADVP -> PRP
ADVP -> RB
ADVP -> RB CC RB
ADVP -> RB DT
ADVP -> RB IN
ADVP -> RB IN DT
ADVP -> RB IN JJ
ADVP -> RB IN NN
ADVP -> RB JJ
ADVP -> RB JJ NN
ADVP -> RB JJR
ADVP -> RB NN
ADVP -> RB RB
ADVP -> RB RB IN
ADVP -> RB RB JJ
ADVP -> RB RB RB
ADVP -> RB RBR
ADVP -> RB RBR RB
ADVP -> RB RP
ADVP -> RBR
ADVP -> RBR IN
ADVP -> RBR IN RB
ADVP -> RBR RB
ADVP -> RBS
ADVP -> RBS RB
ADVP -> RP
ADVP -> RP RB
ADVP -> VB
ADVP -> WRB
'''
# possible_labels = ['NN',
#                        'NNS',
#                        'NNP',
#                        'PRP',
#                        'PRPS',
#                        'VB',
#                        'VBP',
#                        'VBZ',
#                        'VBG',
#                        'VBD',
#                        'VBN',
#                        'MD',
#                        'JJ',
#                        'JJR',
#                        'JJS',
#                        'RB',
#                        'RBR',
#                        'RBS',
#                        'DT',
#                        'IN',
#                        'TO']
# CC - coordinating conjunction
# CD - cardinal number
# PDT - predeterminers
# POS - possessive endings

# sentence grammar
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

VP -> TO VP
VP -> MD VP
VP -> V
VP -> V VP
VP -> V VP
VP -> V NP
VP -> V NP ADVP  
VP -> V NP PP

PP -> IN NP
PP -> TO NP
PP -> IN
PP -> IN PP
PP -> IN ADJP
PP -> JJ NP
PP -> ADVP IN NP
PP -> CC NP
PP -> TO
PP -> IN ADVP
PP -> RB IN NP
PP -> RB NP
PP -> IN IN NP
PP -> NP IN NP
PP -> PP CC PP
PP -> DT IN NP
PP -> IN NP ADVP
PP -> RB ADJP

ADJP -> J
ADJP -> J J
ADJP -> JJ CC JJ
ADJP -> J PP
ADJP -> R J
ADJP -> R J PP
ADJP -> R R
ADJP -> ADJP PP
ADJP -> ADJP CC ADJP
ADJP -> DT J
ADJP -> NP JJ
ADJP -> RB RB JJ

ADVP -> R
ADVP -> R R
ADVP -> RB RB RB
ADVP -> RB PP
ADVP -> RB JJ
ADVP -> RB NP
ADVP -> NP RB
ADVP -> IN DT
ADVP -> IN PP
ADVP -> ADVP PP
ADVP -> JJR

N -> NN | NNS

J -> JJ | JC
JC -> JJR | JJS

V -> VPG | VDN
VPG -> VB | VBP | VBZ | VBG
VDN -> VBD | VBN

R -> RB | RBC
RBC -> RBR | RBS

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

MD -> "MD"

JJ -> "JJ"
JJR -> "JJR"
JJS -> "JJS"

RB -> "RB"
RBR -> "RBR"
RBS -> "RBS"

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

grammars_voc = {
    'NP': {
        "NN",
        "NNS",
        "NNP",

        "PRP",
        "PRP$",

        "JJ",

        "DT",

        "IN",

        "TO"
    },

    'VP': {
        "NN",
        "NNS",
        "NNP",

        "PRP",
        "PRP$",

        "VB",
        "VBP",
        "VBZ",
        "VBG",
        "VBD",
        "VBN",

        "JJ",

        "DT",

        "IN",

        "TO",

        "RB"
    },

    'PP': {
        "NN",
        "NNS",
        "NNP",

        "PRP",
        "PRP$",

        "JJ",

        "DT",

        "IN",

        "TO"
    },

    'ADJP': {
        'JJ',

        'DT'
    },

    'ADVP': {
        'RB'
    }
}

g_parsers = [(RecursiveDescentParser(CFG.fromstring(g)), g_type)
             for g, g_type in grammars]


def preparse_tags(parser_type, _tags):
    if not grammars_voc[parser_type].issuperset(set(_tags)):
        return False

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

    observed_tags = load_observed_tags()

    if observed_tags is None:
        observed_tags = dict()

    terminal_rules = load_terminal_rules()

    for tt_gram in tqdm(tt_ngrams, desc='parsing phrases from %d-grams' % n):
        symbols = tuple(tags_seq_to_symbols([tag
                                             for _, tag in tt_gram]))

        phrase = tt_gram

        # check if tags phrase has been already observed
        tags_str = ngram2str(symbols)

        if tags_str in observed_tags:
            if observed_tags[tags_str] is not None:
                phrases.append(phrase)
                phrases_types.append(observed_tags[tags_str])

            continue

        # if tags phrase hasn't been observed
        # for p, p_type in g_parsers:
        #     if preparse_tags(p_type, tags):
        #         trees = list(p.parse(tags))
        #
        #         if trees:
        #             observed_tags[tags_str] = p_type
        #
        #             phrases.append(phrase)
        #             phrases_types.append(p_type)
        #             break

        p_types_dict = terminal_rules.get(tags_str)

        if p_types_dict:
            p_type = max(p_types_dict, key=lambda k: p_types_dict[k])

            phrases.append(phrase)
            phrases_types.append(p_type)

            observed_tags[tags_str] = p_type

        else:
            observed_tags[tags_str] = None

    try:
        save_observed_tags(observed_tags)

    except Exception as e:
        traceback.print_exc(e)

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
