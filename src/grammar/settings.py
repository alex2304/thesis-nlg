import os

# TODO:
# extract_grammar_from = 'phrases'
extract_grammar_from = 'sents'

sents_tags = ['S']

phrases_tags = ['NP', 'VP', 'PP', 'ADJP', 'ADVP']

pos_tags = ['CC',
            'CD',
            'DT',
            'EX',
            'FW',
            'IN',
            'JJ',
            'JJR',
            'JJS',
            'LS',
            'MD',
            'NN',
            'NNS',
            'NNP',
            'NNPS',
            'PDT',
            'POS',
            'PRP',
            'PRP$',
            'RB',
            'RBR',
            'RBS',
            'RP',
            'SYM',
            'TO',
            'UH',
            'VB',
            'VBD',
            'VBG',
            'VBN',
            'VBP',
            'VBZ',
            'WDT',
            'WP',
            'WP$',
            'WRB']

marks_tags = [',',
              ':',
              ';',
              '.',
              '?',
              '!',
              '-',
              '—']

tags_to_symbols = {
    'PRP$': 'PRPS',
    'WP$': 'WPS',
    ',': 'COMMA',
    '-': 'DASH',
    '—': 'DASH',
    '?': 'QUESTION',
    '!': 'EXCLAM',
    '.': 'DOT',
    ':': 'COLON',
    ';': 'SEMICOLON'
}


# rules will be extracted for symbols from target_labels
target_labels = None

# symbols which will be considered as terminals
terminals = None

# symbols which will be considered as not-terminals
not_terminals = None

# grammar extraction for sentences
if extract_grammar_from == 'sents':
    prods_file_path = os.path.join(os.path.dirname(__file__), 'productions_sents')
    grammar_file_path = os.path.join(os.path.dirname(__file__), 'grammar_sents.txt')

    grammar_out_file_path = os.path.join(os.path.dirname(__file__), 'terminal_rules_sents.json')

    target_labels = sents_tags

    terminals = marks_tags + pos_tags + phrases_tags
    not_terminals = sents_tags

# grammar extraction for phrases
else:
    prods_file_path = os.path.join(os.path.dirname(__file__), 'productions')
    grammar_file_path = os.path.join(os.path.dirname(__file__), 'grammar.txt')

    grammar_out_file_path = os.path.join(os.path.dirname(__file__), 'terminal_rules.json')

    target_labels = phrases_tags

    terminals = marks_tags + pos_tags
    not_terminals = phrases_tags

# rules in which tags not from accepted are faced, will be ignored
accepted_tags = terminals + not_terminals
