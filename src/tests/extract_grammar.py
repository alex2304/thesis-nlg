from collections import defaultdict
from pprint import pprint

from src.ngrams import str2ngram
from src.tests.parsing_test import load_prods, target_labels, terminals, replacements


def choose_best_productions(productions, threshold=1.0, min_freq=0.00, count_limit=None):
    occurrences_number = sum(productions.values())

    for rule in productions:
        productions[rule] /= occurrences_number

    sorted_productions = sorted(productions.items(), key=lambda kv: kv[1], reverse=True)

    if count_limit is not None:
        productions_limit = count_limit

    else:
        productions_limit = int(len(sorted_productions) * threshold)

    best_productions = [rule
                        for rule, rule_freq in sorted_productions
                        if rule_freq > min_freq][:productions_limit]

    return best_productions


def empty_productions():
    return {
        label: defaultdict(list)
        for label in target_labels
    }


def get_prods_str(label, rules):
    return '\n'.join(sorted(['%s -> %s' % (label, rule)
                             for rule in rules]))


def get_terminals_str():
    return '\n'.join(['%s -> %s' % (t, replacements.get(t) or t)
                      for t in terminals])


def extract_grammar(target_label=None):
    grammar = load_prods()

    best_productions = empty_productions()

    for label, productions in grammar.items():
        best_label_prods = choose_best_productions(productions, threshold=0.25, min_freq=5e-5, count_limit=100)

        # print(label, len(best_label_prods))

        for rule in best_label_prods:
            rule_tags_set = set(str2ngram(rule, sep=' '))

            if rule_tags_set.isdisjoint(target_labels):
                best_productions[label]['terminals'].append(rule)

            else:
                best_productions[label]['not_terminals'].append(rule)

    # pprint(best_productions)

    grammar_str = ''

    for label, prods_dict in best_productions.items():
        if target_label is None or target_label == label:
            label_terminal_prods = get_prods_str(label, prods_dict['terminals'])
            label_not_terminal_prods = get_prods_str(label, prods_dict['not_terminals'])

            grammar_str += '%s\n\n%s\n\n\n' % (label_not_terminal_prods, label_terminal_prods)

    grammar_str += get_terminals_str()

    with open('grammar.txt', mode='w', encoding='utf-8') as f:
        f.write(grammar_str)


if __name__ == '__main__':
    extract_grammar()
