import os
from collections import defaultdict

from src.ngrams import str2ngram
from src.tests.parsing_test import load_prods, target_labels, terminals, replacements

grammar_file_path = os.path.join(os.path.dirname(__file__), 'grammar.txt')


def choose_best_productions(productions, min_freq, min_count, max_count):
    occurrences_number = sum(productions.values())

    for rule in productions:
        productions[rule] /= occurrences_number

    sorted_productions = sorted(productions.items(), key=lambda kv: kv[1], reverse=True)

    best_productions = [rule for rule, _ in sorted_productions[:min_count]]

    best_productions.extend([rule
                             for rule, rule_freq in sorted_productions[min_count:max_count]
                             if rule_freq > min_freq])

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
    return '\n'.join(['%s -> "%s"' % (replacements.get(t) or t, t)
                      for t in terminals])


def extract_grammar(min_freq, min_count, max_count, target_label=None):
    grammar = load_prods()

    best_productions = empty_productions()

    for label, productions in grammar.items():
        best_label_prods = choose_best_productions(productions,
                                                   min_freq=min_freq,
                                                   min_count=min_count, max_count=max_count)

        print(label, len(best_label_prods))

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

    with open(grammar_file_path, mode='w', encoding='utf-8') as f:
        f.write(grammar_str)


if __name__ == '__main__':
    extract_grammar(min_freq=3e-3,
                    min_count=30, max_count=50)
