import json
from collections import defaultdict

from src.ngrams import str2ngram
from src.tests.parsing_test import load_prods
from src.tests.settings import prods_file_path, target_labels, grammar_file_path, terminals, tag_to_symbol, \
    symbol_to_tag


def choose_best_productions(grammar, min_freq, min_count, max_count, labels):
    # occurrences_number = sum(productions.values())
    #
    # for rule in productions:
    #     productions[rule] /= occurrences_number

    best_productions = empty_productions()

    for label, productions in grammar.items():
        if label not in labels:
            continue

        sorted_productions = sorted(productions.items(),
                                    key=lambda kv: kv[1],
                                    reverse=True)

        best_label_prods = [(rule, prob)
                            for rule, prob in sorted_productions[:min_count]]

        best_label_prods.extend([(rule, prob)
                                 for rule, prob in sorted_productions[min_count:max_count]
                                 if prob > min_freq])

        print(label, len(best_label_prods))

        for rule, freq in best_label_prods:
            rule_tags = [symbol_to_tag(s) for s in str2ngram(rule, sep=' ')]

            if set(rule_tags).issubset(terminals):
                best_productions[label]['terminals'].append((rule, freq))

            else:  # rule_tags_set.issubset(not_terminals):
                best_productions[label]['not_terminals'].append((rule, freq))

            # else:
            #     # TODO:
            #     best_productions[label]['terminals'].append((rule, freq))

    return best_productions


def empty_productions():
    return {
        label: defaultdict(list)
        for label in target_labels
    }


def get_prods_str(label, rules):
    return '\n'.join(sorted(['%s -> %s' % (label, rule)
                             for rule, _ in rules]))


def get_terminals_str():
    return '\n'.join(['%s -> "%s"' % (tag_to_symbol(t), t)
                      for t in terminals])


def save_text_grammar(grammar, target_label=None):
    grammar_str = ''

    for label, prods_dict in grammar.items():
        if target_label is None or target_label == label:
            label_terminal_prods = get_prods_str(label, prods_dict['terminals'])
            label_not_terminal_prods = get_prods_str(label, prods_dict['not_terminals'])

            grammar_str += '%s\n\n%s\n\n\n' % (label_not_terminal_prods, label_terminal_prods)

    grammar_str += get_terminals_str()

    with open(grammar_file_path, mode='w', encoding='utf-8') as f:
        f.write(grammar_str)


def save_terminals(grammar):
    terminal_rules = defaultdict(dict)

    for label, prods_dict in grammar.items():
        label_terminal_rules = prods_dict['terminals']

        for rule, freq in label_terminal_rules:
            tags = rule.replace(' ', '_')

            terminal_rules[tags][label] = freq

    json.dump(terminal_rules, open('terminal_rules.json', mode='w', encoding='utf-8'))


def main():
    grammar = load_prods(prods_file_path)

    best_productions = choose_best_productions(grammar,
                                               min_freq=2,
                                               min_count=30, max_count=1500,
                                               labels=target_labels)

    save_text_grammar(best_productions)

    save_terminals(best_productions)


if __name__ == '__main__':
    main()
