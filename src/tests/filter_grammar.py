from functools import reduce
from itertools import combinations

from src.tests.parsing_test import load_prods

prods = load_prods()

phrases_prods = dict()

for phrase, productions in prods.items():
    phrases_prods[phrase] = [p for p, prob in productions.items() if prob > 50]

    print(phrase, len(phrases_prods[phrase]))

combs = list(combinations(phrases_prods.keys(), 2))
for k1, k2 in combs:
    print(k1, k2, set(phrases_prods[k1]).intersection(phrases_prods[k2]))
# print(reduce(lambda x, y: x.intersection(y), phrases_prods.values(), set()))
