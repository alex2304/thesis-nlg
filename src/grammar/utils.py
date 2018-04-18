from typing import Union

from src.grammar.settings import tags_to_symbols, accepted_tags


symbols_to_tags = {symbol: tag
                   for tag, symbol in tags_to_symbols.items()}


def tag_to_symbol(tag):
    return tags_to_symbols.get(tag, tag)


def symbol_to_tag(symbol):
    return symbols_to_tags.get(symbol, symbol)


def symbols_seq_to_tags(symbols_sequence):
    tags = []

    for s in symbols_sequence:
        tag = symbol_to_tag(s)

        if tag not in accepted_tags:
            return tuple()

    return tuple(tags)


def tags_seq_to_symbols(tags_sequence: Union[tuple, list]) -> tuple:
    symbols = []

    for tag in tags_sequence:
        if tag not in accepted_tags:
            return tuple()

        symbols.append(tag_to_symbol(tag))

    return tuple(symbols)
