from collections import defaultdict
from typing import Tuple, List

import lark
import regex
from lark import Lark
from lark.grammar import Rule, Terminal, NonTerminal
from lark.indenter import PythonIndenter
from lark.lexer import Pattern as LarkPattern, Token as LarkToken
from lark.lexer import PatternRE as LarkPatternRE
from lark.lexer import PatternStr as LarkPatternStr

from incremental_parsing.lex_earley.incremental_pattern import IncrementalPattern, IncrementalPatternString, \
    IncrementalPatternRegex
from incremental_parsing.lex_earley.lex_earley import LexEarleyAlgorithmContext
from incremental_parsing.lex_earley.lexer import IncrementalLexer
from incremental_parsing.lex_earley.python_lex_wrapper import PythonLexWrapper
from incremental_parsing.lex_earley.simple_bnf import BNFTerminal, BNFNonterminal, BNFElement, SimpleBNFProduction, \
    SimpleBNFRule, \
    SimpleBNF
from incremental_parsing.utils.flags_to_regex_flags import flags_to_regex_flags


# Utilities to convert from Lark's internal representation to our internal representation


def get_name_of_rule(rule_origin: NonTerminal):
    origin_name = rule_origin.name
    if isinstance(origin_name, LarkToken):
        return origin_name.value
    elif isinstance(origin_name, str):
        return origin_name
    else:
        raise ValueError(origin_name)


def pattern_to_pattern(p: LarkPattern) -> IncrementalPattern:
    if isinstance(p, LarkPatternStr):
        return IncrementalPatternString(p.value)
    elif isinstance(p, LarkPatternRE):
        return IncrementalPatternRegex(regex.compile(p.value, flags_to_regex_flags(p.flags)))
    else:
        raise ValueError(p)


def rule_to_simple_bnf(r: Rule) -> Tuple[str, SimpleBNFProduction]:
    bnf_list: List[BNFElement] = []
    for symbol in r.expansion:
        if isinstance(symbol, Terminal):
            bnf_list.append(BNFTerminal(symbol.name))
        else:
            bnf_list.append(BNFNonterminal(symbol.name))

    return get_name_of_rule(r.origin), SimpleBNFProduction(tuple(bnf_list))


def lark_to_lex_earley_context_python(l: Lark) -> LexEarleyAlgorithmContext:
    tokens = {}
    for terminal_def in l.terminals:
        tokens[terminal_def.name] = pattern_to_pattern(terminal_def.pattern)

    # Otherwise we will interpret '''' then end of file as two strings
    # Will still get subsumed by long-strings, which is what we want
    tokens["BAD_STRING_1"] = IncrementalPatternString("''''")
    tokens["BAD_STRING_2"] = IncrementalPatternString('""""')

    lexer = IncrementalLexer(tokens)
    lexer_wrapped = PythonLexWrapper(lexer, l.ignore_tokens)

    rules = defaultdict(list)
    for rule in l.rules:
        rule_name, rule_body = rule_to_simple_bnf(rule)
        rules[rule_name].append(rule_body)

    rules_bnf = {name: SimpleBNFRule(tuple(productions)) for name, productions in rules.items()}
    grammar = SimpleBNF(rules_bnf, tuple(l.options.start))

    return LexEarleyAlgorithmContext(grammar=grammar, lexer=lexer_wrapped)


def get_python_context():
    kwargs = dict(postlex=PythonIndenter(), start='file_input')
    l = lark.Lark.open("../../grammars/python.lark", rel_to=__file__, **kwargs)
    return lark_to_lex_earley_context_python(l)


def lark_to_context(l: Lark, use_true_leftmost_longest: bool = False) -> LexEarleyAlgorithmContext:
    tokens = {}
    for terminal_def in l.terminals:
        tokens[terminal_def.name] = pattern_to_pattern(terminal_def.pattern)

    lexer = IncrementalLexer(tokens, use_true_leftmost_longest)
    rules = defaultdict(list)
    for rule in l.rules:
        rule_name, rule_body = rule_to_simple_bnf(rule)
        rules[rule_name].append(rule_body)

    rules_bnf = {name: SimpleBNFRule(tuple(productions)) for name, productions in rules.items()}
    grammar = SimpleBNF(rules_bnf, tuple(l.options.start))

    return LexEarleyAlgorithmContext(grammar=grammar, lexer=lexer)


def get_calc_lang_context():
    l = lark.Lark.open("../../grammars/calculator.lark", start='start', rel_to=__file__)
    return lark_to_context(l, True)


if __name__ == '__main__':
    get_python_context()
