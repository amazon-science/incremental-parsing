import unittest
from collections import defaultdict
from typing import NamedTuple, Union, Tuple, Dict, AbstractSet, Iterable, FrozenSet, Set, Container, \
    DefaultDict, List
from numpy.ma.testutils import assert_equal
from incremental_parsing._native import BridgeBNFElement, BridgeBNFProduction, BridgeBNFRule, BridgeBNFGrammar


class BNFTerminal(NamedTuple):
    name: str

    def __str__(self):
        return self.name

    def to_bridge(self):
        return BridgeBNFElement.Terminal(self.name)


class BNFNonterminal(NamedTuple):
    name: str

    def __str__(self):
        return self.name

    def to_bridge(self):
        return BridgeBNFElement.Nonterminal(self.name)


BNFElement = Union[BNFTerminal, BNFNonterminal]


class SimpleBNFProduction(NamedTuple):
    elements: Tuple[BNFElement, ...]

    def __str__(self):
        if len(self.elements) == 0:
            return 'λ'
        else:
            return ' '.join(map(str, self.elements))

    def reverse(self) -> "SimpleBNFProduction":
        return SimpleBNFProduction(tuple(reversed(self.elements)))

    def to_bridge(self) -> BridgeBNFProduction:
        return BridgeBNFProduction(tuple(element.to_bridge() for element in self.elements))


class SimpleBNFRule(NamedTuple):
    productions: Tuple[SimpleBNFProduction, ...]

    def __str__(self):
        return '\n | '.join(map(str, self.productions))

    def reverse(self) -> "SimpleBNFRule":
        return SimpleBNFRule(tuple(prod.reverse() for prod in self.productions))

    def to_bridge(self) -> BridgeBNFProduction:
        return BridgeBNFRule(tuple(prod.to_bridge() for prod in self.productions))


class SimpleBNF:
    rules: Dict[str, SimpleBNFRule]
    nullable_rules: AbstractSet[str]
    top_level_rules: Tuple[str, ...]

    def __init__(self, rules: Dict[str, SimpleBNFRule], top_level_rules: Tuple[str, ...],
                 remove_unreachable_rules=True):
        self.reachable_rules = frozenset(self.get_reachable_rules(top_level_rules, rules))
        if remove_unreachable_rules:
            self.rules = {rule_name: rule for rule_name, rule in rules.items() if rule_name in self.reachable_rules}
        else:
            self.rules = rules
        self.nullable_rules = frozenset(self.get_nullable_rules(self.rules))
        self.top_level_rules = top_level_rules  # Always reachable

    @staticmethod
    def get_reachable_rules(top_level_rules: Iterable[str], rules: Dict[str, SimpleBNFRule]) -> AbstractSet[str]:
        """
        Returns a list of rules that are reachable from the top level rules.
        """
        processed_reachable_rules = set()
        recently_reachable_rules = list(top_level_rules)

        while len(recently_reachable_rules) > 0:
            rule_to_check = recently_reachable_rules.pop()

            if rule_to_check not in processed_reachable_rules:
                processed_reachable_rules.add(rule_to_check)
                for production in rules[rule_to_check].productions:
                    for element in production.elements:
                        if isinstance(element, BNFNonterminal):
                            recently_reachable_rules.append(element.name)

        return processed_reachable_rules

    @staticmethod
    def get_nullable_rules(rules: Dict[str, SimpleBNFRule]) -> AbstractSet[str]:
        """
        Returns a list of rules that are nullable.
        Essentially the algorithm in
        https://github.com/jeffreykegler/old_kollos/blob/master/notes/misc/loup2.md

        We don't care if any lexer tokens are nullable here- a nullable lexer token will
        still become a parser token; the BNF rule which produces this is not nullable
        """
        nullable_rules = set()
        recently_nullable_rules = []

        rules_referenced_by_rules = defaultdict(set)

        for rule_name, body in rules.items():
            for production in body.productions:
                if len(production.elements) == 0:
                    nullable_rules.add(rule_name)
                    recently_nullable_rules.append(rule_name)
                else:
                    for element in production.elements:
                        if isinstance(element, BNFNonterminal):
                            rules_referenced_by_rules[element.name].add(rule_name)

        while len(recently_nullable_rules) > 0:
            rule_to_check = recently_nullable_rules.pop()
            for rule_referenced_by_rule in rules_referenced_by_rules[rule_to_check]:
                if rule_referenced_by_rule not in nullable_rules:
                    for production in rules[rule_referenced_by_rule].productions:
                        if all(isinstance(element, BNFNonterminal) and element.name in nullable_rules
                               for element in production.elements):
                            nullable_rules.add(rule_referenced_by_rule)
                            recently_nullable_rules.append(rule_referenced_by_rule)
                            break

        return nullable_rules

    def get_all_final_terminals(self) -> FrozenSet[str]:
        final_terminals: Set[str] = set()

        processed: Set[str] = set()
        to_process: Set[str] = set(self.top_level_rules)
        while len(to_process) > 0:
            p = to_process.pop()
            if p in processed:
                continue
            processed.add(p)

            for production in self.rules[p].productions:
                last_element = production.elements[-1]
                if isinstance(last_element, BNFTerminal):
                    final_terminals.add(last_element.name)
                else:
                    to_process.add(last_element.name)

        return frozenset(final_terminals)

    def to_bnf_ending_in(self, permissible_final_terminals: Container[str]):
        """
        Returns a BNF in which all paths must end in one of the permissible_final_terminals
        """
        last_productions_referenced_by_rules: DefaultDict[str, Set[Tuple[str, int, int]]] = defaultdict(set)

        acceptable_productions: Set[Tuple[str, int, int]] = set()
        seen_dest_rules: Set[str] = set()
        dest_rule_queue: List[str] = []

        def potential_last_elements(production: SimpleBNFProduction):
            for idx, element in reversed(list(enumerate(production.elements))):
                yield idx, element
                if isinstance(element, BNFNonterminal) and element.name in self.nullable_rules:
                    continue
                else:
                    break

        for rule_name, body in self.rules.items():
            for prod_idx, production in enumerate(body.productions):
                for end_idx, element in potential_last_elements(production):
                    if isinstance(element, BNFNonterminal):
                        last_productions_referenced_by_rules[element.name].add((rule_name, prod_idx, end_idx))
                    else:
                        if element.name in permissible_final_terminals:
                            acceptable_productions.add((rule_name, prod_idx, end_idx))
                            if rule_name not in seen_dest_rules:
                                seen_dest_rules.add(rule_name)
                                dest_rule_queue.append(rule_name)

        while len(dest_rule_queue) != 0:
            dest_rule = dest_rule_queue.pop()
            for referring_rule, referring_prod_idx, referring_prod_end_idx in last_productions_referenced_by_rules[
                dest_rule]:
                last_element = self.rules[referring_rule].productions[referring_prod_idx].elements[
                    referring_prod_end_idx]
                if isinstance(last_element, BNFTerminal) and last_element.name not in permissible_final_terminals:
                    assert False
                acceptable_productions.add((referring_rule, referring_prod_idx, referring_prod_end_idx))
                if referring_rule not in seen_dest_rules:
                    seen_dest_rules.add(referring_rule)
                    dest_rule_queue.append(referring_rule)

        modified_bnf_elements: DefaultDict[str, List[SimpleBNFProduction]] = defaultdict(list)
        for rule_name, prod_idx, prod_end_idx in acceptable_productions:
            original_production = self.rules[rule_name].productions[prod_idx]
            # Let's say that the production is A B C, and C is nullable in the original BNF
            # We add two productions: A B-final, and A B C-final, where B-final and C-final are both guaranteed to end
            # in a permissible final terminal
            # Note that C-final is no longer nullable
            last_element = original_production.elements[prod_end_idx]
            modified_last_element: BNFElement
            if isinstance(last_element, BNFTerminal):
                assert last_element.name in permissible_final_terminals
                modified_last_element = last_element
            else:
                modified_last_element = BNFNonterminal(name=(last_element.name + "/final"))

            modified_bnf_elements[rule_name].append(SimpleBNFProduction(
                original_production.elements[:prod_end_idx] + (modified_last_element,)
            ))

        rules: Dict[str, SimpleBNFRule] = self.rules.copy()
        for name, productions in modified_bnf_elements.items():
            rules[name + "/final"] = SimpleBNFRule(tuple(productions))

        top_level_rules = tuple(tlr + "/final" for tlr in self.top_level_rules if tlr in modified_bnf_elements)
        return SimpleBNF(rules, top_level_rules)

    def reverse(self) -> "SimpleBNF":
        return SimpleBNF(
            rules={rule_name: rule.reverse() for rule_name, rule in self.rules.items()},
            top_level_rules=self.top_level_rules,
            remove_unreachable_rules=False  # This transformation doesn't change reachability
        )

    def __str__(self):
        return "\n\n".join(f"{name} : {rule}" for name, rule in self.rules.items())

    def to_bridge(self):
        return BridgeBNFGrammar(rules={name: rule.to_bridge() for (name, rule) in self.rules.items()},
                                top_level_rules=self.top_level_rules)

    def to_native(self):
        return self.to_bridge().to_native()


class TestBNFEndingIn(unittest.TestCase):
    def test_calc_lang(self):
        from incremental_parsing.lex_earley.lark_grammar import get_calc_lang_context
        from incremental_parsing.lex_earley.lex_earley import LexEarleyAlgorithmContext
        from lex_earley import lex_earley_parse

        calc_lang_context = get_calc_lang_context()
        assert_equal(lex_earley_parse(calc_lang_context, "(1-2)"), True)
        assert_equal(lex_earley_parse(calc_lang_context, "(1-2)+2"), True)
        assert_equal(lex_earley_parse(calc_lang_context, "(1-2)+2+(2)"), True)

        modified_context = LexEarleyAlgorithmContext(calc_lang_context.grammar.to_bnf_ending_in(["RPAR"]),
                                                     calc_lang_context.lexer)

        assert_equal(lex_earley_parse(modified_context, "(1-2)"), True)
        assert_equal(lex_earley_parse(modified_context, "(1-2)+2"), False)
        assert_equal(lex_earley_parse(modified_context, "(1-2)+2+(2)"), True)

        modified_context = LexEarleyAlgorithmContext(calc_lang_context.grammar.to_bnf_ending_in(["NUMBER"]),
                                                     calc_lang_context.lexer)

        assert_equal(lex_earley_parse(modified_context, "(1-2)"), False)
        assert_equal(lex_earley_parse(modified_context, "(1-2)+2"), True)
        assert_equal(lex_earley_parse(modified_context, "(1-2)+2+(2)"), False)
