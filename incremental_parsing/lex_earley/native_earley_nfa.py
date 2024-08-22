from typing import List, Dict

from incremental_parsing.lex_earley.earley_nfa import AbstractEarleyNFA
from incremental_parsing.lex_earley.middle_earley import ContainsStatesAndCreationMethods
from incremental_parsing.lex_earley.native_earley_trie import NativeEarleyChart
from incremental_parsing.lex_earley.simple_bnf import SimpleBNF
from incremental_parsing.utils.indexable_container import IndexableContainer
from incremental_parsing.utils.simple_nfa import SimpleNFA

from incremental_parsing._native import NativeGrammar, NativeEarleyCharts


class NativeEarleyNFA(AbstractEarleyNFA):
    def __init__(self, grammar: NativeGrammar, charts: NativeEarleyCharts, state_mapping: Dict[int, int], reverse_state_mapping: List[int]):
        self._charts = charts
        self._grammar = grammar
        self._state_mapping = state_mapping
        self._reverse_state_mapping = reverse_state_mapping
        self._cache = {}

    @classmethod
    def create(cls, grammar: SimpleBNF, nfa: SimpleNFA[str, str]):
        native_grammar = grammar.to_native()

        reverse_state_mapping = list(nfa.states)
        state_mapping = {state: idx for idx, state in enumerate(reverse_state_mapping)}

        transitions = []
        for start_state, outgoing_transitions in nfa.atom_transitions_forward.items():
            for nonterminal, dests in outgoing_transitions.items():
                for dest in dests:
                    transitions.append((state_mapping[start_state], state_mapping[dest], nonterminal))

        start_states = [state_mapping[s] for s in nfa.start_states]

        charts = NativeEarleyCharts.create_earley_nfa(native_grammar, len(state_mapping), start_states,
                                                      transitions)

        return cls(native_grammar, charts, state_mapping, reverse_state_mapping)

    @property
    def charts(self) -> "NativeEarleyNFA":
        return self

    def __getitem__(self, key: int) -> ContainsStatesAndCreationMethods:
        if key in self._cache:
            return self._cache[key]

        element = NativeEarleyChart(self._grammar, self._charts, self._state_mapping[key], self._reverse_state_mapping)
        self._cache[key] = element
        return element
