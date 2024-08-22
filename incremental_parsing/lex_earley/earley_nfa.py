import abc
import unittest
from abc import abstractmethod
from collections import deque, defaultdict
from typing import List, Tuple, Set, Dict, Deque, DefaultDict, Iterable, Sequence, Callable, FrozenSet

from incremental_parsing.lex_earley.earley_base import LexEarleyState, StateCreationMethod, \
    completer, predictor, Completed, Scanned, TopLevel
from incremental_parsing.lex_earley.lexer import Token
from incremental_parsing.lex_earley.middle_earley import ContainsStatesAndCreationMethods
from incremental_parsing.lex_earley.simple_bnf import SimpleBNF, BNFNonterminal, SimpleBNFRule, SimpleBNFProduction, \
    BNFTerminal
from incremental_parsing.utils.indexable_container import IndexableContainer
from incremental_parsing.utils.simple_nfa import SimpleNFA, SimpleNFAMutable

StatePlusCreationMethodsMut = Tuple[LexEarleyState, Set[StateCreationMethod]]


# This file implements a modified version of the Earley algorithm for parsing a regular language
# To our knowledge, nobody has performed this modification so far
# Parsing a regular string becomes a special case (the regular language that only accepts one string)


class EarleyNFAChart:
    chart_idx: int
    processed_states_ordered: List[StatePlusCreationMethodsMut]
    # Invariant: processed_states_ordered[processed_states[state]][0] == state
    processed_states: Dict[LexEarleyState, int]
    """
    When a complete parse of nonterminal A is encountered in chart i (with span start j), the Earley algorithm's 
    completer searches chart j to see if there are any states of the form ɑ•Aβ (k). 
    If so, it will add ɑA•β (k) to chart i.
    However, with NFA Earley, we might add something to chart j after the completer runs.
    Therefore, the completer also adds a completer hook; if later on, a state which expects A appears in chart j,
    then an advanced state will also be added to chart i.
    This dict is (A)->[(i, n)] where n is the index of the complete state in chart i.
    """
    completer_hooks: DefaultDict[str, List[Tuple[int, int]]]

    def __init__(self, chart_idx: int):
        self.chart_idx = chart_idx
        self.processed_states_ordered = []
        self.processed_states = dict()
        self.completer_hooks = defaultdict(list)

    def process_new_state(self, state: LexEarleyState, context: "EarleyNFA") -> int:
        """
        Add a new state to this chart, and perform the earley algorithm to maintain invariants
        :return: New state index
        """

        def adder(new_state: LexEarleyState, creation_method: StateCreationMethod):
            context.add_state(self.chart_idx, new_state, creation_method)

        assert state not in self.processed_states
        state_idx = len(self.processed_states_ordered)
        self.processed_states[state] = state_idx
        self.processed_states_ordered.append((state, set()))

        if state.is_complete():
            states_from_span_start = (item[0] for item in context._charts[state.span_start].processed_states_ordered)

            completer(
                items_in_span_start=states_from_span_start,
                rule_name=state.rule_name,
                current_proc_state_idx=state_idx,
                current_chart_idx=self.chart_idx,
                span_start_idx=state.span_start,
                grammar=context.grammar,
                adder=adder
            )

            context._charts[state.span_start].completer_hooks[state.rule_name].append((self.chart_idx, state_idx))
        else:
            next_element = state.next_element(context.grammar)
            if isinstance(next_element, BNFNonterminal):
                predictor(
                    state=state,
                    next_rule_name=next_element.name,
                    grammar=context.grammar,
                    current_chart_idx=self.chart_idx,
                    adder=adder,
                    predicted_from_state_idx=state_idx
                )

                if next_element.name in self.completer_hooks:
                    for (finished_rule_chart_idx, finished_rule_state_idx) in self.completer_hooks[next_element.name]:
                        context.add_state(finished_rule_chart_idx, state.advance(),
                                          Completed(
                                              finished_rule_chart_idx=finished_rule_chart_idx,
                                              finished_rule_state_idx=finished_rule_state_idx,
                                              complete_into_chart_idx=self.chart_idx,
                                              complete_into_state_idx=state_idx
                                          ))

            else:
                if self.chart_idx in context.nfa.atom_transitions_forward \
                        and next_element.name in context.nfa.atom_transitions_forward[self.chart_idx]:
                    for next_chart_idx in context.nfa.atom_transitions_forward[self.chart_idx][next_element.name]:
                        context.add_state(next_chart_idx, state.advance(),
                                          Scanned(from_chart_idx=self.chart_idx,
                                                  from_state_idx=state_idx))

        return state_idx

    def get_states_and_creation_methods(self) -> Sequence[Tuple[LexEarleyState, Iterable[StateCreationMethod]]]:
        return self.processed_states_ordered

    def __len__(self):
        return len(self.processed_states_ordered)

    def __getitem__(self, index : int) -> Tuple[LexEarleyState, Iterable[StateCreationMethod]]:
        return self.processed_states_ordered[index]


class AbstractEarleyNFA(abc.ABC):
    @classmethod
    @abstractmethod
    def create(cls, grammar: SimpleBNF, nfa: SimpleNFA[str, str]):
        pass

    @property
    @abstractmethod
    def charts(self) -> IndexableContainer[ContainsStatesAndCreationMethods]:
        pass


class EarleyNFA(AbstractEarleyNFA):
    grammar: SimpleBNF
    nfa: SimpleNFA[str, str]
    _charts: Dict[int, EarleyNFAChart]
    state_processing_queue: Deque[Tuple[int, LexEarleyState, StateCreationMethod]]

    def __init__(self, grammar: SimpleBNF, nfa: SimpleNFA[str, str]):
        self.grammar = grammar
        self.nfa = nfa
        self._charts = {chart_idx: EarleyNFAChart(chart_idx) for chart_idx in self.nfa.states}
        self.state_processing_queue = deque()
        self._add_initial_states_to_queue()
        self._process_all_states()

    @classmethod
    def create(cls, grammar: SimpleBNF, nfa: SimpleNFA[str, str]):
        return cls(grammar, nfa)

    def _add_top_level_to_chart(self, chart_idx: int):
        for initial_rule in self.grammar.top_level_rules:
            for prod_idx, production in enumerate(self.grammar.rules[initial_rule].productions):
                self.state_processing_queue.append((chart_idx, LexEarleyState(
                    span_start=chart_idx,
                    rule_name=initial_rule,
                    production_index=prod_idx,
                    position=0,
                    max_position=len(production.elements)
                ), TopLevel()))

    def _add_initial_states_to_queue(self):
        for initial_nfa_state in self.nfa.start_states:
            self._add_top_level_to_chart(initial_nfa_state)

    def _process_all_states(self):
        while len(self.state_processing_queue) > 0:
            chart_idx, state, creation_method = self.state_processing_queue.popleft()
            self._process_state(chart_idx, state, creation_method)

    def _process_state(self, chart_idx: int, state: LexEarleyState, creation_method: StateCreationMethod):
        state_idx = self._charts[chart_idx].processed_states.get(state, None)
        if state_idx is None:
            state_idx = self._charts[chart_idx].process_new_state(state=state, context=self)

        self._charts[chart_idx].processed_states_ordered[state_idx][1].add(creation_method)

    def add_state(self, chart_idx: int, state: LexEarleyState, creation_method: StateCreationMethod):
        self.state_processing_queue.append((chart_idx, state, creation_method))

    def is_valid_final_state(self, state_idx: int) -> bool:
        return any(state.is_complete() and state.rule_name in self.grammar.top_level_rules
                   and state.span_start in self.nfa.start_states
                   for state, _ in self._charts[state_idx].processed_states_ordered)

    def is_complete(self) -> bool:
        return any(self.is_valid_final_state(state) for state in self.nfa.end_states)

    @property
    def charts(self) -> IndexableContainer[ContainsStatesAndCreationMethods]:
        return self._charts


def token_to_nfa(token: Token, nfa: SimpleNFAMutable, prev_state: int, token_end_state: int):
    """
    Note that this adds an outgoing transition from token_end_state;
    if the NFA gets composed in certain ways you need to make sure this won't cause issues
    """
    if token.loose_behavior:
        nfa.add_eps_transition(prev_state, token_end_state)
        if token.max_loosiness is None:
            nfa.add_atom_transition(token_end_state, token_end_state, token.name)
        else:
            prev_state_in_chain = prev_state
            for i in range(token.max_loosiness):
                next_state_in_chain = nfa.add_state()
                nfa.add_atom_transition(prev_state_in_chain, next_state_in_chain, token.name)
                nfa.add_eps_transition(next_state_in_chain, token_end_state)
                prev_state_in_chain = next_state_in_chain
    else:
        nfa.add_atom_transition(prev_state, token_end_state, token.name)


def tokens_to_nfa(tokens: Iterable[Token]) -> Tuple[SimpleNFA[str, str], FrozenSet[int]]:
    """
    Turn a string of tokens (which may include repeats) into a NFA that represents this string
    """
    nfa: SimpleNFAMutable[str] = SimpleNFAMutable()
    prev_state = 0

    for token in tokens:
        token_end_state = nfa.add_state()
        token_to_nfa(token, nfa, prev_state, token_end_state)

        prev_state = token_end_state

    nfa.end_states = {prev_state}

    all_final_states = nfa.compute_backwards_eps_reachability(prev_state)

    comparison: Callable[[str, str], bool] = lambda a, b: a == b
    finalized_nfa = nfa.finalize(comparison)
    return finalized_nfa, frozenset(all_final_states.intersection(finalized_nfa.end_states))


def token_branches_to_nfa(token_streams: Sequence[Sequence[Token]]) -> Tuple[
    SimpleNFA[str, str], List[Tuple[List[int], FrozenSet[int]]]]:
    """
    When there are multiple languages that have identical prefixes, you can model all of these languages as a single
    NFA (as long as there are no outoing transitions once the languages diverge).
    These languages just have different final states, but they share the underlying automaton,
    and this saves a lot of work
    See test_parallel_token_streams for the desired result
    """
    nfa: SimpleNFAMutable[str] = SimpleNFAMutable()

    equiv_sets = [(list(i for i in range(len(token_streams))), 0)]
    final_equiv_sets: List[Tuple[List[int], int]] = []

    max_len = max(len(x) for x in token_streams)
    for index in range(max_len + 1):
        next_equiv_sets: List[Tuple[List[int], int]] = []
        for equiv_set, prev_end_state in equiv_sets:
            child_equiv_sets: Dict[Token, Tuple[List[int], int]] = dict()
            child_none_equiv_set: List[int] = []
            for prev_stream_idx_in_equiv_set in equiv_set:
                assert index <= len(token_streams[prev_stream_idx_in_equiv_set])
                if index == len(token_streams[prev_stream_idx_in_equiv_set]):
                    child_none_equiv_set.append(prev_stream_idx_in_equiv_set)
                else:
                    next_token = token_streams[prev_stream_idx_in_equiv_set][index]
                    if next_token in child_equiv_sets:
                        child_equiv_sets[next_token][0].append(prev_stream_idx_in_equiv_set)
                    else:
                        next_state = nfa.add_state()
                        token_to_nfa(next_token, nfa, prev_end_state, next_state)
                        child_equiv_sets[next_token] = ([prev_stream_idx_in_equiv_set], next_state)
            if len(child_none_equiv_set) > 0:
                final_equiv_sets.append((child_none_equiv_set, prev_end_state))
            next_equiv_sets.extend(child_equiv_sets.values())
        if index == max_len:
            assert len(next_equiv_sets) == 0
        equiv_sets = next_equiv_sets

    seen_branches = []
    branches_and_final_states = []
    flat_final_states: Set[int] = set()

    for branches_in_final_set, final_state in final_equiv_sets:
        seen_branches.extend(branches_in_final_set)
        these_final_states = frozenset(nfa.compute_backwards_eps_reachability(final_state))
        flat_final_states.update(these_final_states)
        branches_and_final_states.append((branches_in_final_set, these_final_states))

    assert len(seen_branches) == len(set(seen_branches))
    assert len(seen_branches) == len(token_streams)

    nfa.end_states = set(flat_final_states)

    comparison: Callable[[str, str], bool] = lambda a, b: a == b
    nfa_finalized = nfa.finalize(comparison)
    return nfa_finalized, [(branch_ids, final_states.intersection(nfa_finalized.end_states))
                           for (branch_ids, final_states) in branches_and_final_states]


class TestEarleyNFA(unittest.TestCase):
    def test_parse_orig_example(self):
        example_bnf = SimpleBNF(rules={
            "S": SimpleBNFRule((
                SimpleBNFProduction((BNFTerminal("b"), BNFNonterminal("A"))),
                SimpleBNFProduction((BNFNonterminal("A"), BNFTerminal("d")))
            )),
            "A": SimpleBNFRule((SimpleBNFProduction((BNFTerminal("b"), BNFTerminal("c"))),))
        }, top_level_rules=("S",))

        example_token_stream = [Token("b", "", True, 1), Token("c", "")]
        nfa, _ = tokens_to_nfa(example_token_stream)
        enfa = EarleyNFA(example_bnf, nfa)
        self.assertFalse(enfa.is_complete())

        example_token_stream = [Token("b", "", True, 2), Token("c", "")]
        nfa, _ = tokens_to_nfa(example_token_stream)
        enfa = EarleyNFA(example_bnf, nfa)
        self.assertTrue(enfa.is_complete())

    def test_parallel_token_streams(self):
        example_streams = [
            [Token("a", ""), Token("b", "", True, 1), Token("c", "")],
            [Token("a", ""), Token("b", "", True, 2), Token("c", "")],
            [Token("a", ""), Token("b", "", True, 2), Token("c", "")],
            [Token("d", ""), Token("b", "", True, 1), Token("c", "")],
            [Token("d", ""), Token("b", "", True, None)],
            [Token("d", ""), Token("b", "", True, 2), Token("c", "", True, 1)],
        ]

        res_nfa, streams = token_branches_to_nfa(example_streams)
        self.assertEqual(set(tuple(s[0]) for s in streams),
                         {(0,), (1, 2), (3,), (4,), (5,)})

        def assertMatchesBranch(sequence, branch, true_or_false):
            branch_endpoints = None
            for branches, endpoints in streams:
                if branch in branches:
                    branch_endpoints = endpoints
                    break

            assert branch_endpoints is not None
            states = res_nfa.start_states
            for s in sequence:
                states = res_nfa.step_forward(states, s)

            if true_or_false:
                self.assertTrue(any(endpoint in states for endpoint in branch_endpoints))
            else:
                self.assertFalse(any(endpoint in states for endpoint in branch_endpoints))

        def assertMatchesBranchAll(sequence, branches):
            for branch in range(len(example_streams)):
                assertMatchesBranch(sequence, branch, branch in branches)

        assertMatchesBranchAll(("a", "c"), (0, 1, 2))
        assertMatchesBranchAll(("a", "b", "c"), (0, 1, 2))
        assertMatchesBranchAll(("a", "b", "b", "c"), (1, 2))
        assertMatchesBranchAll(("d", "b", "b"), (4, 5))
        assertMatchesBranchAll(("d", "b", "b", "c"), (5,))
        assertMatchesBranchAll(("d", "b", "b", "b", "b"), (4,))
        assertMatchesBranchAll(("d", "b"), (4, 5))
        self.assertFalse(res_nfa.fullmatch(("a", "b", "b")))
        self.assertFalse(res_nfa.fullmatch(("a", "b", "b", "b", "c")))
