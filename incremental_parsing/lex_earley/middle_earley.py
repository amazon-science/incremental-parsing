from collections import deque, defaultdict
from typing import Tuple, Set, Deque, NamedTuple, Union, List, Sequence, Iterable, Container, DefaultDict

import networkx
from typing_extensions import Protocol

from incremental_parsing.lex_earley.earley_base import LexEarleyState, TopLevel, Scanned, \
    Predicted, PredictedNullableCompletion, Completed, StateCreationMethod
from incremental_parsing.lex_earley.simple_bnf import SimpleBNF, BNFTerminal, BNFNonterminal, SimpleBNFProduction, \
    BNFElement, SimpleBNFRule
from incremental_parsing.utils.indexable_container import IndexableContainer


class ParseNodeNonTerminal(NamedTuple):
    rule_name: str
    span_start: int


class ParseNodeTerminal(NamedTuple):
    chart_idx: int
    state_idx: int

    def __str__(self):
        return f"Terminal<{self.chart_idx}, {self.state_idx}>"


class ParseEdge(NamedTuple):
    production_index: int
    position: int


class ContainsStatesAndCreationMethods(Protocol):
    def get_states_and_creation_methods(self) -> Sequence[Tuple[LexEarleyState, Iterable[StateCreationMethod]]]:
        ...

    def __getitem__(self, idx: int) -> Tuple[LexEarleyState, Iterable[StateCreationMethod]]:
        ...

    def __len__(self) -> int:
        ...


def create_parse_hierarchy(grammar: SimpleBNF,
                           charts: IndexableContainer[ContainsStatesAndCreationMethods],
                           final_chart_indices: Iterable[int],
                           reverse_state_positions: bool = False) -> networkx.MultiDiGraph:
    """
    Creates a parse hierarchy from the charts (2nd attempt)
    """

    processed_states: Set[Tuple[int, int]] = set()
    states_to_process: Deque[Tuple[int, int]] = deque()

    graph = networkx.MultiDiGraph()

    top_level = set()

    def state_to_graph_node(state: LexEarleyState) -> ParseNodeNonTerminal:
        return ParseNodeNonTerminal(state.rule_name, state.span_start)

    # Collect all scannable states
    for final_chart_idx in final_chart_indices:
        for state_idx, (state, _) in enumerate(charts[final_chart_idx].get_states_and_creation_methods()):
            if reverse_state_positions:
                state = state.reverse_position()

            if not state.is_complete() and isinstance(state.next_element(grammar), BNFTerminal):
                scannable_state = ParseNodeTerminal(final_chart_idx, state_idx)
                states_to_process.append(scannable_state)
                graph.add_node(scannable_state, style="bold", shape="box")
                graph.add_edge(scannable_state, state_to_graph_node(state),
                               key=(state.production_index, state.position),
                               label=state.to_str(grammar, True))

    while states_to_process:
        state_num = states_to_process.popleft()
        if state_num in processed_states:
            continue

        chart_idx, state_idx_in_chart = state_num
        state_currently_processing: LexEarleyState
        state_currently_processing, creation_methods = \
            charts[chart_idx].get_states_and_creation_methods()[state_idx_in_chart]
        if reverse_state_positions:
            state_currently_processing = state_currently_processing.reverse_position()

        graph_node = state_to_graph_node(state_currently_processing)

        graph_label = f"{state_currently_processing.rule_name} ({state_currently_processing.span_start})"

        graph.add_node(graph_node, label=graph_label)

        processed_states.add(state_num)

        for creation_method in creation_methods:
            if isinstance(creation_method, TopLevel):
                graph.nodes[graph_node]["style"] = "bold"
                graph.nodes[graph_node]["shape"] = "doublecircle"
                top_level.add(graph_node)
            elif isinstance(creation_method, Scanned):
                created_from_state_num = (creation_method.from_chart_idx, creation_method.from_state_idx)
                states_to_process.append(created_from_state_num)
            elif isinstance(creation_method, Predicted):
                created_from_state_num = (creation_method.from_chart_idx, creation_method.from_state_idx)
                created_from_state: LexEarleyState = \
                    charts[created_from_state_num[0]].get_states_and_creation_methods()[created_from_state_num[1]][0]
                if reverse_state_positions:
                    created_from_state = created_from_state.reverse_position()
                graph.add_edge(graph_node, state_to_graph_node(created_from_state),
                               key=(created_from_state.production_index, created_from_state.position),
                               label=created_from_state.to_str(grammar, True))
                states_to_process.append(created_from_state_num)
            elif isinstance(creation_method, PredictedNullableCompletion):
                created_from_state_num = (creation_method.from_chart_idx, creation_method.from_state_idx)
                states_to_process.append(created_from_state_num)
            elif isinstance(creation_method, Completed):
                created_from_state_num = (
                    creation_method.complete_into_chart_idx, creation_method.complete_into_state_idx)
                states_to_process.append(created_from_state_num)

    return graph


class BNFSubstringNonTerminal(NamedTuple):
    rule_name: str
    prefix_span_start: int
    suffix_span_end: int


class BNFSuffixNonTerminal(NamedTuple):
    rule_name: str
    prefix_span_start: int


class BNFPrefixNonTerminal(NamedTuple):
    rule_name: str
    suffix_span_end: int


BNFMiddleNonTerminal = Union[BNFSubstringNonTerminal, BNFSuffixNonTerminal, BNFPrefixNonTerminal, BNFNonterminal]
BNFMiddleElement = Union[BNFTerminal, BNFMiddleNonTerminal]


def to_rule_name(nonterminal: BNFMiddleNonTerminal) -> str:
    if isinstance(nonterminal, BNFSubstringNonTerminal):
        return f"{nonterminal.rule_name}<{nonterminal.prefix_span_start}-{nonterminal.suffix_span_end}>"
    elif isinstance(nonterminal, BNFSuffixNonTerminal):
        return f"{nonterminal.rule_name}<{nonterminal.prefix_span_start}->"
    elif isinstance(nonterminal, BNFPrefixNonTerminal):
        return f"{nonterminal.rule_name}<-{nonterminal.suffix_span_end}>"
    else:
        return nonterminal.name


def simplify_rule(element: BNFMiddleNonTerminal, prefix_final_chart_indices: Container[int],
                  suffix_final_chart_indices: Container[int]) -> BNFMiddleNonTerminal:
    if isinstance(element, BNFSubstringNonTerminal):
        if element.prefix_span_start in prefix_final_chart_indices and element.suffix_span_end in suffix_final_chart_indices:
            return BNFNonterminal(element.rule_name)
        elif element.prefix_span_start in prefix_final_chart_indices:
            return BNFPrefixNonTerminal(element.rule_name, suffix_span_end=element.suffix_span_end)
        elif element.suffix_span_end in suffix_final_chart_indices:
            return BNFSuffixNonTerminal(element.rule_name, prefix_span_start=element.prefix_span_start)
        else:
            return element
    elif isinstance(element, BNFSuffixNonTerminal):
        if element.prefix_span_start in prefix_final_chart_indices:
            return BNFNonterminal(element.rule_name)
        else:
            return element
    elif isinstance(element, BNFPrefixNonTerminal):
        if element.suffix_span_end in suffix_final_chart_indices:
            return BNFNonterminal(element.rule_name)
        else:
            return element
    else:
        return element


def get_rules_from_single_element(grammar: SimpleBNF,
                                  element: BNFMiddleElement,
                                  prefix_hierarchy: networkx.MultiDiGraph,
                                  suffix_hierarchy: networkx.MultiDiGraph) -> Sequence[Sequence[BNFMiddleElement]]:
    outputs: List[Sequence[BNFMiddleElement]] = []
    if isinstance(element, BNFSubstringNonTerminal):
        prefix_node = ParseNodeNonTerminal(element.rule_name, element.prefix_span_start)

        suffix_node = ParseNodeNonTerminal(element.rule_name, element.suffix_span_end)
        for incoming_prefix_node, _, (incoming_prefix_prod_idx, incoming_prefix_pos) in prefix_hierarchy.in_edges(
                prefix_node, keys=True):
            for incoming_suffix_node, _, (incoming_suffix_prod_idx, incoming_suffix_pos) in suffix_hierarchy.in_edges(
                    suffix_node, keys=True):
                if incoming_prefix_prod_idx != incoming_suffix_prod_idx:
                    continue
                if incoming_prefix_pos > incoming_suffix_pos:
                    continue

                if incoming_prefix_pos == incoming_suffix_pos:
                    if isinstance(incoming_prefix_node, ParseNodeTerminal) and isinstance(incoming_suffix_node,
                                                                                          ParseNodeTerminal):
                        outputs.append(())
                elif incoming_prefix_pos + 1 == incoming_suffix_pos:
                    orig_rule = grammar.rules[element.rule_name] \
                        .productions[incoming_prefix_prod_idx].elements[incoming_prefix_pos]

                    if isinstance(orig_rule, BNFTerminal):
                        assert isinstance(incoming_prefix_node, ParseNodeTerminal)
                        assert isinstance(incoming_suffix_node, ParseNodeTerminal)
                        outputs.append((orig_rule,))
                    else:
                        assert isinstance(orig_rule, BNFNonterminal)
                        if isinstance(incoming_prefix_node, ParseNodeNonTerminal) and isinstance(incoming_suffix_node,
                                                                                                 ParseNodeNonTerminal):
                            assert incoming_prefix_node.rule_name == incoming_suffix_node.rule_name
                            assert incoming_prefix_node.rule_name == orig_rule.name
                            outputs.append((BNFSubstringNonTerminal(orig_rule.name,
                                                                    incoming_prefix_node.span_start,
                                                                    incoming_suffix_node.span_start),))
                        elif isinstance(incoming_prefix_node, ParseNodeNonTerminal):
                            assert incoming_prefix_node.rule_name == orig_rule.name
                            outputs.append((BNFSuffixNonTerminal(orig_rule.name,
                                                                 prefix_span_start=incoming_prefix_node.span_start),))
                        elif isinstance(incoming_suffix_node, ParseNodeNonTerminal):
                            assert incoming_suffix_node.rule_name == orig_rule.name
                            outputs.append(
                                (
                                    BNFPrefixNonTerminal(orig_rule.name,
                                                         suffix_span_end=incoming_suffix_node.span_start),))
                        else:
                            outputs.append((orig_rule,))
                else:
                    prod_elements = grammar.rules[element.rule_name].productions[incoming_prefix_prod_idx].elements
                    first_rule = prod_elements[incoming_prefix_pos]
                    middle_rules = prod_elements[incoming_prefix_pos + 1:incoming_suffix_pos - 1]
                    last_rule = prod_elements[incoming_suffix_pos - 1]

                    this_production: List[BNFMiddleElement] = []
                    if isinstance(first_rule, BNFTerminal) or isinstance(incoming_prefix_node, ParseNodeTerminal):
                        this_production.append(first_rule)
                    else:
                        assert incoming_prefix_node.rule_name == first_rule.name
                        this_production.append(
                            BNFSuffixNonTerminal(first_rule.name, prefix_span_start=incoming_prefix_node.span_start))

                    this_production.extend(middle_rules)

                    if isinstance(last_rule, BNFTerminal) or isinstance(incoming_suffix_node, ParseNodeTerminal):
                        this_production.append(last_rule)
                    else:
                        assert incoming_suffix_node.rule_name == last_rule.name
                        this_production.append(
                            BNFPrefixNonTerminal(last_rule.name, suffix_span_end=incoming_suffix_node.span_start))

                    outputs.append(tuple(this_production))

    elif isinstance(element, BNFSuffixNonTerminal):
        prefix = ParseNodeNonTerminal(element.rule_name, span_start=element.prefix_span_start)
        for incoming_prefix_node, _, (incoming_prefix_prod_idx, incoming_prefix_pos) in prefix_hierarchy.in_edges(
                prefix, keys=True):
            prod_elements = grammar.rules[element.rule_name].productions[incoming_prefix_prod_idx].elements

            if incoming_prefix_pos == len(prod_elements):
                outputs.append(())
            else:
                this_production = []
                first_element = prod_elements[incoming_prefix_pos]

                if isinstance(first_element, BNFTerminal) or isinstance(incoming_prefix_node, ParseNodeTerminal):
                    this_production.append(first_element)
                else:
                    assert incoming_prefix_node.rule_name == first_element.name
                    this_production.append(
                        BNFSuffixNonTerminal(first_element.name, prefix_span_start=incoming_prefix_node.span_start))

                this_production.extend(prod_elements[incoming_prefix_pos + 1:])
                outputs.append(tuple(this_production))
    elif isinstance(element, BNFPrefixNonTerminal):
        suffix = ParseNodeNonTerminal(element.rule_name, span_start=element.suffix_span_end)
        for incoming_suffix_node, _, (incoming_suffix_prod_idx, incoming_suffix_pos) in suffix_hierarchy.in_edges(
                suffix, keys=True):
            prod_elements = grammar.rules[element.rule_name].productions[incoming_suffix_prod_idx].elements

            if incoming_suffix_pos == 0:
                outputs.append(())
            else:
                this_production = []
                this_production.extend(prod_elements[:incoming_suffix_pos - 1])

                last_element = prod_elements[incoming_suffix_pos - 1]

                if isinstance(last_element, BNFTerminal) or isinstance(incoming_suffix_node, ParseNodeTerminal):
                    this_production.append(last_element)
                else:
                    assert incoming_suffix_node.rule_name == last_element.name
                    this_production.append(
                        BNFPrefixNonTerminal(last_element.name, suffix_span_end=incoming_suffix_node.span_start))

                outputs.append(tuple(this_production))
    else:
        assert isinstance(element, BNFNonterminal)
        assert False, "Plain BNFNonterminal should have just been copied from the grammar"

    return outputs


def create_middle_bnf(grammar: SimpleBNF,
                      prefix_hierarchy: networkx.MultiDiGraph,
                      suffix_hierarchy: networkx.MultiDiGraph,
                      prefix_final_chart_indices: Container[int],
                      suffix_final_chart_indices: Container[int]) -> SimpleBNF:
    top_level: List[BNFMiddleNonTerminal] = []
    if len(prefix_hierarchy) > 0 and len(suffix_hierarchy) > 0:
        for top_level_rule_name in grammar.top_level_rules:
            if ParseNodeNonTerminal(top_level_rule_name, span_start=0) in prefix_hierarchy.nodes \
                    and ParseNodeNonTerminal(top_level_rule_name, span_start=0) in suffix_hierarchy.nodes:
                top_level.append(BNFSubstringNonTerminal(top_level_rule_name, 0, 0))
    elif len(prefix_hierarchy) > 0:
        for top_level_rule_name in grammar.top_level_rules:
            if ParseNodeNonTerminal(top_level_rule_name, span_start=0) in prefix_hierarchy.nodes:
                top_level.append(BNFSuffixNonTerminal(top_level_rule_name, prefix_span_start=0))
    elif len(suffix_hierarchy) > 0:
        for top_level_rule_name in grammar.top_level_rules:
            if ParseNodeNonTerminal(top_level_rule_name, span_start=0) in suffix_hierarchy.nodes:
                top_level.append(BNFPrefixNonTerminal(top_level_rule_name, suffix_span_end=0))
    else:
        return grammar

    queue: Deque[BNFMiddleNonTerminal] = deque(top_level)

    assert queue, "No top level rules found"
    elements = grammar.rules.copy()

    while queue:
        element = queue.popleft()
        if to_rule_name(element) in elements:
            continue

        rules = get_rules_from_single_element(grammar, element, prefix_hierarchy, suffix_hierarchy)
        finished_productions: List[SimpleBNFProduction] = []
        for rule in rules:
            production_elements: List[BNFElement] = []
            for production_element in rule:
                if isinstance(production_element, BNFTerminal):
                    production_elements.append(production_element)
                else:
                    simplified_element = simplify_rule(production_element, prefix_final_chart_indices,
                                                       suffix_final_chart_indices)
                    production_elements.append(BNFNonterminal(to_rule_name(simplified_element)))
                    queue.append(simplified_element)
            finished_productions.append(SimpleBNFProduction(tuple(production_elements)))

        elements[to_rule_name(element)] = SimpleBNFRule(productions=tuple(finished_productions))

    top_level_names = tuple(to_rule_name(element) for element in top_level)
    return SimpleBNF(rules=elements, top_level_rules=top_level_names)


def create_bnf_direct(grammar: SimpleBNF,
                      charts: IndexableContainer[ContainsStatesAndCreationMethods],
                      final_chart_indices: Iterable[int],
                      is_right_context: bool = False) -> SimpleBNF:
    """
    This method corresponds to the paper better, but creates more verbose and slightly uglier grammars
    I think that this method is actually more correct: the parse-hierarchy based method doesn't account for
    final_chart_indices in which the corresponding NFA states have outgoing transitions.
    However, this condition doesn't really occur in our usage, so it doesn't matter which method gets used
    We therefore use the parse-hierarchy method in the implementation for the grammar size (and visualization) benefits

    :param grammar: The original grammar
    :param charts: Results from Earley parsing
    :param final_chart_indices: Index of final state(s)
    :param is_right_context: Generate rules like foo<-3> instead of foo<3->
    """
    new_rules: DefaultDict[BNFSuffixNonTerminal, Set[Tuple[BNFElement, ...]]] = defaultdict(set)
    frontier: List[Tuple[int, int]] = []
    processed_states: Set[Tuple[int, int]] = set()
    top_level: Set[str] = set()

    fci = set(final_chart_indices)

    def get_rule_key(rule_name, span_start):
        # This doesn't matter _too_ much, but means it displays more cleanly (and doesn't cause issues if used for both
        # right and left context
        if is_right_context:
            return BNFPrefixNonTerminal(rule_name, span_start)
        else:
            return BNFSuffixNonTerminal(rule_name, span_start)

    for final_chart_index in fci:
        for state_idx, (state, creation_methods) in enumerate(
                charts[final_chart_index].get_states_and_creation_methods()):
            frontier.append((final_chart_index, state_idx))

            # We can add this rule, even if state is complete, but it is redundant
            new_rules[get_rule_key(state.rule_name, state.span_start)].add(
                grammar.rules[state.rule_name].productions[state.production_index].elements[state.position:])

    while len(frontier) != 0:
        chart_idx, state_idx = frontier.pop()
        if (chart_idx, state_idx) in processed_states:
            continue
        processed_states.add((chart_idx, state_idx))

        state, creation_methods = charts[chart_idx].get_states_and_creation_methods()[state_idx]

        for creation_method in creation_methods:
            if isinstance(creation_method, Scanned) or isinstance(creation_method, PredictedNullableCompletion):
                frontier.append((creation_method.from_chart_idx, creation_method.from_state_idx))
            elif isinstance(creation_method, Completed):
                frontier.append((creation_method.complete_into_chart_idx, creation_method.complete_into_state_idx))
            elif isinstance(creation_method, Predicted):
                pred_from_state, _ = charts[creation_method.from_chart_idx] \
                    .get_states_and_creation_methods()[creation_method.from_state_idx]

                pred_from_rule: SimpleBNFProduction \
                    = grammar.rules[pred_from_state.rule_name].productions[pred_from_state.production_index]
                pred_nonterminal = pred_from_rule.elements[pred_from_state.position]
                assert isinstance(pred_nonterminal, BNFNonterminal)

                pred_from_nonterminal = get_rule_key(pred_from_state.rule_name, pred_from_state.span_start)

                pred_partial_suffix_nonterminal = get_rule_key(state.rule_name, state.span_start)
                pred_rule_seq = (BNFNonterminal(to_rule_name(pred_partial_suffix_nonterminal)),) + \
                                pred_from_rule.elements[pred_from_state.position + 1:]

                assert pred_partial_suffix_nonterminal in new_rules

                new_rules[pred_from_nonterminal].add(pred_rule_seq)
                frontier.append((creation_method.from_chart_idx, creation_method.from_state_idx))
            else:
                assert isinstance(creation_method, TopLevel)
                suffix_nonterminal = get_rule_key(state.rule_name, state.span_start)
                assert suffix_nonterminal in new_rules
                top_level.add(to_rule_name(suffix_nonterminal))

    all_rules = grammar.rules.copy()
    for nonterminal, productions in new_rules.items():
        assert not isinstance(nonterminal, BNFNonterminal)

        all_rules[to_rule_name(nonterminal)] = SimpleBNFRule(
            productions=tuple(
                SimpleBNFProduction(elements=production) for production in productions
            )
        )

    return SimpleBNF(all_rules, tuple(top_level))
