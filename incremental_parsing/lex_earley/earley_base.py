from dataclasses import dataclass
from typing import NamedTuple, Tuple, Iterable, List, Callable, Union, Dict, Sequence

from incremental_parsing.lex_earley.lexer import Token
from incremental_parsing.lex_earley.simple_bnf import SimpleBNF, BNFElement, BNFTerminal, BNFNonterminal


class LexEarleyState(NamedTuple):
    span_start: int
    rule_name: str
    production_index: int
    position: int
    max_position: int

    def next_element(self, grammar: SimpleBNF) -> BNFElement:
        """
        Will error if is_complete is true
        """
        return grammar.rules[self.rule_name].productions[self.production_index].elements[self.position]

    def is_complete(self) -> bool:
        return self.position == self.max_position

    def advance(self) -> 'LexEarleyState':
        return self._replace(position=self.position + 1)

    def to_str(self, grammar: SimpleBNF, only_elements: bool = False):
        production = grammar.rules[self.rule_name].productions[self.production_index]
        prev_prods = " ".join(e.name for e in production.elements[:self.position])
        after_prods = " ".join(e.name for e in production.elements[self.position:])
        if only_elements:
            return f"{prev_prods} • {after_prods}"
        else:
            return f"{self.rule_name} -> {prev_prods} • {after_prods} ({self.span_start})"

    def reverse_position(self) -> "LexEarleyState":
        return self._replace(position=(self.max_position - self.position))


# Various ways that an Earley state could be created

@dataclass(frozen=True)
class TopLevel:
    pass


@dataclass(frozen=True)
class Scanned:
    token: Token
    from_chart_idx: int
    from_state_idx: int


@dataclass(frozen=True)
class Predicted:
    from_chart_idx: int
    from_state_idx: int


@dataclass(frozen=True)
class PredictedNullableCompletion:
    from_chart_idx: int
    from_state_idx: int
    production_name: str


@dataclass(frozen=True)
class Completed:
    finished_rule_chart_idx: int
    finished_rule_state_idx: int
    complete_into_chart_idx: int
    complete_into_state_idx: int


StateCreationMethod = Union[TopLevel, Scanned, Predicted, PredictedNullableCompletion, Completed]

StatePlusCreationMethods = Tuple[LexEarleyState, Tuple[StateCreationMethod, ...]]


# A somewhat textbook implementation of the Earley Algorithm for the rest of the file

class LexEarleyAlgorithmChart(NamedTuple):
    states: Tuple[StatePlusCreationMethods, ...]

    def reverse_state_positions(self):
        return LexEarleyAlgorithmChart(
            states=tuple((state.reverse_position(), creation_methods) for state, creation_methods in self.states)
        )

    def get_states_and_creation_methods(self) -> Sequence[Tuple[LexEarleyState, Iterable[StateCreationMethod]]]:
        return self.states


def scan_all(grammar: SimpleBNF, prev_chart: Iterable[StatePlusCreationMethods], prev_chart_idx: int, symbol: Token) \
        -> List[Tuple[LexEarleyState, StateCreationMethod]]:
    next_states_ordered: List[Tuple[LexEarleyState, StateCreationMethod]] = []
    for state_idx, (state, _) in enumerate(prev_chart):
        if not state.is_complete():
            next_element = state.next_element(grammar)
            if isinstance(next_element, BNFTerminal):
                if next_element.name == symbol.name:
                    next_states_ordered.append((state.advance(), Scanned(token=symbol,
                                                                         from_chart_idx=prev_chart_idx,
                                                                         from_state_idx=state_idx)))

    return next_states_ordered


def predictor_completer(prev_charts: Sequence[LexEarleyAlgorithmChart],
                        items_from_scanner: Sequence[Tuple[LexEarleyState, StateCreationMethod]],
                        bnf: SimpleBNF) -> Tuple[LexEarleyAlgorithmChart, Iterable[str]]:
    """
    :return: all states of this chart, and all possible terminals which could show up next
    """
    states: Dict[LexEarleyState, int] = dict()  # To index in processed_states_ordered
    states_ordered: List[Tuple[LexEarleyState, List[StateCreationMethod]]] = []
    processed_state_idx = 0
    allowed_next_symbols = set()

    def adder(earley_state: LexEarleyState, creation_method: StateCreationMethod):
        if earley_state in states:
            # Already exists, add the creation method
            states_ordered[states[earley_state]][1].append(creation_method)
        else:
            # New state
            states[earley_state] = len(states_ordered)
            states_ordered.append((earley_state, [creation_method]))

    for state, creation_method in items_from_scanner:
        adder(state, creation_method)

    while processed_state_idx < len(states_ordered):
        state_to_process, _creation_methods = states_ordered[processed_state_idx]

        if state_to_process.is_complete():
            items_from_span_start: Sequence[Tuple[LexEarleyState, Sequence[StateCreationMethod]]]

            if state_to_process.span_start == len(prev_charts):
                items_from_span_start = states_ordered
            else:
                items_from_span_start = prev_charts[state_to_process.span_start].states

            states_in_span_start = (item[0] for item in items_from_span_start)
            completer(
                items_in_span_start=states_in_span_start,
                rule_name=state_to_process.rule_name,
                adder=adder,
                grammar=bnf,
                current_chart_idx=len(prev_charts),
                span_start_idx=state_to_process.span_start,
                current_proc_state_idx=processed_state_idx
            )
        else:
            next_element = state_to_process.next_element(bnf)
            if isinstance(next_element, BNFNonterminal):
                predictor(
                    state=state_to_process,
                    next_rule_name=next_element.name,
                    grammar=bnf,
                    current_chart_idx=len(prev_charts),
                    adder=adder,
                    predicted_from_state_idx=processed_state_idx
                )
            else:
                allowed_next_symbols.add(next_element.name)

        processed_state_idx += 1

    immutable_states_ordered = tuple(
        (state, tuple(creation_methods))
        for state, creation_methods in states_ordered
    )

    return LexEarleyAlgorithmChart(states=immutable_states_ordered), allowed_next_symbols


Adder = Callable[[LexEarleyState, StateCreationMethod], None]


def completer(items_in_span_start: Iterable[LexEarleyState],
              rule_name,
              adder: Adder,
              grammar: SimpleBNF,
              span_start_idx: int,
              current_chart_idx: int,
              current_proc_state_idx: int):
    for complete_into_state_idx, previous_state in enumerate(items_in_span_start):
        if not previous_state.is_complete():
            expected_item = previous_state.next_element(grammar)
            if isinstance(expected_item, BNFNonterminal):
                if expected_item.name == rule_name:
                    adder(previous_state.advance(),
                          Completed(
                              finished_rule_chart_idx=current_chart_idx,
                              finished_rule_state_idx=current_proc_state_idx,
                              complete_into_chart_idx=span_start_idx,
                              complete_into_state_idx=complete_into_state_idx
                          ))


def predictor(state: LexEarleyState, next_rule_name: str, grammar: SimpleBNF, current_chart_idx: int, adder: Adder,
              predicted_from_state_idx: int):
    matching_rule = grammar.rules[next_rule_name]

    for i, production in enumerate(matching_rule.productions):
        adder(LexEarleyState(
            span_start=current_chart_idx,
            rule_name=next_rule_name,
            production_index=i,
            position=0,
            max_position=len(production.elements)
        ), Predicted(
            from_state_idx=predicted_from_state_idx,
            from_chart_idx=current_chart_idx
        ))

    if next_rule_name in grammar.nullable_rules:
        adder(state.advance(), PredictedNullableCompletion(
            from_chart_idx=current_chart_idx,
            from_state_idx=predicted_from_state_idx,
            production_name=next_rule_name
        ))


def process_token(grammar: SimpleBNF, charts: Sequence[LexEarleyAlgorithmChart], token: Token) -> Tuple[
    LexEarleyAlgorithmChart, Iterable[str]]:
    scanned_rules = scan_all(grammar=grammar, prev_chart=charts[-1].states, prev_chart_idx=len(charts) - 1,
                             symbol=token)
    return predictor_completer(charts, scanned_rules, grammar)


def initial_chart_allowed_tokens(grammar: SimpleBNF) -> Tuple[LexEarleyAlgorithmChart, Iterable[str]]:
    initial_earley_states = []
    for initial_rule in grammar.top_level_rules:
        for prod_idx, production in enumerate(grammar.rules[initial_rule].productions):
            initial_earley_states.append((LexEarleyState(
                span_start=0,
                rule_name=initial_rule,
                production_index=prod_idx,
                position=0,
                max_position=len(production.elements)
            ), TopLevel()))
    return predictor_completer(prev_charts=(), items_from_scanner=initial_earley_states, bnf=grammar)


def charts_completable(charts: Sequence[LexEarleyAlgorithmChart]) -> bool:
    """
    Returns true if the state is completable.
    """
    return len(charts[-1].states) > 0
