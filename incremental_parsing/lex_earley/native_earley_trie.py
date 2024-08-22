from builtins import str
from typing import Tuple, overload, List, Sequence, Dict, Iterable
from typing_extensions import Self

from incremental_parsing.lex_earley.earley_base import LexEarleyAlgorithmChart, LexEarleyState, StateCreationMethod, \
    TopLevel, Scanned, Predicted, PredictedNullableCompletion, Completed
from incremental_parsing.lex_earley.earley_trie import AbstractEarleyTrieNode
from incremental_parsing.lex_earley.lexer import Token

from incremental_parsing._native import NativeGrammar, NativeEarleyCharts

from incremental_parsing.lex_earley.simple_bnf import SimpleBNF
from incremental_parsing.utils.indexable_container import IndexableContainer


def deserialize_state_creation_method(info: List[int], chart_map: IndexableContainer[int]) -> List[StateCreationMethod]:
    current_index = 0
    result = []
    while current_index < len(info):
        val_at_idx = info[current_index]
        if val_at_idx == 0:
            result.append(TopLevel())
            current_index += 1
        elif val_at_idx == 1:
            result.append(Scanned(chart_map[info[current_index + 1]], info[current_index + 2]))
            current_index += 3
        elif val_at_idx == 2:
            result.append(Predicted(chart_map[info[current_index+1]], info[current_index+2]))
            current_index += 3
        elif val_at_idx == 3:
            result.append(PredictedNullableCompletion(chart_map[info[current_index+1]], info[current_index+2]))
            current_index += 3
        elif val_at_idx == 4:
            result.append(Completed(chart_map[info[current_index+1]], info[current_index+2], chart_map[info[current_index+3]], info[current_index+4]))
            current_index += 5

    return result


class IdentMap:
    def __init__(self):
        pass

    def __getitem__(self, item):
        return item


class NativeEarleyChart:
    def __init__(self, grammar: NativeGrammar, charts: NativeEarleyCharts, index: int, chart_map: IndexableContainer[int] = IdentMap()):
        self.grammar = grammar
        self.charts = charts
        self.index = index
        self.chart_map = chart_map
        self.cache = [None for _ in range(len(self))]

    def __getitem__(self, item: int) -> Tuple[LexEarleyState, Iterable[StateCreationMethod]]:
        if self.cache[item] is not None:
            return self.cache[item]

        span_start, name, prod_idx, dot_idx, prod_length, creation_methods = self.charts.get_earley_state(self.grammar, self.index, item)
        result = (LexEarleyState(
            span_start=self.chart_map[span_start],
            rule_name=name,
            position=dot_idx,
            max_position=prod_length,
            production_index=prod_idx
        ), deserialize_state_creation_method(creation_methods, self.chart_map))

        self.cache[item] = result
        return result

    def __len__(self):
        return self.charts.get_chart_len(self.index)

    def get_states_and_creation_methods(self) -> Sequence[Tuple[LexEarleyState, Iterable[StateCreationMethod]]]:
        return [self[i] for i in range(len(self))]

    def __repr__(self):
        return list(self[i] for i in range(len(self))).__repr__()


class NativeEarleyTrieNode(AbstractEarleyTrieNode):
    @classmethod
    def create_root(cls, grammar: SimpleBNF):
        native_grammar = grammar.to_native()
        charts, root_chart_num, allowed_terminals, completable, complete = NativeEarleyCharts.create_initial_earley_charts(native_grammar)
        return cls(
            native_grammar=native_grammar,
            native_charts=charts,
            chart_num=root_chart_num,
            allowed_token_names=allowed_terminals,
            completable=completable,
            complete=complete,
            depth=1,
            cache={},
            orig_grammar=grammar,
        )

    def __init__(self, native_grammar: NativeGrammar, native_charts: NativeEarleyCharts, chart_num: int, allowed_token_names: Sequence[str], completable: bool, complete: bool, depth: int, cache: Dict[int, NativeEarleyChart], orig_grammar: SimpleBNF):
        self.native_grammar = native_grammar
        self.native_charts = native_charts
        self.chart_num = chart_num
        self._allowed_token_names = allowed_token_names
        self.children : Dict[str, NativeEarleyTrieNode] = {}
        self._completable = completable
        self._complete = complete
        self._depth = depth
        self._cache = cache
        self.orig_grammar = orig_grammar

    def get_child(self, token: Token) -> Self:
        if token.name not in self.children:
            assert not token.loose_behavior
            next_chart_num, next_allowed_token_names, completable, complete = self.native_charts.parse(self.native_grammar, self.chart_num, token.name)
            self.children[token.name] = NativeEarleyTrieNode(self.native_grammar, self.native_charts, next_chart_num, next_allowed_token_names, completable, complete, self._depth + 1, self._cache, self.orig_grammar)

        return self.children[token.name]

    def is_completable(self) -> bool:
        return self._completable

    def is_complete(self) -> bool:
        return self._complete

    @overload
    def __getitem__(self, index: int) -> LexEarleyAlgorithmChart:
        ...

    @overload
    def __getitem__(self, index: slice) -> List[LexEarleyAlgorithmChart]:
        ...

    def __getitem__(self, item):
        if isinstance(item, int):
            if item in self._cache:
                return self._cache[item]

            element = NativeEarleyChart(self.native_grammar, self.native_charts, item)
            self._cache[item] = element
            return element
        raise NotImplemented

    def __len__(self):
        return self._depth

    @property
    def allowed_token_names(self) -> Sequence[str]:
        return self._allowed_token_names