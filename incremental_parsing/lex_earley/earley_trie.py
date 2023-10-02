from abc import ABCMeta, abstractmethod
from typing import Dict, Tuple, Optional, overload, List, Generator

from typing_extensions import Self

from incremental_parsing.lex_earley.earley_base import process_token, charts_completable, LexEarleyAlgorithmChart
from incremental_parsing.lex_earley.lexer import Token
from incremental_parsing.lex_earley.simple_bnf import SimpleBNF
from incremental_parsing.utils.lookback_trie import LookbackTrieNode


class AbstractEarleyTrieNode(metaclass=ABCMeta):
    @abstractmethod
    def get_child(self, token: Token) -> Self:
        pass

    @abstractmethod
    def is_completable(self) -> bool:
        pass

    @abstractmethod
    def is_complete(self) -> bool:
        pass

    @overload
    def __getitem__(self, index: int) -> LexEarleyAlgorithmChart:
        ...

    @overload
    def __getitem__(self, index: slice) -> List[LexEarleyAlgorithmChart]:
        ...

    @abstractmethod
    def __getitem__(self, item):
        pass

    @property
    @abstractmethod
    def allowed_token_names(self) -> Tuple[str, ...]:
        pass


class EarleyTrieNode(AbstractEarleyTrieNode):
    """
    A simple structure to cache parse results to avoid needless computation
    """
    def __init__(self, grammar: SimpleBNF,
                 charts: LookbackTrieNode[LexEarleyAlgorithmChart],
                 allowed_token_names: Tuple[str, ...]):
        self.grammar = grammar
        self.charts = charts
        self.children: Dict[Token, EarleyTrieNode] = {}
        self._allowed_token_names = allowed_token_names  # This parameter doesn't really matter for the initial chart

    def get_child(self, token: Token) -> "EarleyTrieNode":
        if token not in self.children:
            assert not token.loose_behavior, ("Use Earley NFA for loose behavior. "
                                              "Potentially modify NFA so that it can be built incrementally?")
            next_chart, allowed_tokens = process_token(self.grammar, self.charts, token)
            self.children[token] = EarleyTrieNode(self.grammar,
                                                  self.charts.get_child(next_chart),
                                                  tuple(allowed_tokens))

        return self.children[token]

    def is_completable(self):
        return charts_completable(self.charts)

    def is_complete(self) -> bool:
        last_chart = self[-1]
        return any(state.is_complete() and state.rule_name in self.grammar.top_level_rules
                   and state.span_start == 0 for state, _ in
                   last_chart.states)

    @property
    def allowed_token_names(self) -> Tuple[str, ...]:
        return self._allowed_token_names

    def __getitem__(self, item):
        return self.charts.__getitem__(item)


class DummyEarleyTrieNode(AbstractEarleyTrieNode):
    def __init__(self,
                 parent: Optional["DummyEarleyTrieNode"],
                 this_token: Optional[Token],
                 allowed_token_names: Tuple[str, ...]):
        self.parent = parent
        self.this_token = this_token
        self._allowed_token_names = allowed_token_names
        self.children: Dict[Token, DummyEarleyTrieNode] = {}

    def get_child(self, token: Token) -> "DummyEarleyTrieNode":
        if token not in self.children:
            self.children[token] = DummyEarleyTrieNode(self, token, self.allowed_token_names)

        return self.children[token]

    def is_completable(self) -> bool:
        return True

    def is_complete(self) -> bool:
        return True

    @property
    def allowed_token_names(self) -> Tuple[str, ...]:
        return self._allowed_token_names

    def __getitem__(self, item):
        raise NotImplementedError("Dummy object")

    def get_reverse_token_sequence(self) -> Generator[Token, None, None]:
        node = self
        while True:
            if node.parent is not None:
                assert node.this_token is not None
                yield node.this_token
                node = node.parent
            else:
                break
