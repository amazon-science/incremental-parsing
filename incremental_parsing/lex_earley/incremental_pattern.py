import abc
from typing import Tuple, NamedTuple, FrozenSet, Optional

import regex
from lark.utils import get_regexp_width
from regex import Pattern

from incremental_parsing.regex_ixes.regex_compile import compile


class FullMatchResult(NamedTuple):
    """
    See documentation for IncrementalPattern::fullmatch and MatchResult
    """
    is_full_match: bool
    is_partial_match: bool
    is_inextensible_match: bool

    def to_partial_match_result(self, string_len: int) -> "MatchResult":
        return MatchResult(is_full_match=self.is_full_match,
                           is_partial_match=self.is_partial_match,
                           is_inextensible_match=self.is_inextensible_match,
                           match_end=string_len)


class MatchResult(NamedTuple):
    """
    See documentation for IncrementalPattern::match
    """
    is_full_match: bool  # Does the string match this lexeme
    is_partial_match: bool  # Is there some string s (including empty), such that if you concatenate this string with s,
    # it matches the lexeme
    is_inextensible_match: bool  # Is is_full_match true, and there not a non-empty string s such that this string
    # concat with s matches the lexeme
    match_end: int

    def to_full_match_result(self, string_len: int) -> "FullMatchResult":
        if self.match_end == string_len:
            return FullMatchResult(is_full_match=self.is_full_match,
                                   is_partial_match=self.is_partial_match,
                                   is_inextensible_match=self.is_inextensible_match)
        else:
            return FullMatchResult(False, False, False)


class IncrementalPattern(abc.ABC):
    """The implementations of IncrementalPatternString are fairly simple, it is worth looking through that"""
    @abc.abstractmethod
    def fullmatch(self, text: str, pos: int = 0, endpos: Optional[int] = None) -> FullMatchResult:
        """
        Does the entirety of text match this pattern
        """
        pass

    @abc.abstractmethod
    def match(self, text: str, pos: int = 0, endpos: Optional[int] = None) -> MatchResult:
        """
        Is there some prefix of text that matches this pattern? If so, what is the longest one.
        """
        pass

    @abc.abstractmethod
    def sort_order(self) -> Tuple[int, ...]:
        pass

    @abc.abstractmethod
    def is_nullable(self) -> bool:
        pass

    # The rest of this interface essentially deals with NFA states directly

    @property
    @abc.abstractmethod
    def initial_states(self) -> FrozenSet[int]:
        pass

    @property
    @abc.abstractmethod
    def final_states(self) -> FrozenSet[int]:
        pass

    @abc.abstractmethod
    def step_forwards_any(self, states: FrozenSet[int]) -> FrozenSet[int]:
        """
        What states are reachable after performing a single step forward with _any_ character?
        """
        pass

    @abc.abstractmethod
    def step_forwards(self, states: FrozenSet[int], text: str) -> FrozenSet[int]:
        pass

    @abc.abstractmethod
    def step_backwards(self, states: FrozenSet[int], text: str) -> FrozenSet[int]:
        pass

    @abc.abstractmethod
    def reachable_forward(self, states: FrozenSet[int]) -> FrozenSet[int]:
        pass

    @abc.abstractmethod
    def is_extensible(self, states: FrozenSet[int]) -> bool:
        pass


class IncrementalPatternString(IncrementalPattern):
    """
    For exact string matches, using a whole NFA is overkill. This implements a lightweight version of IncrementalPattern
    """
    def __init__(self, pattern: str):
        self.pattern = pattern

    def fullmatch(self, text: str, pos: int = 0, endpos: Optional[int] = None) -> FullMatchResult:
        endpos = endpos if endpos is not None else len(text)
        if (endpos - pos) > len(self.pattern):
            return FullMatchResult(False, False, False)

        endpoint = min(len(text), endpos, pos + len(self.pattern))
        relevant_text = text[pos:endpoint]
        if relevant_text == self.pattern:
            return FullMatchResult(True, True, True)
        elif self.pattern.startswith(relevant_text):
            return FullMatchResult(is_full_match=False, is_partial_match=True, is_inextensible_match=False)
        else:
            return FullMatchResult(False, False, False)

    def match(self, text: str, pos: int = 0, endpos: Optional[int] = None) -> MatchResult:
        endpos = endpos if endpos is not None else len(text)
        endpoint = min(len(text), endpos, pos + len(self.pattern))
        relevant_text = text[pos:endpoint]
        if relevant_text == self.pattern:
            return MatchResult(True, True, True, endpoint)
        elif self.pattern.startswith(relevant_text):
            return MatchResult(is_full_match=False, is_partial_match=True, is_inextensible_match=False,
                               match_end=endpoint)
        elif relevant_text.startswith(self.pattern):
            assert False, "This should not happen, covered by the == case"
        else:
            return MatchResult(False, False, False, -1)

    def sort_order(self) -> Tuple[int, ...]:
        return 0, len(self.pattern)

    def is_nullable(self) -> bool:
        return len(self.pattern) == 0

    @property
    def initial_states(self) -> FrozenSet[int]:
        return frozenset({0})

    @property
    def final_states(self) -> FrozenSet[int]:
        return frozenset({len(self.pattern)})

    def step_forwards(self, states: FrozenSet[int], text: str) -> FrozenSet[int]:
        result = []
        for state in states:
            if self.pattern[state:].startswith(text):
                result.append(state + len(text))

        return frozenset(result)

    def step_forwards_any(self, states: FrozenSet[int]) -> FrozenSet[int]:
        return frozenset(i + 1 for i in states if i < len(self.pattern))

    def step_backwards(self, states: FrozenSet[int], text: str) -> FrozenSet[int]:
        result = []
        for state in states:
            if self.pattern[:state].endswith(text):
                result.append(state - len(text))

        return frozenset(result)

    def reachable_forward(self, states: FrozenSet[int]) -> FrozenSet[int]:
        if len(states) == 0:
            return frozenset()
        min_state = min(states)
        return frozenset(range(min_state, len(self.pattern) + 1))

    def is_extensible(self, states: FrozenSet[int]) -> bool:
        return any(s < len(self.pattern) for s in states)


# These are BANNED in regular expressions that we use
LAZY_QUANTIFIERS = regex.compile("([\\*\\+\\?]|\\{\\d+(,(\\d+)?)?})\\?")


class IncrementalPatternRegex(IncrementalPattern):

    def __init__(self, pattern: Pattern):
        self.pattern = pattern
        if LAZY_QUANTIFIERS.search(pattern.pattern) is not None:
            raise ValueError(f"Pattern {pattern.pattern} contains a lazy quantifier")
        (self.min_pattern_width, self.max_pattern_width) = get_regexp_width(pattern.pattern)

        self.nfa_regex = compile(pattern.pattern, pattern.flags)

    def fullmatch(self, text: str, pos: int = 0, endpos: Optional[int] = None) -> FullMatchResult:
        endpos = endpos if endpos is not None else len(text)

        # noinspection PyArgumentList
        m = self.pattern.fullmatch(text, pos, endpos, partial=True)  # type:ignore[call-overload]
        if m:
            if m.partial:
                return FullMatchResult(is_full_match=False, is_partial_match=True, is_inextensible_match=False)
            elif len(text) == self.max_pattern_width:
                return FullMatchResult(True, True, True)
            else:
                return FullMatchResult(is_full_match=True, is_partial_match=True, is_inextensible_match=False)
        else:
            return FullMatchResult(False, False, False)

    def match(self, text: str, pos: int = 0, endpos: Optional[int] = None) -> MatchResult:
        endpos = endpos if endpos is not None else len(text)

        # noinspection PyArgumentList
        m = self.pattern.match(text, pos, endpos, partial=True)  # type:ignore[call-overload]
        if m:
            if m.partial:
                return MatchResult(is_full_match=False, is_partial_match=True, is_inextensible_match=False,
                                   match_end=m.end())
            elif len(text) == self.max_pattern_width:
                return MatchResult(True, True, True, match_end=m.end())
            else:
                return MatchResult(is_full_match=True, is_partial_match=True, is_inextensible_match=False,
                                   match_end=m.end())
        else:
            return MatchResult(False, False, False, match_end=-1)

    def sort_order(self) -> Tuple[int, ...]:
        return 10, self.max_pattern_width

    def is_nullable(self) -> bool:
        return self.min_pattern_width == 0

    @property
    def initial_states(self) -> FrozenSet[int]:
        return self.nfa_regex.start_states

    @property
    def final_states(self) -> FrozenSet[int]:
        return self.nfa_regex.end_states

    def step_forwards_any(self, states: FrozenSet[int]) -> FrozenSet[int]:
        return self.nfa_regex.step_forward_any(states)

    def step_forwards(self, states: FrozenSet[int], text: str) -> FrozenSet[int]:
        for char in text:
            states = self.nfa_regex.step_forward(states, char)
        return states

    def step_backwards(self, states: FrozenSet[int], text: str) -> FrozenSet[int]:
        for char in text[::-1]:
            states = self.nfa_regex.step_backward(states, char)
        return states

    def reachable_forward(self, states: FrozenSet[int]) -> FrozenSet[int]:
        return self.nfa_regex.get_reachable_forward(states)

    def is_extensible(self, states: FrozenSet[int]) -> bool:
        return self.nfa_regex.is_extensible(states)
