import abc
import unittest
from builtins import sorted
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple, NamedTuple, Any, Union, Optional, List, FrozenSet, Sequence, Callable, \
    TypeVar, Generic, DefaultDict, Set

import regex
from typing_extensions import Self

from incremental_parsing.lex_earley.branch_guard.banned_regex_set import LexerBranchGuardBannedRegexSet
from incremental_parsing.lex_earley.branch_guard.lexer_branch_guard import LexerBranchGuard
from incremental_parsing.lex_earley.branch_guard.single_char_banned_regex_set import \
    LexerBranchGuardBannedRegexSetRealBehavior
from incremental_parsing.lex_earley.incremental_pattern import IncrementalPattern, IncrementalPatternRegex


class Token(NamedTuple):
    name: str
    text: str
    loose_behavior: bool = False  # Add to the previous chart (until a fixpoint is obtained)
    max_loosiness: Optional[int] = 0


TCov = TypeVar('TCov', covariant=True)
T = TypeVar('T')
U = TypeVar('U')


@dataclass(frozen=True)
class LexResultSuccess(Generic[TCov]):
    token_sequence: Tuple[Token, ...]
    next_lexer_state: TCov


@dataclass(frozen=True)
class LexResultPartial(Generic[TCov]):
    next_lexer_state: TCov


@dataclass(frozen=True)
class LexResultFailure:
    pass


@dataclass(frozen=True)
class LexResultBranch(Generic[TCov]):
    branches: Tuple[Tuple["LexResult[TCov]", Optional[LexerBranchGuard], Any], ...]
    """
    Result, Branch guard, Branch Guard State
    (Or None for branch guard if it is not needed)
    """


LexResult = Union[LexResultSuccess[TCov], LexResultPartial[TCov], LexResultFailure, LexResultBranch[TCov]]


def map_lex_result_state(lexer_state_mapper: Callable[[T], U], result: LexResult[T]) -> LexResult[U]:
    """
    This would be a good use of generics, but I could never figure out Python generics.
    Essentially just a function ([A -> B], LexResult<A>) -> LexResult<B>
    """
    if isinstance(result, LexResultFailure):
        return result
    elif isinstance(result, LexResultSuccess):
        return LexResultSuccess(
            next_lexer_state=lexer_state_mapper(result.next_lexer_state),
            token_sequence=result.token_sequence
        )
    elif isinstance(result, LexResultPartial):
        return LexResultPartial(
            next_lexer_state=lexer_state_mapper(result.next_lexer_state)
        )
    elif isinstance(result, LexResultBranch):
        return LexResultBranch(
            branches=tuple((map_lex_result_state(lexer_state_mapper, branch), bg, bgs) for branch, bg, bgs in
                           result.branches)
        )
    else:
        raise ValueError(f'Unexpected lex result type: {type(result)}')


def map_filter_lex_result_state(f: Callable[[T], Optional[U]], result: LexResult[T]) -> LexResult[U]:
    if isinstance(result, LexResultFailure):
        return result
    elif isinstance(result, LexResultSuccess):
        res = f(result.next_lexer_state)
        if res is not None:
            return LexResultSuccess(
                next_lexer_state=res,
                token_sequence=result.token_sequence
            )
        else:
            return LexResultFailure()
    elif isinstance(result, LexResultPartial):
        res = f(result.next_lexer_state)
        if res is not None:
            return LexResultPartial(
                next_lexer_state=res
            )
        else:
            return LexResultFailure()
    elif isinstance(result, LexResultBranch):
        return LexResultBranch(
            branches=tuple((map_filter_lex_result_state(f, branch), bg, bgs) for branch, bg, bgs in
                           result.branches)
        )
    else:
        raise ValueError(f'Unexpected lex result type: {type(result)}')


class AbstractLexer(abc.ABC, Generic[T]):
    """
    The initial idea was to have an interface for an incremental lexer; i.e. it lexes purely from left to right,
    storing the current lexer state in an immutable data structure (modulo constrained mutability to reduce computation)
    However, the requirements of the lexer grew: we need to also handle the right context, and then somehow integrate
    that into the left context.
    It doesn't make sense to create completely separte lexers for the right context; this would duplicate a lot of
    code already found in the left context, and then it is unclear how to constrain generation between the
    left and right contexts.
    The solution I ended up implementing is `to_suffix_lexer_state` and `to_middle_lexer_state`.
    Instead of calling different functions for the right context, the lexer can internally decide how to handle the
    suffix and fill-in-middle fragments.
    Workflow:
    First call initialize, obtain result
    Call advance_lexer_state for each character, until reaching the end of the left context
    Call to_suffix_lexer_state, obtain a bunch of branches for the suffix lexer
    Call advance_lexer_state for each character of the right context, then end_of_file when finished
    Finally, call to_middle_lexer_state.
    Advance lexer state for each character of the fill-in-middle context.
    Call end_of_file to check if the fill-in-middle lexeme is complete
    """

    @abc.abstractmethod
    def initialize(self, initial_hint: Optional[Iterable[str]] = None) -> LexResult[T]:
        """
        Creates a new lexer state.
        Called once, at the beginning of a file
        """
        pass

    @abc.abstractmethod
    def lexer_hint(self, state: T, allowed_tokens: Iterable[str]) -> LexResult[T]:
        """
        If we have information about what tokens are allowed in the current state,
        we can use that to fail fast when encountering a token we don't expect.
        """
        pass

    @abc.abstractmethod
    def advance_lexer_state(self, lexer_state: T, char: str) -> LexResult[T]:
        pass

    @abc.abstractmethod
    def end_of_file(self, lexer_state: T) -> LexResult[T]:
        """
        Called when we reach the end of the file.
        Or, for a "middle" lexer state, called when we want to check whether the fill-in-middle context
        blends into the right context.
        This is where we should return any remaining tokens (or failure to lex).
        """
        pass

    @abc.abstractmethod
    def to_suffix_lexer_state(self, text: str, lexer_state: T) -> Tuple[Tuple[int, T], ...]:
        """
        Convert a lexer state to a (set of) suffix lexer state(s)
        A branch is created for every element of the output
        """
        pass

    @abc.abstractmethod
    def to_middle_lexer_state(self, lexer_state: T) -> Tuple[T, Optional[Tuple[str, ...]]]:
        """
        Convert a suffix lexer state to an intermediate lexer state; i.e. a lexer that allows
        elements between the prefix and the suffix
        :return: The middle state, plus the possible tokens that the middle state can end on
        (or None if there is no restriction)
        """
        pass

    @abc.abstractmethod
    def get_all_possible_token_names(self) -> Iterable[str]:
        pass


class IncrementalLexerPrefixStateWrapper(abc.ABC):
    @abc.abstractmethod
    def get_prefix_state(self) -> "IncrementalLexerPrefixState":
        pass

    @abc.abstractmethod
    def set_prefix_state(self, state: "IncrementalLexerPrefixState") -> Self:
        pass


@dataclass(frozen=True)
class IncrementalLexerPrefixState(IncrementalLexerPrefixStateWrapper):
    matched_so_far: str
    finished_tokens: Tuple[str, ...]
    possible_tokens: Tuple[Tuple[str, FrozenSet[int]], ...]
    banned_but_possible_tokens: FrozenSet[str]

    def get_prefix_state(self) -> "IncrementalLexerPrefixState":
        return self

    def set_prefix_state(self, state: "IncrementalLexerPrefixState") -> "IncrementalLexerPrefixState":
        return state


class SuffixTokenContinuationInformation(NamedTuple):
    suffix_text_in_token: str
    possible_patterns: Tuple[Tuple[str, FrozenSet[int]], ...]


@dataclass(frozen=True)
class IncrementalLexerSuffixState(IncrementalLexerPrefixStateWrapper):
    prefix_state: IncrementalLexerPrefixState
    continuation_state: Optional[SuffixTokenContinuationInformation]  # None if the suffix starts on a token boundary
    suffix_state: IncrementalLexerPrefixState

    def get_prefix_state(self) -> "IncrementalLexerPrefixState":
        return self.suffix_state

    def set_prefix_state(self, state: "IncrementalLexerPrefixState") -> "IncrementalLexerSuffixState":
        return IncrementalLexerSuffixState(
            prefix_state=self.prefix_state,
            continuation_state=self.continuation_state,
            suffix_state=state
        )


@dataclass(frozen=True)
class IncrementalLexerMiddleState(IncrementalLexerPrefixStateWrapper):
    current_state: IncrementalLexerPrefixState
    expected_continuation: SuffixTokenContinuationInformation

    def get_prefix_state(self) -> "IncrementalLexerPrefixState":
        return self.current_state

    def set_prefix_state(self, state: "IncrementalLexerPrefixState") -> "IncrementalLexerMiddleState":
        return IncrementalLexerMiddleState(
            current_state=state,
            expected_continuation=self.expected_continuation
        )


IncrementalLexerState = Union[IncrementalLexerPrefixState,
IncrementalLexerSuffixState,
IncrementalLexerMiddleState]

MS = TypeVar("MS", bound=IncrementalLexerPrefixStateWrapper)


def lift_lex_result_from_prefix_state(prev_state: MS,
                                      result: LexResult[IncrementalLexerPrefixState]) -> LexResult[MS]:
    return map_lex_result_state(lambda s: prev_state.set_prefix_state(s), result)


class IncrementalLexer(AbstractLexer[IncrementalLexerState]):
    def __init__(self, tokens: Dict[str, IncrementalPattern], use_true_leftmost_longest: bool = False):
        self.tokens = tokens
        self.tokens_in_priority_order = tuple(sorted(tokens.keys(), key=lambda tok: tokens[tok].sort_order()))

        self.nullable_tokens_in_priority_order = tuple(
            tok for tok in self.tokens_in_priority_order if self.tokens[tok].is_nullable())

        self.initial_nfa_states = tuple((tok, tokens[tok].initial_states) for tok in self.tokens_in_priority_order)

        self.use_true_leftmost_longest = use_true_leftmost_longest

    def create_lexer_state(self) -> IncrementalLexerPrefixState:
        return IncrementalLexerPrefixState(
            matched_so_far="",
            finished_tokens=self.nullable_tokens_in_priority_order,
            banned_but_possible_tokens=frozenset(),
            possible_tokens=self.initial_nfa_states
        )

    def initialize(self, initial_hint: Optional[Iterable[str]] = None) -> LexResult[IncrementalLexerState]:
        if initial_hint is None:
            return LexResultPartial(next_lexer_state=self.create_lexer_state())
        else:
            return self.lexer_hint(self.create_lexer_state(), initial_hint)

    def lexer_hint(self, state: IncrementalLexerState, allowed_tokens: Iterable[str]) -> LexResult[
        IncrementalLexerState]:

        lexer_prefix_state = state.get_prefix_state()
        return lift_lex_result_from_prefix_state(state,
                                                 self.lexer_hint_prefix_state(lexer_prefix_state, allowed_tokens))

    def lexer_hint_prefix_state(self, state: IncrementalLexerPrefixState, allowed_tokens: Iterable[str]) -> LexResult[
        IncrementalLexerPrefixState]:
        possible_token_names = frozenset(name for (name, _) in state.possible_tokens)
        allowed_tokens = frozenset(allowed_tokens)

        banned_but_possible = frozenset(possible_token_names - allowed_tokens).union(state.banned_but_possible_tokens)

        return self.process_tokens(
            possible_tokens=state.possible_tokens,
            finished_tokens=state.finished_tokens,
            banned_but_possible_tokens=banned_but_possible,
            text=state.matched_so_far,
            previous_lexer_state=state
        )

    def possible_and_finished_tokens(self, new_char: str, tokens: Iterable[Tuple[str, FrozenSet[int]]],
                                     banned_tokens: FrozenSet[str]) \
            -> Tuple[List[Tuple[str, FrozenSet[int]]], List[str], FrozenSet[str]]:
        finished_tokens = []
        possible_tokens = []
        possible_token_names = []
        for token, nfa_states in tokens:
            pattern = self.tokens[token]
            next_nfa_states = pattern.step_forwards(nfa_states, new_char)
            if next_nfa_states:
                possible_tokens.append((token, next_nfa_states))
                possible_token_names.append(token)

                if any(state in pattern.final_states for state in next_nfa_states):
                    finished_tokens.append(token)

        banned_tokens = frozenset(banned_tokens.intersection(possible_token_names))
        return possible_tokens, finished_tokens, banned_tokens

    def highest_priority_token(self, text: str, possible_token_names: Sequence[str]) -> Token:
        # As long as create_lexer state returns tokens in order of priority, we maintain this ordering during all
        # operations and can just return the first token.
        return Token(possible_token_names[0], text)

    def process_trivial_tokens(self, text: str, possible_tokens: Sequence[Tuple[str, FrozenSet[int]]],
                               finished_tokens: Sequence[str], banned_but_possible_tokens: FrozenSet[str]) -> Optional[
        Token]:
        """
        The purpose of this is so that we can fail fast in any lex wrappers when encountering a token that shouldn't be
        there, but for some reason hasn't been covered by lexer_hint. For example, if we close one too many parentheses.
        """
        if len(possible_tokens) == 1 and len(finished_tokens) == 1 and len(banned_but_possible_tokens) == 0:
            this_token, states = possible_tokens[0]
            is_inextensible_match = len(self.tokens[this_token].step_forwards_any(states)) == 0
            if is_inextensible_match:
                return Token(this_token, text)

        return None

    def process_tokens(self, text: str,
                       possible_tokens: Sequence[Tuple[str, FrozenSet[int]]],
                       finished_tokens: Sequence[str],
                       banned_but_possible_tokens: FrozenSet[str],
                       previous_lexer_state: IncrementalLexerPrefixState) -> LexResult[
        IncrementalLexerPrefixState]:

        possible_token_names = (name for (name, _) in possible_tokens)

        if self.use_true_leftmost_longest:
            if banned_but_possible_tokens.issuperset(possible_token_names):  # No allowed completion of the token
                return LexResultFailure()

            # The result to use for not matching the current text, but it is possible to match future text
            partial_result = LexResultPartial(next_lexer_state=IncrementalLexerPrefixState(
                finished_tokens=tuple(finished_tokens),
                possible_tokens=tuple(possible_tokens),
                banned_but_possible_tokens=banned_but_possible_tokens,
                matched_so_far=text
            ))

            if len(finished_tokens) == 0:
                return partial_result

            # Check if we have a trivial match
            # I.E. It is inextensible, and there are no possible longer matches.
            # This part isn't actually necessary, but is a nice optimization
            trivial_token = self.process_trivial_tokens(text, possible_tokens, finished_tokens,
                                                        banned_but_possible_tokens)
            if trivial_token:
                return LexResultSuccess(token_sequence=(trivial_token,),
                                        next_lexer_state=self.create_lexer_state())

            # There is no trivial token; look at the highest-priority token, but keep in mind that there
            # is more than one token available
            next_highest_priority_token = self.highest_priority_token(text, finished_tokens)
            if next_highest_priority_token.name in banned_but_possible_tokens:
                # There is some completion of the token if we add more text, but it ain't this one
                # Example: The word "and" when we only expect an identifier.
                # The highest-priortiy token will be "and", which is disallowed,
                # but the next letter could be a letter, which makes it a valid identifier
                return partial_result

            # If we get here, there is some complete token, and that token is allowed
            # However, we must consider the possibility that a longer match will come in the future.
            # Therefore, the lexer splits into two branches: one branch where the complete token is _the_ token,
            # and we add a guard to discard this branch if there is a longer match.
            # The second branch considers the case where there_is_ a longer match;
            # it does not need a branch guard because
            # the lex will fail if no such match materializes
            branches: List[
                Tuple[LexResult[IncrementalLexerPrefixState], Optional[LexerBranchGuard], Optional[
                    Tuple[Tuple[FrozenSet[int], ...], int]
                ]]] = []

            # Branch where complete token is _the_ token
            shorter_token_branch_result = LexResultSuccess(token_sequence=(next_highest_priority_token,),
                                                           next_lexer_state=self.create_lexer_state())

            # guard to fail for a match with length > len(next_str)
            next_token_patterns = tuple(self.tokens[pat] for pat, _ in possible_tokens)
            shorter_token_branch_guard = LexerBranchGuardBannedRegexSet(next_token_patterns, len(text) + 1)
            shorter_token_branch_guard_state = tuple(nfa_state for _, nfa_state in possible_tokens), len(text)
            branches.append((shorter_token_branch_result, shorter_token_branch_guard, shorter_token_branch_guard_state))

            # Case where there is a longer match
            branches.append((partial_result, None, None))

            return LexResultBranch(tuple(branches))
        else:
            if len(possible_tokens) == 0:
                # We just went past the end of a token
                # Look at the previous lexer state and see if there was a valid match there
                if previous_lexer_state.matched_so_far == '' or len(previous_lexer_state.finished_tokens) == 0:
                    return LexResultFailure()

                previous_highest_priority_token = self.highest_priority_token(previous_lexer_state.matched_so_far,
                                                                              previous_lexer_state.finished_tokens)
                if previous_highest_priority_token.name in previous_lexer_state.banned_but_possible_tokens:
                    return LexResultFailure()
                else:
                    next_possible, next_finished, next_banned = self.possible_and_finished_tokens(
                        new_char=text[-1],
                        tokens=self.initial_nfa_states,
                        banned_tokens=frozenset(),
                    )
                    next_state = IncrementalLexerPrefixState(
                        finished_tokens=tuple(next_finished),
                        possible_tokens=tuple(next_possible),
                        banned_but_possible_tokens=next_banned,
                        matched_so_far=text[-1]
                    )
                    return LexResultSuccess(token_sequence=(previous_highest_priority_token,),
                                            next_lexer_state=next_state)
            else:
                # There are still possible longer tokens
                if banned_but_possible_tokens.issuperset(possible_token_names):
                    return LexResultFailure()
                else:
                    return LexResultPartial(next_lexer_state=IncrementalLexerPrefixState(
                        finished_tokens=tuple(finished_tokens),
                        possible_tokens=tuple(possible_tokens),
                        banned_but_possible_tokens=banned_but_possible_tokens,
                        matched_so_far=text
                    ))

    def advance_lexer_state(self, state: IncrementalLexerState, char: str) -> LexResult[IncrementalLexerState]:
        prefix_state = state.get_prefix_state()
        return lift_lex_result_from_prefix_state(state, self.advance_lexer_state_prefix(prefix_state, char))

    def advance_lexer_state_prefix(self, state: IncrementalLexerPrefixState, char: str) -> LexResult[
        IncrementalLexerPrefixState]:
        next_str = state.matched_so_far + char
        next_possible_tokens, next_finished_tokens, next_banned_tokens = \
            self.possible_and_finished_tokens(
                new_char=char,
                tokens=state.possible_tokens,
                banned_tokens=state.banned_but_possible_tokens
            )

        return self.process_tokens(text=next_str,
                                   possible_tokens=next_possible_tokens,
                                   finished_tokens=next_finished_tokens,
                                   banned_but_possible_tokens=next_banned_tokens,
                                   previous_lexer_state=state)

    def to_suffix_lexer_state(self, text: str, lexer_state: IncrementalLexerState) \
            -> Tuple[Tuple[int, IncrementalLexerSuffixState], ...]:
        assert isinstance(lexer_state, IncrementalLexerPrefixState)

        index_to_stoppable_tokens: DefaultDict[int, List[Tuple[str, FrozenSet[int]]]] = defaultdict(list)

        for tok in self.tokens_in_priority_order:
            pattern = self.tokens[tok]
            stoppable_indices = get_stoppable_indices(pattern, text)
            for index, lhs_to_rhs_final in stoppable_indices:
                index_to_stoppable_tokens[index].append((tok, lhs_to_rhs_final))

        # The results always includes a branch for if it starts on a token boundary
        results: List[Tuple[int, IncrementalLexerSuffixState]] = [
            (0, IncrementalLexerSuffixState(
                prefix_state=lexer_state,
                suffix_state=self.create_lexer_state(),
                continuation_state=None
            ))
        ]

        # For every index with a stoppable token, create a branch that skips that many characters but sets up
        # the continuation information so that the middle context can later use it
        for index, stoppable_tokens in index_to_stoppable_tokens.items():
            branch = IncrementalLexerSuffixState(
                prefix_state=lexer_state,
                suffix_state=self.create_lexer_state(),
                continuation_state=SuffixTokenContinuationInformation(
                    suffix_text_in_token=text[:index],
                    possible_patterns=tuple(stoppable_tokens)
                )
            )
            results.append((index, branch))

        return tuple(results)

    def to_middle_lexer_state(self, lexer_state: IncrementalLexerState) -> Tuple[IncrementalLexerState,
    Optional[Tuple[str, ...]]]:
        """
        :return: The middle state, plus the possible tokens that the middle state can end on
        (or None if there is no restriction)
        """
        assert isinstance(lexer_state, IncrementalLexerSuffixState)
        if lexer_state.continuation_state is None:
            # Suffix had begun on a lexeme boundary
            return lexer_state.prefix_state, None
        else:
            # Suffix began in the middle of a lexeme
            return IncrementalLexerMiddleState(
                current_state=lexer_state.prefix_state,
                expected_continuation=lexer_state.continuation_state
            ), tuple(name for (name, _) in lexer_state.continuation_state.possible_patterns)

    def end_of_file(self, state: IncrementalLexerState) -> LexResult[IncrementalLexerState]:
        if isinstance(state, IncrementalLexerPrefixState):
            return self.end_of_file_prefix(state)
        elif isinstance(state, IncrementalLexerMiddleState):
            current_possible = {tok: nfa for (tok, nfa) in state.current_state.possible_tokens}
            for token_name, final_automaton_states in state.expected_continuation.possible_patterns:
                if token_name in current_possible and \
                        any(current_state in final_automaton_states for current_state in current_possible[token_name]):
                    # The current generation connects to the right context
                    # Emit a token (but add a branch guard to make sure that no longer match is present)
                    branch_guard: LexerBranchGuard
                    if self.use_true_leftmost_longest:
                        branch_guard = LexerBranchGuardBannedRegexSet(
                            tuple(self.tokens[tok] for tok, _ in state.current_state.possible_tokens),
                            min_banned_match_length=(len(state.current_state.matched_so_far)
                                                     + len(state.expected_continuation.suffix_text_in_token)
                                                     + 1))
                        branch_guard_state = (tuple(nfa for _, nfa in state.current_state.possible_tokens),
                                              len(state.current_state.matched_so_far))
                    else:
                        branch_guard = LexerBranchGuardBannedRegexSetRealBehavior(
                            tuple(self.tokens[tok] for tok, _ in state.current_state.possible_tokens),
                            min_banned_match_length=(len(state.current_state.matched_so_far)
                                                     + len(state.expected_continuation.suffix_text_in_token)
                                                     + 1))
                        branch_guard_state = (tuple(nfa for _, nfa in state.current_state.possible_tokens),
                                              len(state.current_state.matched_so_far))

                    return LexResultBranch(((LexResultSuccess(
                        token_sequence=(Token(token_name,
                                              state.current_state.matched_so_far + state.expected_continuation.suffix_text_in_token),),
                        next_lexer_state=self.create_lexer_state()
                    ), branch_guard, branch_guard_state),))

            return LexResultFailure()  # There was no token which connects to this
        elif isinstance(state, IncrementalLexerSuffixState):
            return lift_lex_result_from_prefix_state(
                prev_state=state,
                result=self.end_of_file_prefix(state.suffix_state),
            )

    def end_of_file_prefix(self, state: IncrementalLexerPrefixState) -> LexResult[IncrementalLexerPrefixState]:
        if state.matched_so_far == "":
            return LexResultSuccess((), state)

        if self.use_true_leftmost_longest:
            # If there was a valid token, it would have already been part of a LexResultSuccess
            # (possibly in another branch)
            # Otherwise, we don't want to allow a partial token
            return LexResultFailure()
        else:
            # Look at the current lexer state and see if there is a valid match
            if len(state.finished_tokens) == 0:
                # There have been characters seen since the last match, but there is no finished match
                return LexResultFailure()

            highest_priority_token = self.highest_priority_token(state.matched_so_far,
                                                                 state.finished_tokens)
            if highest_priority_token.name in state.banned_but_possible_tokens:
                return LexResultFailure()
            else:
                # There is a valid in-progress match. However, we need to add a branch guard to prevent longer matches
                next_token_patterns = tuple(self.tokens[pat] for pat, _ in state.possible_tokens)
                shorter_token_branch_guard = LexerBranchGuardBannedRegexSetRealBehavior(next_token_patterns,
                                                                                        len(state.matched_so_far) + 1)
                branch_guard_state = tuple(nfa_states for _, nfa_states in state.possible_tokens), \
                    len(state.matched_so_far)
                return LexResultBranch(((LexResultSuccess(token_sequence=(highest_priority_token,),
                                                          next_lexer_state=self.create_lexer_state()),
                                         shorter_token_branch_guard,
                                         branch_guard_state),))

    def get_all_possible_token_names(self) -> Iterable[str]:
        return self.tokens_in_priority_order


def get_stoppable_indices(pattern: IncrementalPattern, text: str) -> List[Tuple[int, FrozenSet[int]]]:
    initial_states_plus_one = pattern.step_forwards_any(pattern.initial_states)
    reachable_initial_states = pattern.reachable_forward(initial_states_plus_one)

    stoppable_indices: List[Tuple[int, FrozenSet[int]]] = []

    current_rhs_reachable = reachable_initial_states

    for char_idx in range(len(text)):  # Faster than direct iteration over text when we break early
        char = text[char_idx]
        current_rhs_reachable = pattern.step_forwards(current_rhs_reachable, char)
        if not current_rhs_reachable:
            break

        final_states_rhs_reachable = current_rhs_reachable.intersection(pattern.final_states)

        if final_states_rhs_reachable:
            stoppable_indices.append((char_idx + 1, final_states_rhs_reachable))

    # Now that we know all the places this pattern can stop, we need to calculate the connecting
    # states from lhs their end index
    if stoppable_indices:
        current_idx, final_states = stoppable_indices[-1]
        current_idx_to_end_idx = [(current_idx, final_states)]

        for prev_index, prev_final_states in reversed(stoppable_indices[:-1]):
            prev_idx_to_end_idx = [(end_idx, pattern.step_backwards(current_to_end, text[prev_index:current_idx]))
                                   for (end_idx, current_to_end) in current_idx_to_end_idx]
            prev_idx_to_end_idx.append((prev_index, prev_final_states))

            current_idx_to_end_idx = remove_subsumed(prev_idx_to_end_idx)
            current_idx = prev_index

        lhs_to_end_idx = [(end_idx, pattern.step_backwards(current_to_end, text[:current_idx]))
                          for (end_idx, current_to_end) in current_idx_to_end_idx]

        return remove_subsumed(lhs_to_end_idx)
    else:
        return []


def remove_subsumed(current_to_end_idx: Sequence[Tuple[int, FrozenSet[int]]]) -> List[Tuple[int, FrozenSet[int]]]:
    """
    Assumption: current_to_end_idx is sorted last-index-first
    """
    states_seen_later: Set[int] = set()
    non_subsumed_values = []
    for end_idx, active_states in current_to_end_idx:
        active_states_not_seen_later = active_states.difference(states_seen_later)
        if not active_states_not_seen_later:
            continue
        else:
            states_seen_later.update(active_states_not_seen_later)
            non_subsumed_values.append((end_idx, active_states_not_seen_later))

    return non_subsumed_values


class TestStoppableIndices(unittest.TestCase):
    def test_string(self):
        STRING_REGEX = '([ubf]?r?|r[ubf])(?:"(?:[^\\\\\\n"]|\\\\.)*"|\'(?:[^\\n\\\\\']|\\\\.)*\')'
        string_pattern = IncrementalPatternRegex(regex.compile(STRING_REGEX, flags=regex.I | regex.S))

        res = get_stoppable_indices(string_pattern, '"asdf\'jkl;"qwerty')
        self.assertEqual([i for (i, _) in res], [11, 6, 1])

        res = get_stoppable_indices(string_pattern, 'l"asdf\'jkl;"qwerty')
        self.assertEqual([i for (i, _) in res], [7, 2])

        res = get_stoppable_indices(string_pattern, '\\"asdf\'jkl;"qwerty')
        self.assertEqual([i for (i, _) in res], [12, 7, 2])  # previous string could still end in "\"

        res = get_stoppable_indices(string_pattern, 'l\\"asdf\'jkl;"qwerty')
        self.assertEqual([i for (i, _) in res], [13, 8])

    def test_comment(self):
        string_pattern = IncrementalPatternRegex(regex.compile("#[^\\n]*", flags=regex.I | regex.S))

        res = get_stoppable_indices(string_pattern, 'foobar')
        self.assertEqual([i for (i, _) in res], [6])

        res = get_stoppable_indices(string_pattern, 'foo\nbar')
        self.assertEqual([i for (i, _) in res], [3])

        res = get_stoppable_indices(string_pattern, 'f' * 100000)
        self.assertEqual([i for (i, _) in res], [100000])

        res = get_stoppable_indices(string_pattern, 'f\n' * 10000000)
        self.assertEqual([i for (i, _) in res], [1])
