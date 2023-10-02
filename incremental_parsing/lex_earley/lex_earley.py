import abc
import itertools
from dataclasses import dataclass
from typing import NamedTuple, Tuple, List, Any, Optional, Iterable, TypeVar, Union, Sequence, Dict

from typing_extensions import Self

from incremental_parsing.lex_earley.branch_guard.combined_guard import LexerBranchGuardCombined
from incremental_parsing.lex_earley.branch_guard.lexer_branch_guard import LexerBranchGuard
from incremental_parsing.lex_earley.earley_base import LexEarleyState, LexEarleyAlgorithmChart, \
    TopLevel, initial_chart_allowed_tokens
from incremental_parsing.lex_earley.earley_nfa import token_branches_to_nfa, EarleyNFA
from incremental_parsing.lex_earley.earley_trie import EarleyTrieNode, DummyEarleyTrieNode, AbstractEarleyTrieNode
from incremental_parsing.lex_earley.lexer import AbstractLexer, LexResult, LexResultFailure, LexResultPartial, \
    LexResultBranch, LexResultSuccess, Token
from incremental_parsing.lex_earley.middle_earley import create_middle_bnf, create_parse_hierarchy
from incremental_parsing.lex_earley.simple_bnf import SimpleBNF
from incremental_parsing.utils.lookback_trie import LookbackTrieNode


class ParserBranchStateWrapper(abc.ABC):
    """
    Sort of a monad situation going on here, though not quite as general
    """

    @abc.abstractmethod
    def get_parser_state(self) -> "EarleyParserBranchState":
        pass

    @abc.abstractmethod
    def set_parser_state(self, state: "EarleyParserBranchState") -> Self:
        pass


class LexerParserBranchStateWrapper(abc.ABC):
    """
    Sort of a monad situation going on here, though not quite as general
    """

    @abc.abstractmethod
    def get_lexer_parser_state(self) -> "EarleyLexerParserBranchState":
        pass

    @abc.abstractmethod
    def set_lexer_parser_state(self, state: "EarleyLexerParserBranchState") -> Self:
        pass


@dataclass(frozen=True)
class EarleyParserBranchState(ParserBranchStateWrapper):
    earley_trie: AbstractEarleyTrieNode
    branch_guard: Optional[LexerBranchGuard]
    branch_guard_state: Optional[Any]

    def get_parser_state(self) -> "EarleyParserBranchState":
        return self

    def set_parser_state(self, state: "EarleyParserBranchState") -> "EarleyParserBranchState":
        return state


@dataclass(frozen=True)
class EarleyLexerParserBranchState(ParserBranchStateWrapper, LexerParserBranchStateWrapper):
    parser_state: EarleyParserBranchState
    lexer_state: Any

    def get_parser_state(self) -> "EarleyParserBranchState":
        return self.parser_state

    def set_parser_state(self, state: "EarleyParserBranchState") -> "EarleyLexerParserBranchState":
        return EarleyLexerParserBranchState(parser_state=state, lexer_state=self.lexer_state)

    def get_lexer_parser_state(self) -> "EarleyLexerParserBranchState":
        return self

    def set_lexer_parser_state(self, state: "EarleyLexerParserBranchState") -> "EarleyLexerParserBranchState":
        return state


@dataclass(frozen=True)
class LexEarleySuffixBranchState(ParserBranchStateWrapper, LexerParserBranchStateWrapper):
    prefix_state: EarleyParserBranchState
    suffix_state: EarleyLexerParserBranchState
    skip_n_chars: int

    def get_parser_state(self) -> "EarleyParserBranchState":
        return self.suffix_state.parser_state

    def set_parser_state(self, state: "EarleyParserBranchState") -> "LexEarleySuffixBranchState":
        return LexEarleySuffixBranchState(
            prefix_state=self.prefix_state,
            suffix_state=EarleyLexerParserBranchState(parser_state=state,
                                                      lexer_state=self.suffix_state.lexer_state),
            skip_n_chars=self.skip_n_chars
        )

    def get_lexer_parser_state(self) -> "EarleyLexerParserBranchState":
        return self.suffix_state

    def set_lexer_parser_state(self, state: "EarleyLexerParserBranchState") -> "LexEarleySuffixBranchState":
        return LexEarleySuffixBranchState(
            prefix_state=self.prefix_state,
            suffix_state=state,
            skip_n_chars=self.skip_n_chars
        )


@dataclass(frozen=True)
class LexEarleyMiddleBranchState(ParserBranchStateWrapper, LexerParserBranchStateWrapper):
    middle_grammar: SimpleBNF  # There is a customized grammar created from the prefix and suffix
    middle_state: EarleyLexerParserBranchState

    # The lexer is assumed to keep track of if a partial token fits in with the suffix

    def get_parser_state(self) -> "EarleyParserBranchState":
        return self.middle_state.parser_state

    def set_parser_state(self, state: "EarleyParserBranchState") -> "LexEarleyMiddleBranchState":
        return LexEarleyMiddleBranchState(
            middle_grammar=self.middle_grammar,
            middle_state=EarleyLexerParserBranchState(parser_state=state,
                                                      lexer_state=self.middle_state.lexer_state),
        )

    def get_lexer_parser_state(self) -> "EarleyLexerParserBranchState":
        return self.middle_state

    def set_lexer_parser_state(self, state: "EarleyLexerParserBranchState") -> "LexEarleyMiddleBranchState":
        return LexEarleyMiddleBranchState(
            middle_grammar=self.middle_grammar,
            middle_state=state,
        )


class LexEarleyAlgorithmPrefixState(NamedTuple):
    branches: Tuple[EarleyLexerParserBranchState, ...]
    lookahead_str: str
    lookahead_str_idx: int
    seen_eof: bool


class LexEarleyAlgorithmSuffixState(NamedTuple):
    branches: Tuple[LexEarleySuffixBranchState, ...]
    lookahead_str: str
    lookahead_str_idx: int
    parsed_in_suffix_so_far: LookbackTrieNode[
        str]  # So that we can use branch lookahead after done parsing the middle state
    seen_eof: bool


class LexEarleyAlgorithmMiddleState(NamedTuple):
    branches: Tuple[LexEarleyMiddleBranchState, ...]
    lookahead_str: str
    lookahead_str_idx: int
    suffix_lookahead: str  # To use for branch lookahead when calling is_complete
    seen_eof: bool


LexEarleyAlgorithmState = Union[
    LexEarleyAlgorithmPrefixState,
    LexEarleyAlgorithmSuffixState,
    LexEarleyAlgorithmMiddleState
]


class LexEarleyAlgorithmContext(NamedTuple):
    grammar: SimpleBNF
    lexer: AbstractLexer


def get_all_possible_earley_states(bnf: SimpleBNF, origin_chart_idx: int) -> List[LexEarleyState]:
    rules = []
    for rule_name in bnf.reachable_rules:
        for production_idx, production in enumerate(bnf.rules[rule_name].productions):
            for element_idx, _ in enumerate(production.elements):
                rules.append(LexEarleyState(
                    rule_name=rule_name,
                    production_index=production_idx,
                    max_position=len(production.elements),
                    span_start=origin_chart_idx,
                    position=element_idx
                ))
            rules.append(LexEarleyState(
                rule_name=rule_name,
                production_index=production_idx,
                max_position=len(production.elements),
                span_start=origin_chart_idx,
                position=len(production.elements)
            ))

    return rules


def process_lex_result(context: LexEarleyAlgorithmContext,
                       parent_branch_state: EarleyParserBranchState,
                       lex_result: LexResult, lookahead: str,
                       lookahead_idx: int) -> List[
    Tuple[EarleyLexerParserBranchState, Tuple[Token, ...]]]:
    """
    Processes the result of a lexing operation, creating new branches as necessary
    ASSUMPTION: parent_branch_guard_state is already up-to-date with the most lookahead information.
    lookahead is only to catch up new branches
    """

    if isinstance(lex_result, LexResultBranch):
        # The lexer has branched; create a parser branch for each lexer branch
        # Most of this is kinda boilerplate to fast-forward any new branch guards to the current lookahead,
        # and then to consolidate them so that we don't have ridiculously deep recursion within CombinedBranchGuards
        results = []

        for (branch_lex_result, branch_lex_guard, branch_lex_guard_state) in lex_result.branches:
            if branch_lex_guard is not None:
                branch_res, branch_lex_guard_state = branch_lex_guard.branch_allowed(lookahead, lookahead_idx,
                                                                                     branch_lex_guard_state)
                branch_lex_guard, branch_lex_guard_state = branch_lex_guard.replace(branch_lex_guard_state)
                if branch_res or (branch_lex_guard is None):
                    # With lookahead, the branch we are examining is definitely OK.
                    # Therefore, we don't need to actually include the condition; just inherit from the parent
                    branch_lex_guard = parent_branch_state.branch_guard
                    branch_lex_guard_state = parent_branch_state.branch_guard_state
                elif branch_res is None:
                    # The current branch is interesting. We should consider both the parent's branch condition
                    # and the new branch condition
                    if parent_branch_state.branch_guard is None:
                        # Though the parent doesn't actually have a branch condition
                        pass
                    else:
                        branch_lex_guard = LexerBranchGuardCombined(
                            (parent_branch_state.branch_guard, branch_lex_guard)
                        )
                        branch_lex_guard_state = LexerBranchGuardCombined.combine_branch_guard_states(
                            (parent_branch_state.branch_guard_state, branch_lex_guard_state))
                else:
                    # The current branch is definitely _not_ OK.
                    # Don't process it.
                    continue
            else:
                # We aren't adding any new conditions to this branch, just inherit the branch guard
                branch_lex_guard = parent_branch_state.branch_guard
                branch_lex_guard_state = parent_branch_state.branch_guard_state

            new_parent_branch = EarleyParserBranchState(
                branch_guard=branch_lex_guard,
                branch_guard_state=branch_lex_guard_state,
                earley_trie=parent_branch_state.earley_trie,
            )

            results.extend(process_lex_result(context=context, parent_branch_state=new_parent_branch,
                                              lex_result=branch_lex_result, lookahead=lookahead,
                                              lookahead_idx=lookahead_idx))

        return results
    elif isinstance(lex_result, LexResultFailure):
        return []
    elif isinstance(lex_result, LexResultPartial):
        # The lexer has consumed the character without yielding any tokens.
        # Just update the lexer state, and continue the current branch
        return [(EarleyLexerParserBranchState(parser_state=parent_branch_state,
                                              lexer_state=lex_result.next_lexer_state), ())]
    else:
        assert isinstance(lex_result, LexResultSuccess)

        if not lex_result.token_sequence:
            return [(EarleyLexerParserBranchState(parser_state=parent_branch_state,
                                                  lexer_state=lex_result.next_lexer_state), ())]

        # The lexer has given us a token (or more than one maybe)
        current_earley_trie = parent_branch_state.earley_trie
        next_tokens = None
        for i, token in enumerate(lex_result.token_sequence):
            # Use the classic Earley algorithm with these new tokens
            current_earley_trie = current_earley_trie.get_child(token)
            next_tokens = current_earley_trie.allowed_token_names

        if current_earley_trie.is_completable():
            assert next_tokens is not None  # For mypy; Guaranteed because lex_result.token_sequence is truthy

            # There is some state in the last chart which is completable
            next_lexer_result = context.lexer.lexer_hint(state=lex_result.next_lexer_state,
                                                         allowed_tokens=next_tokens)

            next_branch_state = EarleyParserBranchState(
                branch_guard=parent_branch_state.branch_guard,
                branch_guard_state=parent_branch_state.branch_guard_state,
                earley_trie=current_earley_trie,
            )

            result_of_hint = process_lex_result(context=context, parent_branch_state=next_branch_state,
                                                lex_result=next_lexer_result, lookahead=lookahead,
                                                lookahead_idx=lookahead_idx)
            return [(branch_state, lex_result.token_sequence + hint_tokens)
                    for branch_state, hint_tokens in result_of_hint]
        else:
            # The last chart state is not completable, terminate the current branch
            return []


PBT = TypeVar('PBT', bound=ParserBranchStateWrapper)
LPBT = TypeVar('LPBT', bound=LexerParserBranchStateWrapper)


def advance_branch_guards(branches: Iterable[PBT], new_lookahead: str, new_lookahead_idx: int) -> Iterable[PBT]:
    for branch in branches:
        parser_branch = branch.get_parser_state()
        if parser_branch.branch_guard is None:
            yield branch
        else:
            lookahead_res, guard_state = parser_branch.branch_guard.branch_allowed(text=new_lookahead,
                                                                                   start_index=new_lookahead_idx,
                                                                                   branch_guard_state=
                                                                                   parser_branch.branch_guard_state)
            branch_guard, guard_state = parser_branch.branch_guard.replace(guard_state)
            if lookahead_res:
                # Succeeds check, no need to check anymore
                next_guard = None
                next_guard_state = None
            elif lookahead_res is None:
                # Still may potentially fail the check
                next_guard = branch_guard
                next_guard_state = guard_state
            else:
                # Lookahead failed, don't even consider this branch
                continue

            yield branch.set_parser_state(EarleyParserBranchState(
                branch_guard=next_guard,
                branch_guard_state=next_guard_state,
                earley_trie=parser_branch.earley_trie,
            ))


def catch_up_branches(branches: Sequence[PBT], lookahead: str,
                      lookahead_idx: int, char: str) -> Tuple[Sequence[PBT], str, int]:
    if lookahead_idx < len(lookahead):
        assert lookahead[lookahead_idx] == char, f"Next character {char} does not match lookahead string {lookahead}"
        if lookahead_idx + 1 == len(lookahead):  # We just saw the last character of the lookahead
            return branches, "", 0
        else:  # We are in other parts of the lookahead
            return branches, lookahead, lookahead_idx + 1
    else:  # We are already past the end of the lookahead, any new characters are never seen before
        caught_up_branches = list(
            advance_branch_guards(branches=branches, new_lookahead=char, new_lookahead_idx=lookahead_idx))
        return caught_up_branches, "", 0


def lex_earley_step_prefix(context: LexEarleyAlgorithmContext, state: LexEarleyAlgorithmPrefixState,
                           char: str) -> LexEarleyAlgorithmPrefixState:
    all_results: List[EarleyLexerParserBranchState] = []
    caught_up_branches, remaining_lookahead, lookahead_idx = catch_up_branches(branches=state.branches,
                                                                               lookahead=state.lookahead_str,
                                                                               lookahead_idx=state.lookahead_str_idx,
                                                                               char=char)

    for branch in caught_up_branches:
        lex_result = context.lexer.advance_lexer_state(branch.lexer_state, char)
        all_results.extend(b for b, _ in process_lex_result(context=context,
                                                            parent_branch_state=branch.parser_state,
                                                            lex_result=lex_result,
                                                            lookahead=remaining_lookahead,
                                                            lookahead_idx=lookahead_idx))

    return LexEarleyAlgorithmPrefixState(
        branches=tuple(all_results),
        lookahead_str=remaining_lookahead,
        lookahead_str_idx=lookahead_idx,
        seen_eof=False
    )


def lex_earley_step_suffix(context: LexEarleyAlgorithmContext, state: LexEarleyAlgorithmSuffixState,
                           char: str) -> LexEarleyAlgorithmSuffixState:
    all_results = []
    caught_up_branches, remaining_lookahead, lookahead_idx = catch_up_branches(branches=state.branches,
                                                                               lookahead=state.lookahead_str,
                                                                               lookahead_idx=state.lookahead_str_idx,
                                                                               char=char)
    for branch in caught_up_branches:
        if branch.skip_n_chars == 0:
            lex_result = context.lexer.advance_lexer_state(branch.suffix_state.lexer_state, char)

            results = process_lex_result(context=context,
                                         parent_branch_state=branch.suffix_state.parser_state,
                                         lex_result=lex_result,
                                         lookahead=remaining_lookahead,
                                         lookahead_idx=lookahead_idx)

            for result_suffix_state, result_tokens in results:
                all_results.append(LexEarleySuffixBranchState(
                    prefix_state=branch.prefix_state,
                    suffix_state=result_suffix_state,
                    skip_n_chars=0
                ))
        else:
            all_results.append(LexEarleySuffixBranchState(
                prefix_state=branch.prefix_state,
                suffix_state=branch.suffix_state,
                skip_n_chars=branch.skip_n_chars - 1
            ))

    return LexEarleyAlgorithmSuffixState(
        branches=tuple(all_results),
        lookahead_str=remaining_lookahead,
        lookahead_str_idx=lookahead_idx,
        parsed_in_suffix_so_far=state.parsed_in_suffix_so_far.get_child(char),
        seen_eof=False
    )


def lex_earley_step_middle(context: LexEarleyAlgorithmContext, state: LexEarleyAlgorithmMiddleState,
                           char: str) -> LexEarleyAlgorithmMiddleState:
    all_results: List[LexEarleyMiddleBranchState] = []
    caught_up_branches, remaining_lookahead, lookahead_idx = catch_up_branches(branches=state.branches,
                                                                               lookahead=state.lookahead_str,
                                                                               lookahead_idx=state.lookahead_str_idx,
                                                                               char=char)

    for branch in caught_up_branches:
        lex_result = context.lexer.advance_lexer_state(branch.middle_state.lexer_state, char)

        middle_context = LexEarleyAlgorithmContext(
            grammar=branch.middle_grammar,
            lexer=context.lexer
        )

        results = process_lex_result(context=middle_context,
                                     parent_branch_state=branch.middle_state.parser_state,
                                     lex_result=lex_result,
                                     lookahead=remaining_lookahead,
                                     lookahead_idx=lookahead_idx)

        all_results.extend(branch.set_lexer_parser_state(state=result[0]) for result in results)

    return LexEarleyAlgorithmMiddleState(
        branches=tuple(all_results),
        lookahead_str=remaining_lookahead,
        suffix_lookahead=state.suffix_lookahead,
        lookahead_str_idx=lookahead_idx,
        seen_eof=False
    )


def lex_earley_step(context: LexEarleyAlgorithmContext, state: LexEarleyAlgorithmState,
                    char: str) -> LexEarleyAlgorithmState:
    """
    Performs a single step of the LexEarley algorithm.
    """
    assert not state.seen_eof, "State has already seen EOF"

    if isinstance(state, LexEarleyAlgorithmPrefixState):
        return lex_earley_step_prefix(context=context, state=state, char=char)
    elif isinstance(state, LexEarleyAlgorithmSuffixState):
        return lex_earley_step_suffix(context=context, state=state, char=char)
    else:
        return lex_earley_step_middle(context=context, state=state, char=char)


def lex_earley_init(context: LexEarleyAlgorithmContext, lookahead: str = "") -> LexEarleyAlgorithmPrefixState:
    """
    Initializes the LexEarley algorithm.
    """
    initial_chart, initial_allowed_tokens = initial_chart_allowed_tokens(context.grammar)
    root_node: LookbackTrieNode[LexEarleyAlgorithmChart] = LookbackTrieNode.create_root_node()
    initial_earley_trie = EarleyTrieNode(grammar=context.grammar,
                                         charts=root_node.get_child(initial_chart),
                                         allowed_token_names=tuple(initial_allowed_tokens))

    initial_lex_result = context.lexer.initialize(initial_hint=initial_allowed_tokens)
    initial_branches_results = process_lex_result(
        parent_branch_state=EarleyParserBranchState(
            branch_guard=None,
            branch_guard_state=None,
            earley_trie=initial_earley_trie,
        ),
        lookahead=lookahead,
        lookahead_idx=0,
        lex_result=initial_lex_result, context=context,
    )

    return LexEarleyAlgorithmPrefixState(
        branches=tuple(initial_branch for initial_branch, _ in initial_branches_results),
        lookahead_str=lookahead,
        lookahead_str_idx=0,
        seen_eof=False
    )


def force_eof_branch(context: LexEarleyAlgorithmContext, branch: LPBT, lookahead: str, lookahead_idx: int) \
        -> Sequence[LPBT]:
    lexer_parser_state = branch.get_lexer_parser_state()
    completer_lexer_result: LexResult = context.lexer.end_of_file(lexer_parser_state.lexer_state)
    last_branches = process_lex_result(parent_branch_state=lexer_parser_state.parser_state,
                                       context=context, lex_result=completer_lexer_result, lookahead=lookahead,
                                       lookahead_idx=lookahead_idx)

    ret = []
    for last_branch, _ in last_branches:
        if last_branch.parser_state.branch_guard is None \
                or last_branch.parser_state.branch_guard.eof_allowed(last_branch.parser_state.branch_guard_state):

            if last_branch.parser_state.earley_trie.is_complete():
                ret.append(branch.set_lexer_parser_state(last_branch))

    return tuple(ret)


def force_eof(context: LexEarleyAlgorithmContext, state: LexEarleyAlgorithmState) -> LexEarleyAlgorithmState:
    assert not state.seen_eof, "State has already seen EOF"

    if isinstance(state, LexEarleyAlgorithmPrefixState):
        return LexEarleyAlgorithmPrefixState(
            branches=tuple(itertools.chain(*[force_eof_branch(context=context,
                                                              branch=branch,
                                                              lookahead=state.lookahead_str,
                                                              lookahead_idx=state.lookahead_str_idx)
                                             for branch in state.branches])),
            lookahead_str=state.lookahead_str,
            lookahead_str_idx=state.lookahead_str_idx,
            seen_eof=True,
        )
    elif isinstance(state, LexEarleyAlgorithmSuffixState):
        return LexEarleyAlgorithmSuffixState(
            branches=tuple(itertools.chain(*[force_eof_branch(context=context,
                                                              branch=branch,
                                                              lookahead=state.lookahead_str,
                                                              lookahead_idx=state.lookahead_str_idx)
                                             for branch in state.branches])),
            lookahead_str=state.lookahead_str,
            parsed_in_suffix_so_far=state.parsed_in_suffix_so_far,
            lookahead_str_idx=state.lookahead_str_idx,
            seen_eof=True
        )
    else:
        assert isinstance(state, LexEarleyAlgorithmMiddleState)
        assert state.lookahead_str_idx == len(
            state.lookahead_str), f"Called force_eof when lookahead string {state.lookahead_str} not empty"
        suffix_lookahead_branches = list(advance_branch_guards(branches=state.branches,
                                                               new_lookahead=state.suffix_lookahead,
                                                               new_lookahead_idx=0))

        all_branches: List[LexEarleyMiddleBranchState] = []
        for branch in suffix_lookahead_branches:
            modified_context = LexEarleyAlgorithmContext(
                grammar=branch.middle_grammar,
                lexer=context.lexer
            )
            all_branches.extend(
                force_eof_branch(context=modified_context,
                                 branch=branch,
                                 lookahead=state.suffix_lookahead,
                                 lookahead_idx=0))

        return LexEarleyAlgorithmMiddleState(
            branches=tuple(all_branches),
            lookahead_str="",
            suffix_lookahead="",
            seen_eof=True,
            lookahead_str_idx=state.lookahead_str_idx
        )


def is_completable(state: LexEarleyAlgorithmState) -> bool:
    return len(state.branches) > 0


def is_complete(context: LexEarleyAlgorithmContext, state: LexEarleyAlgorithmState) -> bool:
    if not state.seen_eof:
        state = force_eof(context=context, state=state)

    return len(state.branches) > 0


def _to_suffix_parser_state(context: LexEarleyAlgorithmContext,
                            state: LexEarleyAlgorithmPrefixState,
                            suffix: str,
                            make_dummy_trie: bool) -> LexEarleyAlgorithmSuffixState:
    assert state.lookahead_str == "", f"Called to_suffix_parser_state when lookahead " \
                                      f"string {state.lookahead_str} not empty"
    assert not state.seen_eof, f"Called to_suffix_parser_state after an EOF"

    initial_suffix_states = get_all_possible_earley_states(context.grammar, 0)
    initial_suffix_chart = LexEarleyAlgorithmChart(
        states=tuple((state, (TopLevel(),)) for state in initial_suffix_states))
    root_node: LookbackTrieNode[LexEarleyAlgorithmChart] = LookbackTrieNode.create_root_node()

    initial_suffix_earley_trie: AbstractEarleyTrieNode
    if make_dummy_trie:
        initial_suffix_earley_trie = DummyEarleyTrieNode(
            parent=None, this_token=None,
            allowed_token_names=tuple(context.lexer.get_all_possible_token_names()))
    else:
        initial_suffix_earley_trie = EarleyTrieNode(grammar=context.grammar,
                                                    charts=root_node.get_child(initial_suffix_chart),
                                                    allowed_token_names=())

    results = []
    for branch in state.branches:
        suffix_lexer_states = context.lexer.to_suffix_lexer_state(lexer_state=branch.lexer_state, text=suffix)
        for suffix_state_lexer_begin, suffix_lexer_state in suffix_lexer_states:
            # TODO If the lexer state is merely a token continuation, we should treat it specially
            # by scanning instead of using initial_suffix_chart
            results.append(LexEarleySuffixBranchState(
                prefix_state=branch.parser_state,
                suffix_state=EarleyLexerParserBranchState(
                    lexer_state=suffix_lexer_state,
                    parser_state=EarleyParserBranchState(
                        branch_guard=None,
                        branch_guard_state=None,
                        earley_trie=initial_suffix_earley_trie,
                    )
                ),
                skip_n_chars=suffix_state_lexer_begin
            ))

    return LexEarleyAlgorithmSuffixState(
        branches=tuple(results),
        lookahead_str=suffix,
        lookahead_str_idx=0,
        parsed_in_suffix_so_far=LookbackTrieNode.create_root_node(),
        seen_eof=False
    )


def _parse_token_sequence(earley_trie: EarleyTrieNode,
                          tokens: Iterable[Token]) -> Optional[EarleyTrieNode]:
    current_node = earley_trie
    for tok in tokens:
        current_node = current_node.get_child(tok)

        if not current_node.is_completable():
            return None

    return current_node


def _to_middle_parser_state(context: LexEarleyAlgorithmContext,
                            state: LexEarleyAlgorithmSuffixState,
                            middle_lookahead: str) -> LexEarleyAlgorithmMiddleState:
    assert state.lookahead_str == "", f"Called to_middle_parser_state when lookahead string not empty"
    assert state.seen_eof, f"Called to_middle_parser_state before EOF"

    reversed_bnf = context.grammar.reverse()

    reverse_token_sequences = []
    for branch in state.branches:
        # Instead of parsing the suffix tokens as they come in, we have just been storing them
        # If there are two branches with tokens like A C D E F G and B C D E F G, we want to reuse the
        # work in parsing C D E F G.
        # This can only happen if we parse from back to front, using a reversed BNF
        # Doing this also means that the "span start" in the suffix really corresponds to the "span end"
        # Which makes calculating the parse hierarchy and middle BNF a bit nicer

        suffix_dummy_trie = branch.suffix_state.parser_state.earley_trie
        assert isinstance(suffix_dummy_trie, DummyEarleyTrieNode)
        reverse_token_sequences.append(tuple(suffix_dummy_trie.get_reverse_token_sequence()))

    token_nfa, stream_endpoints = token_branches_to_nfa(reverse_token_sequences)
    earley_nfa = EarleyNFA(reversed_bnf, token_nfa)

    results: List[LexEarleyMiddleBranchState] = []
    for branch_ids_with_endpoints, suffix_endpoints in stream_endpoints:
        for branch_id in branch_ids_with_endpoints:
            branch = state.branches[branch_id]
            prefix_charts = tuple(branch.prefix_state.earley_trie[:])
            if not any(earley_nfa.charts[final_idx].processed_states_ordered for final_idx in suffix_endpoints):
                # This branch is not a valid parse
                continue

            prefix_hierarchy = create_parse_hierarchy(context.grammar, prefix_charts,
                                                      final_chart_indices=(len(prefix_charts) - 1,))
            suffix_hierarchy = create_parse_hierarchy(context.grammar, earley_nfa.charts,
                                                      final_chart_indices=suffix_endpoints,
                                                      reverse_state_positions=True)

            middle_bnf = create_middle_bnf(grammar=context.grammar,
                                           prefix_hierarchy=prefix_hierarchy,
                                           suffix_hierarchy=suffix_hierarchy,
                                           prefix_final_chart_indices=(len(prefix_charts) - 1,),
                                           suffix_final_chart_indices=suffix_endpoints)

            # Even if the suffix didn't actually parse any whole tokens, we need to create the middle BNF based on the
            # prefix hierarchy.
            # Because _to_middle_branch will still create a new BNF
            # (where the last element is one of the allowed partial parses), and we initialize the parser with the new BNF,
            # creating middle_bnf takes the left context into account.

            results.extend(_to_middle_branch(prefix_branch_guard=branch.prefix_state.branch_guard,
                                             prefix_branch_guard_state=branch.prefix_state.branch_guard_state,
                                             suffix_lexer_state=branch.suffix_state.lexer_state,
                                             context=context,
                                             middle_bnf=middle_bnf,
                                             middle_lookahead=middle_lookahead))

    global _num_branches_in_middle
    _num_branches_in_middle = len(results)

    return LexEarleyAlgorithmMiddleState(
        branches=tuple(results),
        lookahead_str=middle_lookahead,
        lookahead_str_idx=0,
        suffix_lookahead="".join(state.parsed_in_suffix_so_far.get_full_sequence()),
        seen_eof=False
    )


def _to_middle_branch(prefix_branch_guard: Optional[LexerBranchGuard],
                      prefix_branch_guard_state: Any,
                      suffix_lexer_state: Any,
                      context, middle_bnf, middle_lookahead) -> Sequence[LexEarleyMiddleBranchState]:
    middle_lexer_state, last_tokens = context.lexer.to_middle_lexer_state(suffix_lexer_state)

    if last_tokens:
        middle_bnf = middle_bnf.to_bnf_ending_in(frozenset(last_tokens))

    modified_context = LexEarleyAlgorithmContext(
        grammar=middle_bnf,
        lexer=context.lexer
    )

    middle_chart_init, initial_middle_allowed_tokens = initial_chart_allowed_tokens(middle_bnf)
    root_node: LookbackTrieNode[LexEarleyAlgorithmChart] = LookbackTrieNode.create_root_node()
    middle_trie_node = EarleyTrieNode(grammar=middle_bnf,
                                      charts=root_node.get_child(middle_chart_init),
                                      allowed_token_names=tuple(initial_middle_allowed_tokens))

    initial_branch_state = EarleyParserBranchState(
        branch_guard=prefix_branch_guard,
        branch_guard_state=prefix_branch_guard_state,
        earley_trie=middle_trie_node
    )

    lex_result = context.lexer.lexer_hint(middle_lexer_state, allowed_tokens=initial_middle_allowed_tokens)
    initial_middle_branches_tokens = process_lex_result(
        context=modified_context,
        parent_branch_state=initial_branch_state,
        lex_result=lex_result,
        lookahead=middle_lookahead,
        lookahead_idx=0
    )

    return tuple(LexEarleyMiddleBranchState(
        middle_state=middle_branch,
        middle_grammar=middle_bnf
    ) for middle_branch, _ in initial_middle_branches_tokens)


# There isn't really a nice way to obtain deep metrics for the paper, so using global variables :(
_num_branches_in_middle: Optional[int] = None
_num_branches_after_first_suffix_lexeme: Optional[int] = None


def get_and_reset_stats() -> Dict[str, int]:
    global _num_branches_in_middle
    global _num_branches_after_first_suffix_lexeme

    ret: Dict[str, int] = {}
    if _num_branches_in_middle is not None:
        ret["num_branches_in_middle"] = _num_branches_in_middle
        _num_branches_in_middle = None

    if _num_branches_after_first_suffix_lexeme is not None:
        ret["num_branches_after_first_suffix_lexeme"] = _num_branches_after_first_suffix_lexeme
        _num_branches_after_first_suffix_lexeme = None

    return ret


def lex_earley_to_middle(context: LexEarleyAlgorithmContext, state: LexEarleyAlgorithmPrefixState,
                         suffix: str, middle_lookahead: str) -> LexEarleyAlgorithmMiddleState:
    global _num_branches_in_middle
    global _num_branches_after_first_suffix_lexeme
    suffix_state: LexEarleyAlgorithmState = _to_suffix_parser_state(context, state, suffix, True)
    _num_branches_after_first_suffix_lexeme = len(suffix_state.branches)
    suffix_state = lex_earley_run(context, suffix_state, suffix)
    suffix_state = force_eof(context, suffix_state)
    assert isinstance(suffix_state, LexEarleyAlgorithmSuffixState)
    middle_parser_state = _to_middle_parser_state(context, suffix_state, middle_lookahead)
    _num_branches_in_middle = len(middle_parser_state.branches)
    return middle_parser_state


def lex_earley_run(context: LexEarleyAlgorithmContext, state: LexEarleyAlgorithmState, value: str) \
        -> LexEarleyAlgorithmState:
    for char in value:
        state = lex_earley_step(context, state, char)

    return state


def lex_earley_parse(context: LexEarleyAlgorithmContext, value: str) -> Union[str, bool]:
    """
    :return: True if complete, false if valid but incomplete, valid prefix if valid
    """
    state: LexEarleyAlgorithmState = lex_earley_init(context, value)
    for i, char in enumerate(value):
        state = lex_earley_step(context, state, char)
        if not is_completable(state):
            return value[:i]

    return is_complete(context, state)
