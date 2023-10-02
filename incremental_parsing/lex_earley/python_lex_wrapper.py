import unittest
from dataclasses import dataclass
from typing import Iterable, NamedTuple, Tuple, Optional, Sequence, Union, List, TypeVar, Generic, Literal

from incremental_parsing.lex_earley.lexer import AbstractLexer, LexResult, LexResultFailure, LexResultSuccess, \
    LexResultPartial, Token, LexResultBranch, map_lex_result_state, map_filter_lex_result_state


# Based on Lark's Indenter
# Though needed to think through quite a few cases to make sure it is truly incremental
# https://github.com/lark-parser/lark/blob/ccca874e5fbfca0828fe553279db658295031cd7/lark/indenter.py


class PythonIndenterState(NamedTuple):
    indent_levels: Optional[Tuple[int, ...]]
    current_paren_diff_level: int
    init_min_paren_level: int
    init_max_paren_level: Optional[int]  # Inclusive, None for unbounded max paren level
    queued_hint: Optional[Tuple[str, ...]]
    init_max_indent: Optional[int]
    loose_indents_dedents: Union[None, Literal["indent_first"], Literal["none_or_dedent_first"]]
    initial_indents_include: Tuple[int, ...]

    @property
    def min_paren_level(self) -> int:
        return self.init_min_paren_level + self.current_paren_diff_level

    @property
    def max_paren_level(self) -> Optional[int]:
        return self.init_max_paren_level + self.current_paren_diff_level if self.init_max_paren_level is not None else None


TI = TypeVar('TI')  # Type of the inner lexer result


@dataclass(frozen=True)
class PythonLexWrapperPrefixState(Generic[TI]):
    indenter_state: PythonIndenterState
    inner_state: TI


@dataclass(frozen=True)
class PythonLexWrapperSuffixState(Generic[TI]):
    prefix_indenter_state: PythonIndenterState
    suffix_indenter_state: PythonIndenterState
    inner_state: TI


@dataclass(frozen=True)
class PythonLexWrapperMiddleState(Generic[TI]):
    current_indenter_state: PythonIndenterState
    min_finish_paren_level: int
    max_finish_paren_level: Optional[int]  # Inclusive, None for unbounded max paren level
    required_end_indent_levels: Tuple[int, ...]
    required_end_max_indent: Optional[int]
    inner_state: TI


PythonLexWrapperState = Union[
    PythonLexWrapperPrefixState[TI],
    PythonLexWrapperSuffixState[TI],
    PythonLexWrapperMiddleState[TI]
]


def lift_python_lex_wrapper_result_suffix(
        suffix_state: PythonLexWrapperSuffixState[TI],
        result: LexResult[PythonLexWrapperPrefixState[TI]]
) -> LexResult[PythonLexWrapperSuffixState[TI]]:
    """
    Assumption: result contains a PythonLexWrapperPrefixState
    Convert it to a PythonLexWrapperSuffixState, assuming that the previous suffix state is suffix_state
    """
    return map_lex_result_state(lambda next_lexer_state:
                                PythonLexWrapperSuffixState(
                                    prefix_indenter_state=suffix_state.prefix_indenter_state,
                                    suffix_indenter_state=next_lexer_state.indenter_state,
                                    inner_state=next_lexer_state.inner_state
                                ), result)


def lift_python_lex_wrapper_result_middle(
        middle_state: PythonLexWrapperMiddleState[TI],
        result: LexResult[PythonLexWrapperPrefixState[TI]]
) -> LexResult[PythonLexWrapperMiddleState[TI]]:
    """
    Assumption: result contains a PythonLexWrapperPrefixState
    """
    return map_lex_result_state(lambda next_lexer_state:
                                PythonLexWrapperMiddleState(
                                    current_indenter_state=next_lexer_state.indenter_state,
                                    min_finish_paren_level=middle_state.min_finish_paren_level,
                                    max_finish_paren_level=middle_state.max_finish_paren_level,
                                    inner_state=next_lexer_state.inner_state,
                                    required_end_indent_levels=middle_state.required_end_indent_levels,
                                    required_end_max_indent=middle_state.required_end_max_indent
                                ), result)


NL_type = '_NEWLINE'
OPEN_PAREN_types = ['LPAR', 'LSQB', 'LBRACE']
CLOSE_PAREN_types = ['RPAR', 'RSQB', 'RBRACE']
INDENT_type = '_INDENT'
DEDENT_type = '_DEDENT'
TAB_LEN = 8


class PythonLexWrapper(AbstractLexer[PythonLexWrapperState[TI]], Generic[TI]):
    def __init__(self, lexer: AbstractLexer[TI], ignore_tokens: Iterable[str]):
        self.lexer = lexer
        self.ignore_tokens_without_comment = tuple(it for it in ignore_tokens if it != "COMMENT")

    def initialize(self, initial_hint: Optional[Iterable[str]] = None) -> LexResult[PythonLexWrapperState[TI]]:
        initial_indenter_state = PythonIndenterState(
            indent_levels=(0,),
            current_paren_diff_level=0,
            init_min_paren_level=0,
            init_max_paren_level=0,
            queued_hint=None,
            loose_indents_dedents=None,
            initial_indents_include=(0,),
            init_max_indent=None
        )

        init_hint_tup = self.calc_modified_hint(initial_indenter_state,
                                                tuple(initial_hint)) if initial_hint is not None else None

        inner_lex_result = self.lexer.initialize(initial_hint=init_hint_tup)
        return self.process_inner_lexer_state_result(initial_indenter_state, inner_lex_result)

    def lexer_hint(self, state: PythonLexWrapperState, allowed_tokens: Iterable[str]) -> LexResult[
        PythonLexWrapperState[TI]]:
        if isinstance(state, PythonLexWrapperPrefixState):
            return self.lexer_hint_indenter_state(state.indenter_state, state.inner_state, allowed_tokens)
        elif isinstance(state, PythonLexWrapperSuffixState):
            return lift_python_lex_wrapper_result_suffix(
                state,
                self.lexer_hint_indenter_state(state.suffix_indenter_state, state.inner_state, allowed_tokens)
            )
        elif isinstance(state, PythonLexWrapperMiddleState):
            return lift_python_lex_wrapper_result_middle(
                state,
                self.lexer_hint_indenter_state(state.current_indenter_state, state.inner_state, allowed_tokens)
            )
        else:
            raise ValueError(f'Unexpected lex wrapper state: {type(state)}')

    def calc_modified_hint(self, indenter_state: PythonIndenterState, orig_allowed_tokens: Tuple[str, ...]) -> Tuple[
        str, ...]:
        allowed_tokens = orig_allowed_tokens + self.ignore_tokens_without_comment
        if indenter_state.max_paren_level is None or indenter_state.max_paren_level > 0:
            allowed_tokens += (NL_type,)

        if NL_type in orig_allowed_tokens:
            allowed_tokens += ("COMMENT",)

        return allowed_tokens

    def lexer_hint_indenter_state(self, indenter_state: PythonIndenterState, inner_state: TI,
                                  allowed_tokens: Iterable[str]) -> LexResult[PythonLexWrapperPrefixState[TI]]:
        allowed_tokens = self.calc_modified_hint(indenter_state, tuple(allowed_tokens))
        inner_lex_result = self.lexer.lexer_hint(inner_state, allowed_tokens)
        new_indenter_state = indenter_state._replace(queued_hint=allowed_tokens)
        return self.process_inner_lexer_state_result(new_indenter_state, inner_lex_result)

    def process_single_token(self, lexer_state: PythonIndenterState,
                             inner_token: Token) -> Sequence[Tuple[Tuple[Token, ...], PythonIndenterState]]:
        """
        :param lexer_state: The indenter state prior to processing the token
        :return: A set of branches, where each branch contains a sequence of tokens and a next indenter state
        """
        if inner_token.name in self.ignore_tokens_without_comment or inner_token.name == "COMMENT":
            return ((), lexer_state),  # Importantly, this maintains the queued hint
        elif inner_token.name in OPEN_PAREN_types:
            return ((inner_token,), PythonIndenterState(
                indent_levels=lexer_state.indent_levels,
                current_paren_diff_level=lexer_state.current_paren_diff_level + 1,
                init_min_paren_level=lexer_state.init_min_paren_level,
                init_max_paren_level=lexer_state.init_max_paren_level,
                queued_hint=None,
                loose_indents_dedents=lexer_state.loose_indents_dedents,
                initial_indents_include=lexer_state.initial_indents_include,
                init_max_indent=lexer_state.init_max_indent
            )),
        elif inner_token.name in CLOSE_PAREN_types:
            if lexer_state.max_paren_level == 0:
                return ()
            elif lexer_state.min_paren_level == 0:
                # The initialization min parentheses depth must have actually been higher
                # This branch is still possible (because max parens >= 1) but we need to adjust for this new information
                # min paren level will be 0 after this still, but this is through adjusting the init_min and diff
                return ((inner_token,), PythonIndenterState(
                    indent_levels=lexer_state.indent_levels,
                    current_paren_diff_level=lexer_state.current_paren_diff_level - 1,
                    init_min_paren_level=lexer_state.init_min_paren_level + 1,
                    init_max_paren_level=lexer_state.init_max_paren_level,
                    queued_hint=None,
                    loose_indents_dedents=lexer_state.loose_indents_dedents,
                    initial_indents_include=lexer_state.initial_indents_include,
                    init_max_indent=lexer_state.init_max_indent
                )),
            else:
                # Min paren level >= 1; just adjust the diff
                return ((inner_token,), PythonIndenterState(
                    indent_levels=lexer_state.indent_levels,
                    current_paren_diff_level=lexer_state.current_paren_diff_level - 1,
                    init_min_paren_level=lexer_state.init_min_paren_level,
                    init_max_paren_level=lexer_state.init_max_paren_level,
                    queued_hint=None,
                    loose_indents_dedents=lexer_state.loose_indents_dedents,
                    initial_indents_include=lexer_state.initial_indents_include,
                    init_max_indent=lexer_state.init_max_indent
                )),
        elif inner_token.name == NL_type:
            # If min level == 0
            # If max level > 0
            # Branch into max level = 0 and min level = 1
            # Do nothing for min level = 1
            # Process newline for min level = 0

            possible_results: List[Tuple[Tuple[Token, ...], PythonIndenterState]] = []

            if lexer_state.max_paren_level is None or lexer_state.max_paren_level > 0:
                # Ignore the newline, and reuse the queued hint
                if lexer_state.min_paren_level > 0:
                    possible_results.append(((), lexer_state))
                else:
                    # Min paren level=0, and max paren level > 0
                    # We want to add a branch where the min paren level is 1, and the token is ignored
                    # The next if statement will add a branch where the paren level is exactly 0
                    possible_results.append(((), lexer_state._replace(
                        init_min_paren_level=lexer_state.init_min_paren_level + 1
                    )))

            if lexer_state.min_paren_level == 0:
                # Add a branch where min paren level = 0 and max paren level = 0; i.e. the newline is _not_ ignored
                # And an indent or dedent is processed

                indent_str = inner_token.text.rsplit('\n', 1)[-1].rsplit('\r', 1)[-1]  # Tabs and spaces
                indent = indent_str.count(' ') + indent_str.count('\t') * TAB_LEN

                # current_indent_levels = lexer_state.indent_levels
                # result_tokens = [inner_token]
                # skip_this_branch = False

                if lexer_state.indent_levels is None:
                    assert lexer_state.loose_indents_dedents

                    if indent != 0 and lexer_state.loose_indents_dedents == "indent_first":
                        # Handle case where the first newline is really an indent
                        possible_results.append(((inner_token, Token(INDENT_type, "")),
                                                 PythonIndenterState(
                                                     indent_levels=(indent,),
                                                     current_paren_diff_level=lexer_state.current_paren_diff_level,
                                                     init_min_paren_level=lexer_state.init_min_paren_level,
                                                     init_max_paren_level=lexer_state.init_min_paren_level,
                                                     # So that max paren level = 0
                                                     queued_hint=None,
                                                     loose_indents_dedents=lexer_state.loose_indents_dedents,
                                                     initial_indents_include=(),
                                                     init_max_indent=indent - 1
                                                 )))

                    if lexer_state.loose_indents_dedents == "none_or_dedent_first":
                        # Case where the first newline stays on the same level or dedents
                        possible_results.append(
                            ((inner_token, Token(DEDENT_type, "", loose_behavior=True, max_loosiness=None)),
                             PythonIndenterState(
                                 indent_levels=(indent,),
                                 current_paren_diff_level=lexer_state.current_paren_diff_level,
                                 init_min_paren_level=lexer_state.init_min_paren_level,
                                 init_max_paren_level=lexer_state.init_min_paren_level,
                                 queued_hint=None,
                                 loose_indents_dedents=lexer_state.loose_indents_dedents,
                                 initial_indents_include=(indent,),
                                 init_max_indent=None
                             )))
                elif indent > lexer_state.indent_levels[-1]:
                    possible_results.append(((inner_token, Token(INDENT_type, "")),
                                             PythonIndenterState(
                                                 indent_levels=lexer_state.indent_levels + (indent,),
                                                 current_paren_diff_level=lexer_state.current_paren_diff_level,
                                                 init_min_paren_level=lexer_state.init_min_paren_level,
                                                 init_max_paren_level=lexer_state.init_min_paren_level,
                                                 queued_hint=None,
                                                 loose_indents_dedents=lexer_state.loose_indents_dedents,
                                                 initial_indents_include=lexer_state.initial_indents_include,
                                                 init_max_indent=lexer_state.init_max_indent
                                             )))
                else:
                    if lexer_state.loose_indents_dedents:
                        # We need to insert dedents.
                        # If we know all the indents which could possibly be along the way, then we know exactly how
                        # many dedents need to be inserted. But if we dedent more than that, then there is some
                        # uncertainty about how many dedents are actually needed (and thus we insert "loose" dedents)
                        loose_dedent_result = insert_loose_dedents(
                            current_indentation_levels=lexer_state.indent_levels,
                            new_indentation_level=indent
                        )

                        if loose_dedent_result is None:
                            # This branch is not allowed, don't add anything to possible_results
                            pass
                        else:
                            num_strict_dedents, do_loose_dedent, loosiness, \
                                current_indent_levels, seen_new_initial_indent = loose_dedent_result

                            result_tokens = [inner_token]

                            for _ in range(num_strict_dedents):
                                result_tokens.append(Token(DEDENT_type, ""))

                            if do_loose_dedent:
                                result_tokens.append(Token(DEDENT_type, "", loose_behavior=True,
                                                           max_loosiness=loosiness))

                            if seen_new_initial_indent:
                                next_initial_indents_include = lexer_state.initial_indents_include + (indent,)
                            else:
                                next_initial_indents_include = lexer_state.initial_indents_include

                            possible_results.append((tuple(result_tokens),
                                                     PythonIndenterState(
                                                         indent_levels=current_indent_levels,
                                                         current_paren_diff_level=lexer_state.current_paren_diff_level,
                                                         init_min_paren_level=lexer_state.init_min_paren_level,
                                                         init_max_paren_level=lexer_state.init_min_paren_level,
                                                         queued_hint=None,
                                                         loose_indents_dedents=lexer_state.loose_indents_dedents,
                                                         initial_indents_include=next_initial_indents_include,
                                                         init_max_indent=lexer_state.init_max_indent
                                                     )))
                    else:
                        current_indent_levels = lexer_state.indent_levels
                        result_tokens = [inner_token]

                        while indent < current_indent_levels[-1]:
                            current_indent_levels = current_indent_levels[:-1]
                            result_tokens.append(Token(DEDENT_type, ""))

                        if indent == current_indent_levels[-1]:
                            possible_results.append((tuple(result_tokens), PythonIndenterState(
                                indent_levels=current_indent_levels,
                                current_paren_diff_level=lexer_state.current_paren_diff_level,
                                init_min_paren_level=lexer_state.init_min_paren_level,
                                init_max_paren_level=lexer_state.init_min_paren_level,  # So that max paren level = 0
                                queued_hint=None,
                                loose_indents_dedents=lexer_state.loose_indents_dedents,
                                initial_indents_include=lexer_state.initial_indents_include,
                                init_max_indent=lexer_state.init_max_indent
                            )))

            return possible_results
        else:  # Regular non-indent-specific token
            # Don't re-use the same hint
            return ((inner_token,), lexer_state._replace(queued_hint=None)),

    def process_lex_result_success(self, lexer_state: PythonIndenterState,
                                   token_sequence: Tuple[Token, ...],
                                   inner_next_lexer_state: TI) -> LexResult[PythonLexWrapperPrefixState[TI]]:

        current_branches: List[Tuple[Tuple[Token, ...], PythonIndenterState]] = [((), lexer_state)]

        for inner_token in token_sequence:
            next_branches = []
            for (tokens_so_far, indenter_state) in current_branches:
                branch_results = self.process_single_token(
                    lexer_state=indenter_state,
                    inner_token=inner_token
                )
                for (branch_result_tokens, branch_result_state) in branch_results:
                    next_branches.append((tokens_so_far + branch_result_tokens, branch_result_state))

            current_branches = next_branches

        result_branches: List[LexResult[PythonLexWrapperPrefixState]] = []

        for (tokens_so_far, indenter_state) in current_branches:
            if tokens_so_far:
                # We have tokens to give to the parser
                result_branches.append(LexResultSuccess(
                    next_lexer_state=PythonLexWrapperPrefixState(
                        indenter_state=indenter_state,
                        inner_state=inner_next_lexer_state
                    ),
                    token_sequence=tokens_so_far
                ))
            else:
                # We don't have tokens to give to the parser, they were all ignored.
                if indenter_state.queued_hint is not None:
                    result_branches.append(self.process_inner_lexer_state_result(
                        lexer_state=indenter_state,
                        inner_result=self.lexer.lexer_hint(
                            state=inner_next_lexer_state,
                            allowed_tokens=indenter_state.queued_hint
                        )
                    ))
                else:
                    result_branches.append(LexResultPartial(
                        next_lexer_state=PythonLexWrapperPrefixState(
                            indenter_state=indenter_state,
                            inner_state=inner_next_lexer_state
                        )
                    ))

        if len(result_branches) == 0:
            return LexResultFailure()
        elif len(result_branches) == 1:
            return result_branches[0]
        else:
            return LexResultBranch(
                branches=tuple((branch, None, None) for branch in result_branches)
            )

    def process_inner_lexer_state_result(self, lexer_state: PythonIndenterState,
                                         inner_result: LexResult[TI]) -> LexResult[PythonLexWrapperPrefixState[TI]]:
        if isinstance(inner_result, LexResultBranch):
            results = []
            for branch, branch_guard, branch_guard_state in inner_result.branches:
                inner_branch_result = self.process_inner_lexer_state_result(lexer_state, branch)
                results.append((inner_branch_result, branch_guard, branch_guard_state))
            return LexResultBranch(tuple(results))
        elif isinstance(inner_result, LexResultFailure):
            return inner_result
        elif isinstance(inner_result, LexResultPartial):
            return LexResultPartial(
                next_lexer_state=PythonLexWrapperPrefixState(
                    indenter_state=lexer_state,
                    inner_state=inner_result.next_lexer_state
                )
            )
        else:
            return self.process_lex_result_success(lexer_state, inner_result.token_sequence,
                                                   inner_result.next_lexer_state)

    def advance_lexer_state(self, lexer_state: PythonLexWrapperState[TI], char: str) -> LexResult[
        PythonLexWrapperState[TI]]:
        # It is always OK for there to be too few or too many indent levels UNTIL we hit the first non-whitespace
        # non-comment character. This is because the next token in the input could always just be another newline.
        # This is nice, as we don't need to introspect the state of the lexer to fail fast

        inner_result = self.lexer.advance_lexer_state(lexer_state.inner_state, char)

        if isinstance(lexer_state, PythonLexWrapperPrefixState):
            return self.process_inner_lexer_state_result(lexer_state.indenter_state, inner_result)
        elif isinstance(lexer_state, PythonLexWrapperSuffixState):
            prefix_result = self.process_inner_lexer_state_result(lexer_state.suffix_indenter_state, inner_result)
            return lift_python_lex_wrapper_result_suffix(lexer_state, prefix_result)
        elif isinstance(lexer_state, PythonLexWrapperMiddleState):
            prefix_result = self.process_inner_lexer_state_result(lexer_state.current_indenter_state, inner_result)
            return lift_python_lex_wrapper_result_middle(lexer_state, prefix_result)
        else:
            assert False, "Unknown lexer state"

    def ensure_paren_level_overlaps(self, res: LexResult[PythonLexWrapperPrefixState[TI]], min_parens: int,
                                    max_parens: Optional[int]) \
            -> LexResult[PythonLexWrapperPrefixState[TI]]:
        def process_state(prefix_state: PythonLexWrapperPrefixState[TI]) -> Optional[PythonLexWrapperPrefixState[TI]]:
            state = prefix_state.indenter_state
            if state.max_paren_level is not None and state.max_paren_level < min_parens:
                return None
            elif max_parens is not None and state.min_paren_level > max_parens:
                return None
            else:
                constrained_min_parens = max(state.min_paren_level, min_parens)
                if max_parens is None:
                    constrained_max_parens = state.max_paren_level
                elif state.max_paren_level is None:
                    constrained_max_parens = max_parens
                else:
                    constrained_max_parens = min(state.max_paren_level, max_parens)

                constrained_init_min_parens = constrained_min_parens - state.current_paren_diff_level
                assert constrained_init_min_parens >= 0

                if constrained_max_parens is None:
                    constrained_init_max_parens = None
                else:
                    constrained_init_max_parens = constrained_max_parens - state.current_paren_diff_level

                # TODO Make this into something like with the Earley prefix vs suffix branch architecture
                return PythonLexWrapperPrefixState(
                    indenter_state=PythonIndenterState(
                        current_paren_diff_level=state.current_paren_diff_level,
                        init_min_paren_level=constrained_init_min_parens,
                        init_max_paren_level=constrained_init_max_parens,
                        indent_levels=state.indent_levels,
                        queued_hint=state.queued_hint,
                        loose_indents_dedents=state.loose_indents_dedents,
                        initial_indents_include=state.initial_indents_include,
                        init_max_indent=state.init_max_indent
                    ),
                    inner_state=prefix_state.inner_state
                )

        return map_filter_lex_result_state(process_state, res)

    def ensure_initial_indents_included(self, res: LexResult[PythonLexWrapperPrefixState[TI]],
                                        must_include_indents: Tuple[int, ...], init_max_indent: Optional[int]) \
            -> LexResult[PythonLexWrapperPrefixState[TI]]:

        def process_state(prefix_state: PythonLexWrapperPrefixState[TI]) -> Optional[PythonLexWrapperPrefixState[TI]]:
            assert prefix_state.indenter_state.indent_levels is not None
            if all(indent_level in prefix_state.indenter_state.indent_levels for indent_level in must_include_indents) \
                    and (init_max_indent is None or prefix_state.indenter_state.indent_levels[-1] <= init_max_indent):
                return prefix_state
            else:
                return None

        return map_filter_lex_result_state(process_state, res)

    def end_of_file_indenter_state(self, indenter_state: PythonIndenterState, inner_state: TI) -> LexResult[
        PythonLexWrapperPrefixState[TI]]:
        return self.ensure_paren_level_overlaps(
            self.process_inner_lexer_state_result(indenter_state, self.lexer.end_of_file(inner_state)), 0, 0
        )

    def end_of_file(self, lexer_state: PythonLexWrapperState[TI]) -> LexResult[PythonLexWrapperState[TI]]:
        if isinstance(lexer_state, PythonLexWrapperPrefixState):
            return self.end_of_file_indenter_state(lexer_state.indenter_state, lexer_state.inner_state)
        elif isinstance(lexer_state, PythonLexWrapperSuffixState):
            return lift_python_lex_wrapper_result_suffix(
                lexer_state,
                self.end_of_file_indenter_state(lexer_state.suffix_indenter_state, lexer_state.inner_state)
            )
        elif isinstance(lexer_state, PythonLexWrapperMiddleState):
            res = self.process_inner_lexer_state_result(lexer_state=lexer_state.current_indenter_state,
                                                        inner_result=self.lexer.end_of_file(lexer_state.inner_state))
            filtered_res = self.ensure_paren_level_overlaps(res, lexer_state.min_finish_paren_level,
                                                            lexer_state.max_finish_paren_level)
            filtered_res = self.ensure_initial_indents_included(filtered_res, lexer_state.required_end_indent_levels,
                                                                lexer_state.required_end_max_indent)
            return lift_python_lex_wrapper_result_middle(lexer_state, filtered_res)
        else:
            assert False, "Unknown lexer state"

    def to_suffix_lexer_state(self, text: str, lexer_state: PythonLexWrapperState[TI]) -> Tuple[
        Tuple[int, PythonLexWrapperState[TI]], ...]:
        assert isinstance(lexer_state, PythonLexWrapperPrefixState)

        inner_lexer_states = self.lexer.to_suffix_lexer_state(
            text=text,
            lexer_state=lexer_state.inner_state
        )

        results = []

        for start_idx, inner_lexer_state in inner_lexer_states:
            def get_state(loose_indents_dedents: Literal['indent_first', 'none_or_dedent_first']) \
                    -> PythonLexWrapperSuffixState:
                return PythonLexWrapperSuffixState(
                    prefix_indenter_state=lexer_state.indenter_state,
                    suffix_indenter_state=PythonIndenterState(
                        current_paren_diff_level=0,
                        init_min_paren_level=0,
                        init_max_paren_level=None,
                        indent_levels=None,
                        queued_hint=None,
                        loose_indents_dedents=loose_indents_dedents,
                        initial_indents_include=(),
                        init_max_indent=None
                    ),
                    inner_state=inner_lexer_state
                )

            results.append((start_idx, get_state("indent_first")))
            results.append((start_idx, get_state("none_or_dedent_first")))

        return tuple(results)

    def to_middle_lexer_state(self, lexer_state: PythonLexWrapperState[TI]) -> Tuple[PythonLexWrapperState[TI],
    Optional[Tuple[str, ...]]]:
        assert isinstance(lexer_state, PythonLexWrapperSuffixState)

        inner_middle_state, inner_last_tokens = self.lexer.to_middle_lexer_state(lexer_state.inner_state)

        if inner_last_tokens is not None:
            if any(ignored_token in inner_last_tokens for ignored_token in self.ignore_tokens_without_comment) \
                    or "COMMENT" in inner_last_tokens:
                inner_last_tokens = None
            elif NL_type in inner_last_tokens:
                if lexer_state.suffix_indenter_state.init_min_paren_level == 0:
                    inner_last_tokens += (INDENT_type, DEDENT_type)
                else:
                    inner_last_tokens = tuple(t for t in inner_last_tokens if t != NL_type)

        return PythonLexWrapperMiddleState(
            inner_state=inner_middle_state,
            current_indenter_state=lexer_state.prefix_indenter_state,
            min_finish_paren_level=lexer_state.suffix_indenter_state.init_min_paren_level,
            max_finish_paren_level=lexer_state.suffix_indenter_state.init_max_paren_level,
            required_end_indent_levels=lexer_state.suffix_indenter_state.initial_indents_include,
            required_end_max_indent=lexer_state.suffix_indenter_state.init_max_indent
        ), inner_last_tokens

    def get_all_possible_token_names(self) -> Iterable[str]:
        return frozenset(self.lexer.get_all_possible_token_names()).union((INDENT_type, DEDENT_type))


def insert_loose_dedents(current_indentation_levels: Tuple[int, ...], new_indentation_level: int) -> Optional[
    Tuple[int, bool, int, Tuple[int, ...], bool]]:
    """
    Calculate what to do when dedenting in loose indentation mode
    :return: # of regular dedents; whether to insert a loose dedent; max loosiness; new indentation levels;
             # contains an unseen indentation level
    Or None if the branch is invalid
    """
    num_dedents = 0
    indent_levels = list(current_indentation_levels)
    last_indent_level = indent_levels.pop()
    while True:
        if last_indent_level == new_indentation_level:
            return num_dedents, False, 0, tuple(indent_levels) + (new_indentation_level,), False
        elif len(indent_levels) == 0:
            assert new_indentation_level < last_indent_level
            if new_indentation_level + 1 == last_indent_level:
                return num_dedents + 1, False, 0, (new_indentation_level,), True
            else:
                return num_dedents + 1, True, (last_indent_level - new_indentation_level - 1), \
                    (new_indentation_level,), True
        elif new_indentation_level > indent_levels[-1]:
            return None
        else:
            last_indent_level = indent_levels.pop()
            num_dedents += 1


class TestLooseDedents(unittest.TestCase):
    def test_loose_dedent_none(self):
        self.assertEqual(insert_loose_dedents((3,), 3), (0, False, 0, (3,), False))
        self.assertEqual(insert_loose_dedents((3, 5, 6), 6), (0, False, 0, (3, 5, 6), False))
        self.assertEqual(insert_loose_dedents((0,), 0), (0, False, 0, (0,), False))

    def test_multi_loose_dedent(self):
        self.assertEqual(insert_loose_dedents((3, 5), 3), (1, False, 0, (3,), False))
        self.assertEqual(insert_loose_dedents((3, 5, 6, 8), 5), (2, False, 0, (3, 5), False))
        self.assertEqual(insert_loose_dedents((3, 5, 6, 8), 3), (3, False, 0, (3,), False))
        self.assertEqual(insert_loose_dedents((3, 5, 6, 8), 2), (4, False, 0, (2,), True))
        self.assertEqual(insert_loose_dedents((3, 5, 6, 8), 1), (4, True, 1, (1,), True))
        self.assertEqual(insert_loose_dedents((3, 5, 6, 8), 0), (4, True, 2, (0,), True))

    def test_invalid_loose_dedent(self):
        self.assertIsNone(insert_loose_dedents((3, 5, 7), 4))
        self.assertIsNone(insert_loose_dedents((3, 5, 7), 6))
