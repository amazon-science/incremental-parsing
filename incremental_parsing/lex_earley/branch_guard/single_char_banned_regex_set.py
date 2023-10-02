from typing import Tuple, Optional, FrozenSet, Sequence

from incremental_parsing.lex_earley.branch_guard.lexer_branch_guard import LexerBranchGuard
from incremental_parsing.lex_earley.incremental_pattern import IncrementalPattern


class LexerBranchGuardBannedRegexSetRealBehavior(LexerBranchGuard):
    """
    Implements the behavior like the Python lexer, as opposed to true maximal munch.
    Otherwise, see banned_regex_set.py's documentation
    """

    def __init__(self, banned_regex_set: Tuple[IncrementalPattern, ...], min_banned_match_length: int):
        self.banned_pattern_set = banned_regex_set
        self.min_banned_match_length = min_banned_match_length

    def branch_allowed(self, text: str, start_index: int, branch_guard_state: Tuple[Tuple[FrozenSet[int], ...], int]) -> \
            Tuple[Optional[bool], Optional[Tuple[Tuple[FrozenSet[int], ...], int]]]:
        token_nfas: Sequence[FrozenSet[int]]
        token_nfas, current_length = branch_guard_state

        for string_idx in range(start_index, len(text)):
            new_token_nfas = []
            current_length += 1
            for pattern, nfa in zip(self.banned_pattern_set, token_nfas):
                next_nfa = pattern.step_forwards(nfa, text[string_idx])
                new_token_nfas.append(next_nfa)

            if current_length < self.min_banned_match_length:
                token_nfas = new_token_nfas
                continue
            elif current_length == self.min_banned_match_length:
                if any(len(nfa) > 0 for nfa in new_token_nfas):
                    return False, None  # Sees a partial match of a token exactly one longer
                else:
                    return True, None  # No such partial match exists
            else:
                assert False, "length should not be greater than match length"

        return None, (tuple(token_nfas), current_length)

    def eof_allowed(self, current_state: Tuple[Tuple[FrozenSet[int], ...], int]) -> bool:
        # If we matched a banned pattern before, we would have already rejected the branch
        return True
