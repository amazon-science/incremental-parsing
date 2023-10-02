from typing import Tuple, Optional, FrozenSet, Sequence

from incremental_parsing.lex_earley.branch_guard.lexer_branch_guard import LexerBranchGuard
from incremental_parsing.lex_earley.incremental_pattern import IncrementalPattern


class LexerBranchGuardBannedRegexSet(LexerBranchGuard):
    """
    This branch guard is used to make sure there aren't any longer matches
    I.E. if we have generated "abc" so far, and have emitted a branch with a token for identifiers,
    then "identifiers which are >=4 chars" would be a banned regex. If we see "d" next, then "abcd" would be the
    leftmost-longest identifier, and so we should delete the "abc" branch.
    """
    def __init__(self, banned_regex_set: Tuple[IncrementalPattern, ...], min_banned_match_length: int):
        self.banned_pattern_set = banned_regex_set
        self.min_banned_match_length = min_banned_match_length

    def branch_allowed(
            self, text: str, start_index: int, branch_guard_state: Tuple[Tuple[FrozenSet[int], ...], int]) -> \
            Tuple[Optional[bool], Optional[Tuple[Tuple[FrozenSet[int], ...], int]]]:
        token_nfas: Sequence[FrozenSet[int]]
        token_nfas, current_length = branch_guard_state

        for string_idx in range(start_index, len(text)):
            new_token_nfas = []
            seen_any_partial = False

            current_length += 1
            for pattern, nfa in zip(self.banned_pattern_set, token_nfas):
                next_nfa = pattern.step_forwards(nfa, text[string_idx])
                if current_length >= self.min_banned_match_length and any(n in pattern.final_states for n in next_nfa):
                    return False, None  # Seen a full match of a banned token longer than the min len -> branch is bad
                elif next_nfa:
                    seen_any_partial = True
                    new_token_nfas.append(next_nfa)

            if not seen_any_partial:
                return True, None  # All banned tokens have dropped out, this branch is definitely okay
            else:
                token_nfas = new_token_nfas

        return None, (tuple(token_nfas), current_length)

    def eof_allowed(self, current_state: Tuple[Tuple[FrozenSet[int], ...], int]) -> bool:
        # If we matched a banned pattern before, we would have already rejected the branch
        return True
