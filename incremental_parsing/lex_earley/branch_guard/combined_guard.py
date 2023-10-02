from typing import Iterable, Any, Tuple, Optional

from incremental_parsing.lex_earley.branch_guard.lexer_branch_guard import LexerBranchGuard


class LexerBranchGuardCombined(LexerBranchGuard):
    """
    So that a branch can have more than one branch guard
    """
    def __init__(self, branch_guards: Iterable[LexerBranchGuard]):
        self.branch_guards = tuple(branch_guards)

    @staticmethod
    def combine_branch_guard_states(states: Iterable[Any]) -> Any:
        s = tuple(states)
        return s, (True,) * len(s)

    def branch_allowed(self, text: str, start_index: int, branch_guard_state: Any) -> Tuple[Optional[bool], Any]:
        inner_states, still_consider = branch_guard_state
        next_inner_states, next_still_consider = [], []
        any_failed = False
        any_ambiguous = False
        for guard, inner_state, consider in zip(self.branch_guards, inner_states, still_consider):
            if not consider:
                next_inner_states.append(inner_state)
                next_still_consider.append(consider)
            else:
                inner_res, inner_next_state = guard.branch_allowed(text, start_index, inner_state)
                next_inner_states.append(inner_next_state)
                if inner_res:
                    next_still_consider.append(False)
                elif inner_res is None:
                    next_still_consider.append(True)
                    any_ambiguous = True
                else:
                    next_still_consider.append(True)
                    any_failed = True

        if any_failed:
            result = False
        elif any_ambiguous:
            result = None
        else:
            result = True

        return result, (tuple(next_inner_states), tuple(next_still_consider))

    def eof_allowed(self, branch_guard_state: Any) -> bool:
        inner_states, still_considers = branch_guard_state
        return all((not consider) or branch.eof_allowed(inner_state)
                   for branch, inner_state, consider in zip(self.branch_guards, inner_states, still_considers))

    def replace(self, branch_guard_state: Any) -> Tuple[Optional["LexerBranchGuard"], Any]:
        inner_states, still_considers = branch_guard_state
        if all(still_considers):
            return self, branch_guard_state

        rep_branches, rep_inner_states = [], []
        for branch, inner_state, still_consider in zip(self.branch_guards, inner_states, still_considers):
            if not still_consider:
                continue

            repl_branch, repl_state = branch.replace(inner_state)
            if not repl_branch:
                continue

            rep_branches.append(repl_branch)
            rep_inner_states.append(repl_state)

        if len(rep_branches) == 0:
            return None, None
        elif len(rep_branches) == 1:
            return rep_branches[0], rep_inner_states[0]
        else:
            return LexerBranchGuardCombined(rep_branches), (tuple(rep_inner_states), (True,) * len(rep_inner_states))
