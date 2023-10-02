import abc
from typing import Any, Tuple, Optional


class LexerBranchGuard(abc.ABC):
    @abc.abstractmethod
    def branch_allowed(self, text: str, start_index: int, branch_guard_state: Any) -> Tuple[Optional[bool], Any]:
        """
        Is the current lexer branch allowed, given the lookahead text?
        Invariant:
        t1, t2 : str
        t = t1 + t2
        bgs: branch_guard_state
        assert branch_allowed(t, bgs) == branch_allowed(t2, branch_allowed(t1, bgs)[1])
        :param text: Lookahead text to determine whether the branch is allowed by the lexer
        :param start_index: Start index of the lookahead text
        :param branch_guard_state: Some internal state of the branch guard
        :return: True if the branch is definitely allowed and we no longer need to check for future text.
        None if the branch is allowed so far, but there is some future text that would rule out the branch.
        False if the branch is not allowed and should be pruned.
        """
        pass

    @abc.abstractmethod
    def eof_allowed(self, branch_guard_state: Any) -> bool:
        """
        Are we allowed to reach the end of the file here?
        """
        pass

    def replace(self, branch_guard_state: Any) -> Tuple[Optional["LexerBranchGuard"], Any]:
        """
        Can this be replaced by a different branch guard?
        """
        return self, branch_guard_state
