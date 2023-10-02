import abc
from typing import Tuple, Optional

from incremental_parsing.utils.simple_nfa import SimpleNFAMutable


class RegexNode(abc.ABC):
    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def to_nfa(self, nfa: SimpleNFAMutable, start_ix: int, end_ix: int):
        pass


class RegexSequence(RegexNode):
    def __init__(self, nodes: Tuple[RegexNode, ...]):
        self.nodes = nodes

    def __str__(self):
        return ''.join(map(str, self.nodes))

    def to_nfa(self, nfa: SimpleNFAMutable, start_ix: int, end_ix: int):
        if len(self.nodes) == 0:
            nfa.add_eps_transition(start_ix, end_ix)
        else:
            endpoints = [start_ix]
            for i in range(len(self.nodes) - 1):
                endpoints.append(nfa.add_state())
            endpoints.append(end_ix)

            for i, node in enumerate(self.nodes):
                node.to_nfa(nfa, endpoints[i], endpoints[i + 1])


class RegexRepeat(RegexNode):
    def __init__(self, node: RegexNode, min_repeat: int, max_repeat: Optional[int]):
        self.node = node
        self.min_repeat = min_repeat
        self.max_repeat = max_repeat

    def __str__(self):
        if self.min_repeat == 0 and self.max_repeat == 0:
            return ''
        elif self.min_repeat == 0 and self.max_repeat is None:
            return f'{self.node}*'
        elif self.min_repeat == 1 and self.max_repeat is None:
            return f'{self.node}+'
        elif self.min_repeat == 0 and self.max_repeat == 1:
            return f'{self.node}?'
        elif self.min_repeat == 0:
            return f'{self.node}{{,{self.max_repeat}}}'
        elif self.max_repeat is None:
            return f'{self.node}{{{self.min_repeat},}}'
        elif self.min_repeat == self.max_repeat:
            return f'{self.node}{{{self.min_repeat}}}'
        else:
            return f'{self.node}{{{self.min_repeat},{self.max_repeat}}}'

    def to_nfa(self, nfa: SimpleNFAMutable, start_idx: int, end_idx: int):
        idx_after_mandatory_repeats = start_idx

        for i in range(self.min_repeat):
            next_start_idx = nfa.add_state()
            self.node.to_nfa(nfa, idx_after_mandatory_repeats, next_start_idx)
            idx_after_mandatory_repeats = next_start_idx

        nfa.add_eps_transition(idx_after_mandatory_repeats, end_idx)

        if self.max_repeat == self.min_repeat:
            return
        else:
            idx_after_optional_repeats = nfa.add_state()
            self.node.to_nfa(nfa, idx_after_mandatory_repeats, idx_after_optional_repeats)
            nfa.add_eps_transition(idx_after_optional_repeats, end_idx)
            if self.max_repeat is None:
                nfa.add_eps_transition(idx_after_optional_repeats, idx_after_mandatory_repeats)
            else:
                for i in range(self.max_repeat - self.min_repeat - 1):
                    next_idx_after_optional_repeat = nfa.add_state()
                    self.node.to_nfa(nfa, idx_after_optional_repeats, next_idx_after_optional_repeat)
                    nfa.add_eps_transition(next_idx_after_optional_repeat, end_idx)
                    idx_after_optional_repeats = next_idx_after_optional_repeat


class RegexAtom(RegexNode):
    def __init__(self, value: str, flags: int):
        self.value = value
        self.flags = flags

    def __str__(self):
        return self.value

    def to_nfa(self, nfa: SimpleNFAMutable, start_ix: int, end_ix: int):
        nfa.add_atom_transition(start_ix, end_ix, (self.value, self.flags))


class RegexAlternates(RegexNode):
    def __init__(self, nodes: Tuple[RegexNode, ...]):
        self.nodes = nodes

    def __str__(self):
        return '|'.join(map(str, self.nodes))

    def to_nfa(self, nfa: SimpleNFAMutable, start_ix: int, end_ix: int):
        for node in self.nodes:
            node.to_nfa(nfa, start_ix, end_ix)


class RegexGroup(RegexNode):
    def __init__(self, node: RegexNode, group_prefix: str):
        self.node = node
        self.group_prefix = group_prefix

    def __str__(self):
        return f'({self.group_prefix}{self.node})'

    def to_nfa(self, nfa: SimpleNFAMutable, start_ix: int, end_ix: int):
        self.node.to_nfa(nfa, start_ix, end_ix)
