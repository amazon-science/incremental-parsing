from collections import deque, defaultdict
from typing import TypeVar, Generic, AbstractSet, Mapping, Callable, Dict, FrozenSet, Set, Sequence, Iterable, \
    DefaultDict

P = TypeVar("P")  # Pattern Type
A = TypeVar("A")  # Atom Type


class SimpleNFA(Generic[P, A]):
    def __init__(self, states: AbstractSet[int], start_states: AbstractSet[int], end_states: AbstractSet[int],
                 atom_transitions_forward: Mapping[int, Mapping[P, AbstractSet[int]]],
                 atom_transitions_backward: Mapping[int, Mapping[P, AbstractSet[int]]],
                 matcher: Callable[[P, A], bool]):

        self.states = frozenset(states)
        self.start_states = frozenset(start_states)
        self.end_states = frozenset(end_states)

        self.atom_transitions_forward = {state: {atom: frozenset(dests)
                                                 for atom, dests in atom_transitions_forward[state].items()}
                                         for state in atom_transitions_forward}
        self.atom_transitions_backward = {state: {atom: frozenset(dests)
                                                  for atom, dests in atom_transitions_backward[state].items()}
                                          for state in atom_transitions_backward}
        self.matcher = matcher

    def step_state(self, state: int, atom: A, transition_map: Dict[int, Dict[P, FrozenSet[int]]]):
        if state not in transition_map:
            return frozenset()

        next_states: Set[int] = set()

        for possible_atom in transition_map[state]:
            if self.matcher(possible_atom, atom):
                next_states.update(transition_map[state][possible_atom])

        return frozenset(next_states)

    def step_forward(self, state_set: FrozenSet[int], atom: A) -> FrozenSet[int]:
        if len(state_set) == 0:
            return frozenset()
        return frozenset.union(*[self.step_state(state, atom, self.atom_transitions_forward) for state in state_set])

    def step_forward_any(self, state_set: FrozenSet[int]) -> FrozenSet[int]:
        ret_set: Set[int] = set()
        for state in state_set:
            for atom, dests in self.atom_transitions_forward[state].items():
                ret_set.update(dests)

        return frozenset(ret_set)

    def fullmatch(self, string: Sequence[A]) -> bool:
        current_states = self.start_states
        for char in string:
            current_states = self.step_forward(current_states, char)
            if not current_states:
                return False

        return any(current_state in self.end_states for current_state in current_states)

    def step_backward(self, state_set: FrozenSet[int], atom: A) -> FrozenSet[int]:
        if len(state_set) == 0:
            return frozenset()
        return frozenset.union(*[self.step_state(state, atom, self.atom_transitions_backward) for state in state_set])

    def get_reachable_forward(self, state_set: FrozenSet[int]) -> FrozenSet[int]:
        return frozenset(get_reachable(state_set, self.atom_transitions_forward))

    def is_extensible(self, state_set: FrozenSet[int]) -> bool:
        return any((state in self.atom_transitions_forward
                    and any(len(outgoing) > 0 for _, outgoing in self.atom_transitions_forward[state].items()))
                   for state in state_set)


def get_reachable(start_set: Iterable[int], mapping_set: Mapping[int, Mapping[P, AbstractSet[int]]]) -> \
        Set[int]:
    reachable = set()
    frontier = deque(start_set)
    while len(frontier) > 0:
        state = frontier.popleft()
        if state in reachable:
            continue

        reachable.add(state)
        if state in mapping_set:
            for atom, dests in mapping_set[state].items():
                frontier.extend(dests)

    return reachable


class SimpleNFAMutable(Generic[P]):

    def __init__(self):
        self.state_counter = 2
        self.states = {0, 1}
        self.start_states = {0}
        self.end_states = {1}
        self.eps_transitions_forward: DefaultDict[int, Set[int]] = defaultdict(set)
        self.eps_transitions_backward: DefaultDict[int, Set[int]] = defaultdict(set)
        self.atom_transitions_forward: DefaultDict[int, DefaultDict[P, Set[int]]] = defaultdict(
            lambda: defaultdict(set))
        self.atom_transitions_backward: DefaultDict[int, DefaultDict[P, Set[int]]] = defaultdict(
            lambda: defaultdict(set))

    def add_state(self) -> int:
        snum = self.state_counter
        self.state_counter += 1
        self.states.add(snum)
        return snum

    def add_eps_transition(self, origin: int, dest: int):
        self.eps_transitions_forward[origin].add(dest)
        self.eps_transitions_backward[dest].add(origin)

    def add_atom_transition(self, origin: int, dest: int, pattern: P):
        self.atom_transitions_forward[origin][pattern].add(dest)
        self.atom_transitions_backward[dest][pattern].add(origin)

    def eliminate_eps_transition(self, origin: int, dest: int):
        if origin == dest:
            return

        if dest in self.atom_transitions_forward:
            for atom, final_dests in self.atom_transitions_forward[dest].items():
                for final_dest in final_dests:
                    self.atom_transitions_forward[origin][atom].add(final_dest)
                    self.atom_transitions_backward[final_dest][atom].add(origin)

        if origin in self.start_states:
            self.start_states.add(dest)

        if dest in self.end_states:
            self.end_states.add(origin)

    def compute_backwards_eps_reachability(self, orig_state: int) -> Set[int]:
        reachable = set()
        frontier = deque([orig_state])
        while len(frontier) > 0:
            state = frontier.popleft()
            if state in reachable:
                continue
            else:
                reachable.add(state)

            if state in self.eps_transitions_backward:
                for dest in self.eps_transitions_backward[state]:
                    frontier.append(dest)

        return reachable

    def eliminate_all_eps_transitions(self):
        backwards_eps_map = {state: self.compute_backwards_eps_reachability(state) for state in
                             self.eps_transitions_backward.keys()}
        for state, backwards_eps in backwards_eps_map.items():
            for backwards_eps_state in backwards_eps:
                if state != backwards_eps_state:
                    self.eliminate_eps_transition(backwards_eps_state, state)

        self.eps_transitions_forward.clear()
        self.eps_transitions_backward.clear()

    def get_reachable_states(self):
        reachable_states_forward: Set[int] = get_reachable(self.start_states, self.atom_transitions_forward)
        reachable_states_backward: Set[int] = get_reachable(self.end_states, self.atom_transitions_backward)
        return reachable_states_forward.intersection(reachable_states_backward)

    def remove_states(self, states_to_remove: Set[int]):
        assert len(self.eps_transitions_forward) == 0, "Can only remove states if there are no eps transitions; call " \
                                                       "eliminate_all_eps_transitions first"

        new_states = set()
        new_start_states = set()
        new_end_states = set()
        new_eps_transitions_forward: DefaultDict[int, Set[int]] = defaultdict(set)
        new_eps_transitions_backward: DefaultDict[int, Set[int]] = defaultdict(set)
        new_atom_transitions_forward: DefaultDict[int, DefaultDict[P, Set[int]]] = defaultdict(
            lambda: defaultdict(set))
        new_atom_transitions_backward: DefaultDict[int, DefaultDict[P, Set[int]]] = defaultdict(
            lambda: defaultdict(set))

        for state in self.states:
            if state in states_to_remove:
                continue

            new_states.add(state)
            if state in self.start_states:
                new_start_states.add(state)
            if state in self.end_states:
                new_end_states.add(state)

            for dest in self.eps_transitions_forward[state]:
                if dest not in states_to_remove:
                    new_eps_transitions_forward[state].add(dest)
                    new_eps_transitions_backward[dest].add(state)

            for atom, dests in self.atom_transitions_forward[state].items():
                for dest in dests:
                    if dest not in states_to_remove:
                        new_atom_transitions_forward[state][atom].add(dest)
                        new_atom_transitions_backward[dest][atom].add(state)

        self.states = new_states
        self.start_states = new_start_states
        self.end_states = new_end_states
        self.eps_transitions_forward = new_eps_transitions_forward
        self.eps_transitions_backward = new_eps_transitions_backward
        self.atom_transitions_forward = new_atom_transitions_forward
        self.atom_transitions_backward = new_atom_transitions_backward

    def remove_unreachable_states(self):
        reachable_states = self.get_reachable_states()
        self.remove_states(self.states - reachable_states)

    def finalize(self, matcher: Callable[[P, A], bool]):
        self.eliminate_all_eps_transitions()
        self.remove_unreachable_states()
        return SimpleNFA(self.states, self.start_states, self.end_states,
                         self.atom_transitions_forward, self.atom_transitions_backward,
                         matcher)
