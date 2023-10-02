import unittest
from random import random
from typing import Generic, TypeVar, Optional, Tuple, List, Sequence, overload, Dict

import math

T = TypeVar('T')


class LookbackTrieNode(Generic[T], Sequence[T]):
    depth: int
    """
    Depth from root; root node has depth 0
    """

    value: Optional[T]

    parent: Optional["LookbackTrieNode[T]"]

    parent_pointer_stack: Tuple[Tuple[int, "LookbackTrieNode[T]"], ...]
    """
    Each element represents the depth of a parent node, and the node itself.
    These are the non-bottom levels of a skip list 
    """

    children: Dict[T, "LookbackTrieNode[T]"]

    @staticmethod
    def create_root_node() -> "LookbackTrieNode[T]":
        return LookbackTrieNode(0, None, None, ())

    def __init__(self, depth: int, parent: Optional["LookbackTrieNode[T]"],
                 value: Optional[T],
                 parent_pointer_stack: Tuple[Tuple[int, "LookbackTrieNode[T]"], ...]):
        self.depth = depth
        self.parent = parent
        self.value = value
        self.parent_pointer_stack = parent_pointer_stack
        self.children = {}

    def __len__(self) -> int:
        return self.depth

    def get_node_at_depth(self, goal_depth: int):
        """
        First value is at depth 1
        """
        if goal_depth < 0:
            raise IndexError("depth must be nonnegative")
        elif goal_depth > self.depth:
            raise IndexError("n must be less than or equal to the depth of the node")

        current_node = self
        current_node_depth = self.depth

        while current_node_depth > goal_depth:
            assert current_node.parent is not None
            best_parent = current_node.parent
            best_parent_depth = current_node_depth - 1

            for possible_ancestor_depth, possible_ancestor in best_parent.parent_pointer_stack:
                if possible_ancestor_depth >= goal_depth:
                    best_parent = possible_ancestor
                    best_parent_depth = possible_ancestor_depth
                else:
                    break

            current_node = best_parent
            current_node_depth = best_parent_depth

        assert current_node_depth == goal_depth
        return current_node

    def get_value_at_index(self, value_index: int):
        if value_index < 0:
            raise IndexError("value_index must be nonnegative")
        elif value_index >= self.depth:
            raise IndexError("value_index must be less than the depth of the node")

        return self.get_node_at_depth(value_index + 1).value

    def get_child(self, value: T) -> "LookbackTrieNode[T]":
        if value in self.children:
            return self.children[value]

        num_levels = math.floor(-math.log2(random()))

        parent_pointer_stack = [(self.depth, self) for _ in range(num_levels)]
        parent_pointer_stack.extend(self.parent_pointer_stack[num_levels:])

        new_node = LookbackTrieNode(self.depth + 1, self, value, tuple(parent_pointer_stack))
        self.children[value] = new_node
        return new_node

    def get_n_suffix(self, n: int) -> List[T]:
        if n < 0:
            raise IndexError("n must be nonnegative")
        elif n > self.depth:
            raise IndexError("n must be less than or equal to the depth of the node")

        sequence = []
        current_node: LookbackTrieNode[T] = self
        for _ in range(n):
            assert current_node.parent is not None
            assert current_node.value is not None
            sequence.append(current_node.value)
            current_node = current_node.parent
        return list(reversed(sequence))

    def get_full_sequence(self) -> List[T]:
        return self.get_n_suffix(self.depth)

    @overload
    def __getitem__(self, index: int) -> T:
        ...

    @overload
    def __getitem__(self, index: slice) -> List[T]:
        ...

    def __getitem__(self, index):
        if isinstance(index, int):
            real_index = index if index >= 0 else self.depth + index

            return self.get_value_at_index(real_index)
        elif isinstance(index, slice):
            start, stop, step = index.indices(self.depth)
            real_start = start if start >= 0 else self.depth + start
            real_stop = stop if stop >= 0 else self.depth + stop

            if step != 1:
                raise ValueError("slice step must be 1")

            if real_start >= real_stop:
                return []

            end_node = self.get_node_at_depth(real_stop)

            length = real_stop - real_start
            return end_node.get_n_suffix(length)
        else:
            raise NotImplementedError()


class LookbackTrieNodeTest(unittest.TestCase):

    def test_get_full_sequence(self):
        node: LookbackTrieNode[str] = LookbackTrieNode.create_root_node()
        node = node.get_child('a').get_child('b')
        node.get_child("f")
        node = node.get_child('c')
        node.get_child("g")
        self.assertEqual("".join(node.get_full_sequence()), "abc")

    def test_indexing(self):
        node: LookbackTrieNode[str] = LookbackTrieNode.create_root_node()
        for i in range(100):
            node = node.get_child(str(i))

        for i in range(100):
            self.assertEqual(node[i], str(i))

        full_result = list(str(i) for i in range(100))
        for i in range(-100, 100):
            for j in range(-100, 100):
                self.assertEqual(node[i:j], full_result[i:j])

        for i in range(-100, 100):
            self.assertEqual(node[i:], full_result[i:])
            self.assertEqual(node[:i], full_result[:i])

        self.assertEqual(node[100:], [])
        self.assertEqual(node[:], full_result)

    def test_reasonable_speed(self):
        node: LookbackTrieNode[str] = LookbackTrieNode.create_root_node()
        for i in range(100000):
            node = node.get_child(str(i))

        for i in range(100000):
            self.assertEqual(node[i], str(i))
