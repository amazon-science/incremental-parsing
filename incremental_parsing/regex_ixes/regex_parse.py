import re
import unittest
from typing import Tuple

from incremental_parsing.regex_ixes.regex_tree import RegexSequence, RegexAlternates, RegexGroup, RegexAtom, RegexNode, \
    RegexRepeat


def parse_regex(regex: str, flags: int = 0) -> RegexNode:
    assert not regex.startswith('^')
    assert not regex.endswith('$')
    regex_tree, remaining_regex = parse_regex_alternates(regex, flags)
    assert remaining_regex == ""
    return regex_tree


def parse_regex_alternates(regex: str, flags: int) -> Tuple[RegexNode, str]:
    elems = []
    while regex != "" and regex[0] != ')':
        elem, regex = parse_regex_sequence(regex, flags)
        elems.append(elem)
        if regex == "" or regex[0] == ')':
            break
        elif regex[0] == '|':
            regex = regex[1:]
            continue
        else:
            raise ValueError("Expected | or ), got " + regex)

    if not elems:
        return RegexSequence(()), regex
    elif len(elems) == 1:
        return elems[0], regex
    else:
        return RegexAlternates(tuple(elems)), regex


def parse_regex_sequence(regex: str, flags: int) -> Tuple[RegexNode, str]:
    seq = []
    while regex != "" and regex[0] not in '|)':
        if regex[0] == "(":
            elem, regex = parse_group(regex, flags)
        elif regex[0] == "[":
            elem, regex = parse_char_class(regex, flags)
        else:
            elem, regex = parse_char(regex, flags)

        elem, regex = parse_operator(elem, regex, flags)
        seq.append(elem)

    if len(seq) == 1:
        return seq[0], regex
    else:
        return RegexSequence(tuple(seq)), regex


def parse_group(regex: str, flags: int) -> Tuple[RegexNode, str]:
    assert regex[0] == '('
    regex = regex[1:]
    if regex.startswith('?:'):
        prefix = '?:'
        regex = regex[2:]
    elif regex.startswith('?'):
        raise NotImplementedError(f"Regex group ({regex}")
    else:
        prefix = ''

    inside, regex = parse_regex_alternates(regex, flags)
    assert regex[0] == ')'
    regex = regex[1:]
    return RegexGroup(inside, prefix), regex


def parse_char_class(regex: str, flags: int) -> Tuple[RegexNode, str]:
    assert regex[0] == '['
    regex = regex[1:]
    char_class_text = "["

    while True:
        assert "]" in regex, "Unclosed character class"

        pre, regex = regex.split(']', 1)
        char_class_text += pre + "]"

        if len(pre) == 0 or pre[-1] != '\\':
            break
        else:
            # We just saw (and added) a \], didn't close the char class
            continue

    return RegexAtom(char_class_text, flags), regex


def parse_char(regex: str, flags: int) -> Tuple[RegexNode, str]:
    if regex[0] == '\\':
        if regex[1].isdigit():
            if regex[2].isdigit():
                return RegexAtom(regex[:3], flags), regex[3:]
        return RegexAtom(regex[:2], flags), regex[2:]
    else:
        return RegexAtom(regex[0], flags), regex[1:]


number = re.compile(r"\d+")


def parse_number(regex: str) -> Tuple[int, str]:
    match = number.match(regex)
    if match is None:
        raise ValueError(f"Invalid number: {regex}")
    return int(match.group()), regex[match.end():]


def parse_operator(elem: RegexNode, regex: str, flags: int) -> Tuple[RegexNode, str]:
    if regex == "":
        return elem, ""
    elif regex[0] == "*":
        if len(regex) > 1 and regex[1] == "?":
            raise NotImplementedError(f"Lazy regex operator *?")
        return RegexRepeat(elem, 0, None), regex[1:]
    elif regex[0] == "+":
        if len(regex) > 1 and regex[1] == "?":
            raise NotImplementedError(f"Lazy regex operator +?")
        return RegexRepeat(elem, 1, None), regex[1:]
    elif regex[0] == "?":
        if len(regex) > 1 and regex[1] == "?":
            raise NotImplementedError(f"Lazy regex operator ??")
        return RegexRepeat(elem, 0, 1), regex[1:]
    elif regex[0] == "{":
        regex = regex[1:]

        if regex[0] == "}":
            # Make sure we don't have x{}
            raise ValueError(f"Empty number in {regex}")

        if regex[0] == ",":
            min_repeat = 0
        else:
            min_repeat, regex = parse_number(regex)

        if regex[0] == ",":
            regex = regex[1:]
            if regex[0] == "}":
                max_repeat = None
            else:
                max_repeat, regex = parse_number(regex)
        elif regex[0] == "}":
            max_repeat = min_repeat
        else:
            raise ValueError(f"Invalid number in {regex}")

        assert regex[0] == "}"
        regex = regex[1:]
        if len(regex) > 0 and regex[0] == "?":
            raise NotImplementedError(f"Lazy regex operator")

        return RegexRepeat(elem, min_repeat, max_repeat), regex
    else:
        return elem, regex


class TestRegexParses(unittest.TestCase):
    def test_regex_parse_repeat(self):
        self.assertEqual(str(parse_regex('a{,}')), 'a*')
        self.assertEqual(str(parse_regex('a{1,}')), 'a+')

    def test_bad_parses(self):
        self.assertRaises(ValueError, parse_regex, 'a{}')
        self.assertRaises(NotImplementedError, parse_regex, 'a{3,4}?')
        self.assertRaises(NotImplementedError, parse_regex, 'a??')
        self.assertRaises(NotImplementedError, parse_regex, 'a*?')
        self.assertRaises(NotImplementedError, parse_regex, 'a+?')
        self.assertRaises(ValueError, parse_regex, 'a{asdf}')
        self.assertRaises(ValueError, parse_regex, 'a{,asdf}')
        self.assertRaises(ValueError, parse_regex, 'a{1asdf}')
        self.assertRaises(IndexError, parse_regex, 'a(b')
        self.assertRaises(AssertionError, parse_regex, 'a)b')
        self.assertRaises(AssertionError, parse_regex, 'a)')
        self.assertRaises(AssertionError, parse_regex, '[abc\\]')
        self.assertRaises(NotImplementedError, parse_regex, '(?>a)')
