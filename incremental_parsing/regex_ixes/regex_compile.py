import unittest

from incremental_parsing.regex_ixes.regex_nfa import regex_match, ATOM_PATTERN_TYPE
from incremental_parsing.regex_ixes.regex_parse import parse_regex
from incremental_parsing.utils.simple_nfa import SimpleNFA, SimpleNFAMutable


def compile(regex: str, flags: int) -> SimpleNFA[ATOM_PATTERN_TYPE, str]:
    r = parse_regex(regex, flags)
    n: SimpleNFAMutable[ATOM_PATTERN_TYPE] = SimpleNFAMutable()
    r.to_nfa(n, 0, 1)
    return n.finalize(regex_match)


class TestRegexNFA(unittest.TestCase):
    def test_regex_comment(self):
        r = compile(r'#[^\n]*', 0)
        self.assertFalse(r.fullmatch(""))
        self.assertFalse(r.fullmatch(' #'))
        self.assertTrue(r.fullmatch('#'))
        self.assertFalse(r.fullmatch('#\n'))
        self.assertTrue(r.fullmatch('# asdfaweproiajweioprjaoisejf'))

    def test_regex_linecont(self):
        r = compile(r'\\[\t \f]*\r?\n', 0)

        self.assertFalse(r.fullmatch(""))
        self.assertFalse(r.fullmatch('\\\na'))
        self.assertTrue(r.fullmatch('\\\n'))
        self.assertTrue(r.fullmatch('\\ \n'))
        self.assertTrue(r.fullmatch('\\ \t\t\n'))
        self.assertTrue(r.fullmatch('\\ \t\t\r\n'))
        self.assertFalse(r.fullmatch('\\ \t\r\t\r\n'))

    def test_regex_newline(self):
        r = compile(r'((\r?\n[\t ]*|#[^\n]*))+', 0)
        self.assertFalse(r.fullmatch(""))
        self.assertFalse(r.fullmatch('a'))
        self.assertFalse(r.fullmatch('a\n'))
        self.assertFalse(r.fullmatch('\na'))
        self.assertTrue(r.fullmatch('\n'))
        self.assertTrue(r.fullmatch('#hello\n'))
        self.assertFalse(r.fullmatch('#hello\nfoo'))
