from incremental_parsing.lex_earley.lark_grammar import get_python_context
from incremental_parsing.lex_earley.lex_earley import lex_earley_init, lex_earley_run, is_completable, is_complete, \
    lex_earley_to_middle, LexEarleyAlgorithmState, LexEarleyAlgorithmPrefixState


def run_tests():
    context = get_python_context()
    init_state = lex_earley_init(context)

    def assert_state(state, completable, complete):
        if completable:
            assert state is not None
            assert is_completable(state)

            if complete:
                assert is_complete(context, state)
            else:
                assert not is_complete(context, state)
        else:
            assert state is None or not is_completable(state)

    def assert_text(text, completable, complete):
        state = lex_earley_run(context, init_state, text)
        assert_state(state, completable, complete)
        assert_text_context("", text, "", completable, complete)

    def assert_text_context(prefix, middle, suffix, completable, complete):
        # There are numerous more or less equivalent ways to actually invoke the parser
        # Test all of them

        # No left context, no lookahead
        state: LexEarleyAlgorithmState = \
            lex_earley_to_middle(context=context, state=init_state, suffix=suffix, middle_lookahead="")
        state = lex_earley_run(context=context, state=state, value=(prefix + middle))
        assert_state(state, completable, complete)

        # No left context, with lookahead
        # In some cases, a known lookahead can be more efficient
        state = lex_earley_to_middle(context=context, state=init_state, suffix=suffix,
                                     middle_lookahead=(prefix + middle))
        state = lex_earley_run(context=context, state=state, value=(prefix + middle))
        assert_state(state, completable, complete)

        # With left context, no lookahead
        # In some cases, a known lookahead can be more efficient
        state = lex_earley_run(context=context, state=init_state, value=prefix)
        assert isinstance(state, LexEarleyAlgorithmPrefixState)
        state = lex_earley_to_middle(context=context, state=state, suffix=suffix, middle_lookahead="")
        state = lex_earley_run(context=context, state=state, value=middle)
        assert_state(state, completable, complete)

        # With left context, and lookahead
        state = lex_earley_init(context, lookahead=prefix)
        state = lex_earley_run(context=context, state=state, value=prefix)
        assert isinstance(state, LexEarleyAlgorithmPrefixState)
        state = lex_earley_to_middle(context=context, state=state, suffix=suffix, middle_lookahead=middle)
        state = lex_earley_run(context=context, state=state, value=middle)
        assert_state(state, completable, complete)

    assert_text("f(a and", True, False)
    assert_text("f(a andy", False, False)
    assert_text("f(aandy", True, False)
    assert_text("f(aand and b)\n", True, True)
    assert_text("f(aand and b andy)", False, False)
    assert_text("""# foo
assert bar(a) == "b"
assert bar(c) == "d"
""", True, True)

    assert_text("""
blah = 1
      """, True, False)

    assert_text("""
blah = 1
 f""", False, False)

    assert_text("""
if a:
  blah = 1
 """, True, False)

    assert_text("""
if a:
  blah = 1
 f""", False, False)

    assert_text("""
if a:
  blah = 1
f""", True, False)

    assert_text("""
if a:
  blah = 1
   f""", False, False)

    assert_text("""
if a:
  blah = (
 f""", True, False)

    assert_text("""
'''
asdf\\\n
'''
""", True, True)

    assert_text("""
if foo:
   '''
blah
   '''        
""", True, True)

    assert_text("""
if foo:
   '''
blah
   '''        
  return
""", False, False)

    assert_text("assert find_literals('The quick brown fox jumps over the lazy dog.', 'fox') == ('fox', 16, 19)\n", True, True)

    assert_text_context("0", "o", "r 1\n", True, False)
    assert_text_context("0", "or", " 1\n", False, False)
    assert_text_context("0 ", "o", "r 1\n", True, True)
    assert_text_context("def foo(", "asdf)", "    pass\n", True, False)
    assert_text_context("def foo(", "asdf):", "    pass\n", True, True)


if __name__ == "__main__":
    run_tests()