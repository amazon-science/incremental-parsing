import curses
from typing import List, Tuple

from incremental_parsing.lex_earley.lark_grammar import get_python_context
from incremental_parsing.lex_earley.lex_earley import lex_earley_init, LexEarleyAlgorithmState, is_complete, \
    is_completable, lex_earley_step, lex_earley_to_middle


def run(stdscr):
    context = get_python_context()
    current_state: LexEarleyAlgorithmState = lex_earley_init(context)

    if False:
        current_state = lex_earley_to_middle(context, current_state, "", "")

    tokens: List[Tuple[str, LexEarleyAlgorithmState]] = []

    while True:
        stdscr.addstr(0, 0, "Input:")

        input_str = "".join(t[0] for t in tokens)
        stdscr.addstr(1, 0, input_str)

        stdscr.addstr("\n=====\n")

        if current_state is None:
            stdscr.addstr("Invalid")
        elif is_complete(context, current_state):
            stdscr.addstr("Complete")
        elif is_completable(current_state):
            stdscr.addstr("Completable")
        else:
            stdscr.addstr("Invalid")

        c = stdscr.getch()
        stdscr.clear()

        add = False
        char = None

        if 32 <= c < 127:
            char = chr(c)
            add = True
        elif c in (curses.KEY_ENTER, 10, 13):
            add = True
            char = "\n"
        elif c in (curses.KEY_BACKSPACE, 127):
            if len(tokens) > 0:
                _, current_state = tokens.pop()

        if current_state is None:
            add = False

        if add:
            assert char is not None
            tokens.append((char, current_state))
            current_state = lex_earley_step(context, current_state, char)
            # assert len(current_state.branches) <= 2


if __name__ == '__main__':
    curses.wrapper(run)
