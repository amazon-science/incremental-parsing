from typing import Iterable

import regex

flag_map = {
    'i': regex.I,
    'm': regex.M,
    's': regex.S,
    'l': regex.L,
    'u': regex.U,
    'x': regex.X
}


def flags_to_regex_flags(re_flags: Iterable[str]) -> int:
    current_flags = 0
    for flag in re_flags:
        current_flags |= flag_map[flag]

    return current_flags
