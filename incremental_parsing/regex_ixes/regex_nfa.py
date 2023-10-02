from typing import Dict, Tuple

import regex

ATOM_PATTERN_TYPE = Tuple[str, int]
CACHED_ATOM_REGEXES: Dict[ATOM_PATTERN_TYPE, regex.Pattern] = dict()
CACHED_VALUES: Dict[Tuple[ATOM_PATTERN_TYPE, str], bool] = dict()


def regex_match(pattern: ATOM_PATTERN_TYPE, char: str) -> bool:
    val = CACHED_VALUES.get((pattern, char))
    if val is not None:
        return val

    pat = CACHED_ATOM_REGEXES.get(pattern)
    if pat is not None:
        val = (pat.fullmatch(char) is not None)
        CACHED_VALUES[(pattern, char)] = val
        return val

    pat = regex.compile(pattern[0], flags=pattern[1])
    CACHED_ATOM_REGEXES[pattern] = pat
    val = (pat.fullmatch(char) is not None)
    CACHED_VALUES[(pattern, char)] = val
    return val
