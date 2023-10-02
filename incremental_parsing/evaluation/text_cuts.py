import io
import random
import tokenize
from tokenize import TokenInfo
from typing import Tuple, List


def cut_text_random(text: str, min_cut_percentage: float, max_cut_percentage: float, cut_amount: float) -> Tuple[
    str, str, str]:
    """
    Take a random point "p" between min_cut_percentage and max_cut_percentage of the way through the file.
    Split it into 3 parts: before p, from p to p+cut_amount, and from p+cut_amount to the end.
    Note that p+cut_amount might be > 1, in which case middle will be smaller than cut_amount and suffix will be empty
    """
    cut_percentage = min_cut_percentage + (max_cut_percentage - min_cut_percentage) * random.random()
    cut_index = int(cut_percentage * len(text))
    cut_amount = cut_amount * len(text)
    cut_end = cut_index + int(cut_amount)
    prefix = text[:cut_index]
    middle = text[cut_index:cut_end]
    suffix = text[cut_end:]
    return prefix, middle, suffix


IndentationRun = List[Tuple[int, int]]  # A set of (start_idx, end_idx) with the same indentation


def get_runs_of_same_indentation(tokens: List[TokenInfo]) -> List[IndentationRun]:
    """
    Output including (3, 6) would mean that we can cut the string before token 3, 4, or 5
    Output including (3, 4) means only valid cut is before token 3

    Input:
    0: Pass
    1: Indent ----|
    2: Pass       |
    3: Indent -|  |
    4: Pass    |  |
    5: Dedent -|  |
    6: Pass       |
    7: Dedent ----|
    8: Pass
    9: Indent -|
    10:Pass    |
    11:Dedent -|

    So the output would include these four groups [index may be off by one, need to double-check]
    ((0, 1), (8, 9))
    ((2, 3), (6, 7))
    ((4, 5))
    ((10, 11))
    The idea being that all tokens that belong to a group are related to each other by indentation level
    """
    finished_runs: List[IndentationRun] = []
    run_stack: List[IndentationRun] = []
    current_run: IndentationRun = []
    current_run_start = 0

    for i, token in enumerate(tokens):
        if token.type == tokenize.INDENT:
            if current_run_start < i:
                current_run.append((current_run_start, i))
            run_stack.append(current_run)
            current_run = []
            current_run_start = i + 1
        elif token.type == tokenize.DEDENT:
            if current_run_start < i:
                current_run.append((current_run_start, i))
            finished_runs.append(current_run)
            current_run = run_stack.pop()
            current_run_start = i + 1

    assert len(run_stack) == 0
    if current_run_start < len(tokens):
        current_run.append((current_run_start, len(tokens)))

    finished_runs.append(current_run)

    return finished_runs


def extract_absolute_positions(text: str, tokens: List[TokenInfo]) -> List[Tuple[int, int]]:
    """
    Surprisingly it's quite weird going from line/col to absolute pos.
    This function does so (giving the left/right bounds of a token)
    """
    str_line_offsets = [0]
    running_str_offset = 0
    for line in text.split("\n"):
        str_line_offsets.append(running_str_offset)
        running_str_offset += len(line) + 1

    token_positions: List[Tuple[int, int]] = []

    for token in tokens:
        start_pos = str_line_offsets[token.start[0]] + token.start[1]
        end_pos = str_line_offsets[token.end[0]] + token.end[1]

        assert text[start_pos:end_pos] == token.string, \
            f"{token.start[0]}:{token.start[1]}-{token.end[0]}:{token.end[1]}: " \
            f"{text[start_pos:end_pos]} != {token.string}"
        token_positions.append((start_pos, end_pos))

    return token_positions


def select_random_points_from_indentation_run(runs: IndentationRun, k: int) -> List[int]:
    """Pick k points from within the same indentation run (see get_runs_of_same_indentation)"""
    run_choices = random.choices(population=runs, weights=[r[1] - r[0] for r in runs], k=k)
    return [random.randint(r[0], r[1] - 1) for r in run_choices]


def run_weights(runs: List[IndentationRun]) -> List[int]:
    return [sum(r[1] - r[0] for r in run) for run in runs]


def cut_text_same_indentation_levels(text: str) -> Tuple[str, str, str]:
    """
    An attempt to create a method of chopping up text that is hopefully more reminiscent of what insertion points would
    look like in practice.
    The main things being that the prefix ends at the same indentation level the suffix begins,
    and that the suffix begins on a token boundary.
    """
    text_bytes = bytes(text, 'utf-8')
    byte_io = io.BytesIO(text_bytes)
    tokens = list(tokenize.tokenize(byte_io.readline))[1:]
    token_posns = extract_absolute_positions(text, tokens)

    runs = get_runs_of_same_indentation(tokens)
    left_token_num, right_token_num = select_random_points_from_indentation_run(
        random.choices(runs, weights=run_weights(runs), k=1)[0], 2)

    if left_token_num > right_token_num:
        left_token_num, right_token_num = right_token_num, left_token_num

    left_token_start, left_token_end = token_posns[left_token_num]
    left_cutoff_point = random.randint(left_token_start, left_token_end)
    right_token_start, right_token_end = token_posns[right_token_num]
    right_cutoff_point = random.randint(right_token_start, right_token_end)
    return text[:left_cutoff_point], text[left_cutoff_point:right_cutoff_point], text[right_cutoff_point:]
