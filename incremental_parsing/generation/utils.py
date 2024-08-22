import ast
import datetime
from typing import Tuple, Callable, Optional, Sequence, List

import numpy as np
import torch
from ansi.colour import bg, fg
from transformers import PreTrainedTokenizer


def tokenizer_int64(tokenizer: PreTrainedTokenizer, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
    val = tokenizer(text, return_tensors="pt")
    return val["input_ids"].type(torch.int64), val["attention_mask"].type(torch.int64)


def create_balanced_context(pre_input_ids: torch.Tensor,
                            pre_attention_mask: torch.Tensor,
                            post_input_ids: torch.Tensor,
                            post_attention_mask: torch.Tensor,
                            tokenizer: PreTrainedTokenizer,
                            max_generation_length: int,
                            device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    max_tokens_for_context = tokenizer.model_max_length - max_generation_length - 3

    fim_prefix_tokens = tokenizer("<fim-prefix>", return_tensors="pt")
    fim_suffix_tokens = tokenizer("<fim-suffix>", return_tensors="pt")
    fim_middle_tokens = tokenizer("<fim-middle>", return_tensors="pt")

    shorter_context = min(pre_input_ids.shape[1], post_input_ids.shape[1])

    if shorter_context * 2 > max_tokens_for_context:
        # The context is too long, and both sides have over half the allowed amount
        # Restrict each side to half
        max_context_each_side = max_tokens_for_context // 2
    else:
        # One side has less than half of the allowed amount; allocate the rest of the allowed amount to the longer side
        max_context_each_side = max_tokens_for_context - shorter_context

    # Suffix might be empty, but even then this likely helps to prevent the LLM from babbling

    input_ids = torch.concat((
        fim_prefix_tokens["input_ids"],
        pre_input_ids[:, -max_context_each_side:],
        fim_suffix_tokens["input_ids"],
        post_input_ids[:, :max_context_each_side],
        fim_middle_tokens["input_ids"]
    ), dim=1).to(device)
    attention_mask = torch.concat((
        fim_prefix_tokens["attention_mask"],
        pre_attention_mask[:, -max_context_each_side:],
        fim_suffix_tokens["attention_mask"],
        post_attention_mask[:, :max_context_each_side],
        fim_middle_tokens["attention_mask"]
    ), dim=1).to(device)

    return input_ids, attention_mask


colors = [
    (bg.red, fg.black),
    (bg.green, fg.black),
    (bg.yellow, fg.black),
    (bg.blue, fg.black),
    (bg.magenta, fg.black),
    (bg.cyan, fg.black),
    (bg.white, fg.black),
    (bg.black, fg.white)
]


def color_idx(idx: int) -> Callable[[str], str]:
    b, f = colors[idx % len(colors)]
    return lambda s: b(f(s))


def try_incremental_unconstrained(prefix: str, middle: str, suffix: str) -> Tuple[
    Optional[str], Sequence[datetime.timedelta]]:
    """
    Check whether every prefix of unconstrained generation parses
    (for a performance comparison to constrained generation)
    """
    longest_match: Optional[str] = None
    times: List[datetime.timedelta] = []

    for i in range(len(middle) + 1):
        start_time = datetime.datetime.now()
        full_text = prefix + middle[:i] + suffix

        # noinspection PyBroadException
        try:
            ast.parse(full_text)
        except:
            pass
        else:
            longest_match = middle[:i]

        end_time = datetime.datetime.now()
        times.append(end_time - start_time)

    return longest_match, times


def get_p50_p90_mean_count(times: Sequence[datetime.timedelta]) -> Tuple[Optional[float], Optional[float], Optional[float], float]:
    if len(times) == 0:
        return None, None, None, 0

    time_seconds = [time.total_seconds() for time in times]
    return float(np.quantile(time_seconds, .5)), float(np.quantile(time_seconds, .9)), sum(time_seconds) / len(time_seconds), len(time_seconds)
