import argparse
import ast
import datetime
import json
import random
import traceback
from pathlib import Path
from typing import NamedTuple, Optional

import datasets
import transformers
from datasets import Dataset
from transformers import GenerationMixin, PreTrainedTokenizer, AutoTokenizer

from incremental_parsing.evaluation.text_cuts import cut_text_random, cut_text_same_indentation_levels
from incremental_parsing.generation.constrained_generation import do_prefix_suffix_constrained_generation
from incremental_parsing.generation.utils import tokenizer_int64, create_balanced_context, \
    try_incremental_unconstrained, get_p50_p90_mean_count
from incremental_parsing.lex_earley.lark_grammar import get_python_context
from incremental_parsing.lex_earley.lex_earley import LexEarleyAlgorithmContext, lex_earley_parse, get_and_reset_stats


class ProgramParameters(NamedTuple):
    min_seed: int
    max_seed: int  # inclusive
    min_data_idx: int
    max_data_idx: int  # inclusive
    min_cut_idx: int
    max_cut_idx: int  # inclusive
    beam_size: int
    model_name: str
    dataset_name: str
    dataset_path: str
    results_path: Path
    device: str
    max_tokens: int
    nice_cuts: bool
    num_token_finding_attempts: Optional[int]
    redo_completed: bool


class RunParameters(NamedTuple):
    seed: int
    data_index: int
    cut_index: int
    beam_size: int
    results_path: Path
    device: str
    max_tokens: int
    nice_cuts: bool
    num_token_finding_attempts: Optional[int]
    redo_completed: bool


def run_stack_instance(run_parameters: RunParameters, dataset: Dataset,
                       context: LexEarleyAlgorithmContext,
                       tokenizer: PreTrainedTokenizer, model: GenerationMixin,
                       warmup: bool= False):

    save_path = run_parameters.results_path / str(run_parameters.data_index) / str(run_parameters.cut_index) \
                / str(run_parameters.seed)
    if not warmup:
        save_path.mkdir(parents=True, exist_ok=True)

    if not warmup and not run_parameters.redo_completed and (save_path / "unconstrained").exists():
        return  # Allow resuming when interrupted without re-doing work

    chosen_dataset_content = dataset[run_parameters.data_index]["content"]

    # Want to get a cut dependent on the cut index, but not the seed
    random.seed(hash((run_parameters.data_index, run_parameters.cut_index)) % (2 ** 32))
    if run_parameters.nice_cuts:
        prefix, middle, suffix = cut_text_same_indentation_levels(chosen_dataset_content + "\n")
    else:
        prefix, middle, suffix = cut_text_random(chosen_dataset_content, 0, .9, .2, 100)
        suffix = suffix + "\n"

    transformers.set_seed(hash((run_parameters.data_index, run_parameters.cut_index, run_parameters.seed)) % (2 ** 32))

    if not warmup:
        (save_path / "prefix").write_text(prefix)
        (save_path / "ground_truth").write_text(middle)
        (save_path / "suffix").write_text(suffix)

    pre_input_ids, pre_attention_mask = tokenizer_int64(tokenizer, prefix)
    post_input_ids, post_attention_mask = tokenizer_int64(tokenizer, suffix)

    input_ids, attention_mask = create_balanced_context(
        pre_input_ids=pre_input_ids, pre_attention_mask=pre_attention_mask,
        post_input_ids=post_input_ids, post_attention_mask=post_attention_mask,
        tokenizer=tokenizer, max_generation_length=run_parameters.max_tokens, device=run_parameters.device
    )

    new_output_text_constrained, full_constrained, pre_time, full_constrained_time, constrained_overhead_per_tok_times, \
        constrained_overhead_per_eval_times = \
        do_prefix_suffix_constrained_generation(
            tokenizer=tokenizer, model=model, context=context,
            input_ids=input_ids, attention_mask=attention_mask,
            prefix_text=prefix, suffix_text=suffix,
            pre_input_ids=pre_input_ids, post_input_ids=post_input_ids,
            beam_size=run_parameters.beam_size, max_generation_length=run_parameters.max_tokens,
            debug=False, num_token_finding_attempts=run_parameters.num_token_finding_attempts
        )

    constrained_overhead_p50, constrained_overhead_p90, constrained_overhead_mean, constrained_overhead_num = get_p50_p90_mean_count(constrained_overhead_per_tok_times)
    constrained_overhead_eval_p50, constrained_overhead_eval_p90, constrained_overhead_eval_mean, constrained_overhead_eval_num = get_p50_p90_mean_count(constrained_overhead_per_eval_times)

    if not warmup:
        if new_output_text_constrained is not None:
            (save_path / "constrained").write_text(new_output_text_constrained)

        if full_constrained is not None:
            (save_path / "full_constrained").write_text(full_constrained)
        else:
            (save_path / "constrained_max_num_attempts").touch()

    unconstrained_start = datetime.datetime.now()

    outputs_unconstrained = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                           max_new_tokens=run_parameters.max_tokens,
                                           num_beams=run_parameters.beam_size, num_return_sequences=1,
                                           early_stopping=True)

    unconstrained_end = datetime.datetime.now()

    len_input_tokens = input_ids.shape[1]

    new_output_tokens_unconstrained = outputs_unconstrained[0][len_input_tokens:]
    new_output_text_unconstrained = tokenizer.decode(new_output_tokens_unconstrained, skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=False)

    only_correct_unconstrained, unconstrained_overhead_times = try_incremental_unconstrained(
        prefix, new_output_text_unconstrained, suffix)

    unconstrained_overhead_p50, unconstrained_overhead_p90, unconstrained_overhead_mean, unconstrained_overhead_num = get_p50_p90_mean_count(unconstrained_overhead_times)

    if not warmup:
        if only_correct_unconstrained is not None:
            (save_path / "unconstrained_checked").write_text(only_correct_unconstrained)

        (save_path / "stats.json").write_text(json.dumps({
            "pre_time": pre_time.total_seconds(),
            "total_constrained_time": full_constrained_time.total_seconds(),
            "constrained_overhead_p50": constrained_overhead_p50,
            "constrained_overhead_p90": constrained_overhead_p90,
            "constrained_overhead_mean": constrained_overhead_mean,
            "constrained_overhead_num": constrained_overhead_num,  # Should equal num_constrained_tokens, but just making sure
            "constrained_overhead_eval_p50": constrained_overhead_eval_p50,
            "constrained_overhead_eval_p90": constrained_overhead_eval_p90,
            "constrained_overhead_eval_mean": constrained_overhead_eval_mean,
            "constrained_overhead_eval_num": constrained_overhead_eval_num,
            "num_constrained_tokens": len(tokenizer.encode(full_constrained)) if full_constrained is not None else None,
            "num_constrained_chars": len(full_constrained) if full_constrained is not None else None,
            "num_constrained_output_chars": len(new_output_text_constrained) if new_output_text_constrained is not None else None,
            "num_unconstrained_tokens": len(new_output_tokens_unconstrained),
            "num_unconstrained_chars": len(new_output_text_unconstrained),
            "num_unconstrained_output_chars": len(only_correct_unconstrained) if only_correct_unconstrained is not None else None,
            "unconstrained_generation_time": (unconstrained_end - unconstrained_start).total_seconds(),
            "unconstrained_checking_overhead_p50": unconstrained_overhead_p50,
            "unconstrained_checking_overhead_p90": unconstrained_overhead_p90,
            "unconstrained_checking_overhead_mean": unconstrained_overhead_mean,
            "unconstrained_checking_overhead_num": unconstrained_overhead_num,  # Should equal num unconstrained characters
            **get_and_reset_stats()}, indent=1))

        (save_path / "unconstrained").write_text(new_output_text_unconstrained)


def run_stack(program_parameters: ProgramParameters):
    program_parameters.results_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(program_parameters.model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(program_parameters.model_name, trust_remote_code=True) \
        .to(program_parameters.device)

    dataset = datasets.load_dataset(program_parameters.dataset_name, data_dir=program_parameters.dataset_path)["train"]
    earley_context = get_python_context()

    has_warmed_up = False

    for data_index in range(program_parameters.min_data_idx, program_parameters.max_data_idx + 1):
        pyparse_err_path = program_parameters.results_path / f"{data_index}.pyparse.err"
        earleyparse_err_path = program_parameters.results_path / f"{data_index}.earleyparse.err"

        if not program_parameters.redo_completed and (pyparse_err_path.exists() or earleyparse_err_path.exists()):
            continue  # Already looked at this and it is invalid

        # If the file doesn't parse, then we can't expect our parser to work so skip it
        # Probably is Python 2
        file_content = dataset[data_index]["content"] + "\n"
        try:
            ast.parse(file_content)
        except SyntaxError as e:
            pyparse_err_path.write_text(traceback.format_exc())
            continue

        parse_result = lex_earley_parse(earley_context, file_content)

        if isinstance(parse_result, str):
            # It is parsed by python, but our parser can't get past the string in parse_result
            earleyparse_err_path.write_text(parse_result)
            continue
        elif not parse_result:
            # It is a valid file, but not complete
            earleyparse_err_path.write_text("(Parses, but incomplete)")
            continue

        for cut_idx in range(program_parameters.min_cut_idx, program_parameters.max_cut_idx + 1):

            for seed in range(program_parameters.min_seed, program_parameters.max_seed + 1):
                run_err_path = program_parameters.results_path / f"{data_index}.{cut_idx}.{seed}.err"

                run_parameters = RunParameters(
                    seed=seed,
                    cut_index=cut_idx,
                    data_index=data_index,
                    beam_size=program_parameters.beam_size,
                    results_path=program_parameters.results_path,
                    device=program_parameters.device,
                    max_tokens=program_parameters.max_tokens,
                    nice_cuts=program_parameters.nice_cuts,
                    num_token_finding_attempts=program_parameters.num_token_finding_attempts,
                    redo_completed=program_parameters.redo_completed
                )

                if not has_warmed_up:
                    # Get the juices flowing!
                    run_stack_instance(run_parameters=run_parameters, dataset=dataset, context=earley_context,
                                       tokenizer=tokenizer, model=model, warmup=True)
                    has_warmed_up = True

                if not program_parameters.redo_completed and run_err_path.exists():
                    continue

                try:
                    run_stack_instance(run_parameters=run_parameters, dataset=dataset,
                                       context=earley_context,
                                       tokenizer=tokenizer, model=model)
                except Exception as e:
                    # Likely cause: terminating search too early for a token (all candidates rejected)
                    run_err_path.write_text(traceback.format_exc())


def parse_args() -> ProgramParameters:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-seed", type=int)
    parser.add_argument("--max-seed", type=int)
    parser.add_argument("--min-data-idx", type=int)
    parser.add_argument("--max-data-idx", type=int)
    parser.add_argument("--min-cut-idx", type=int)
    parser.add_argument("--max-cut-idx", type=int)
    parser.add_argument("--beam-size", type=int)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument("--dataset-path", type=str)
    parser.add_argument("--results-path", type=Path)
    parser.add_argument("--device", type=str)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--nice-cuts", action='store_true')
    parser.add_argument("--redo-completed", action='store_true')
    parser.add_argument("--num-token-finding-attempts", type=int, default=None)
    args = parser.parse_args()

    return ProgramParameters(**vars(args))


def run():
    params = parse_args()
    run_stack(params)


if __name__ == "__main__":
    run()
