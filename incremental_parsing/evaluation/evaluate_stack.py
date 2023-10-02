import ast
import csv
import json
import multiprocessing
import sys
import traceback
from pathlib import Path
from typing import Dict, Union, List

from tqdm import tqdm

NUM_CUTS = 10
NUM_SEEDS = 1


def parse_top_level_file(top_level_file: Path):
    """
    For a given data index, compute all the results under this index
    """
    results: List[Dict[str, Union[str, int, None]]]
    try:
        if top_level_file.is_file():
            filename_parts = top_level_file.name.split(".")
            if len(filename_parts) == 3 and filename_parts[1] in ("pyparse", "earleyparse") \
                    and filename_parts[2] == "err":
                results = []
                for cut_num in range(NUM_CUTS):
                    for seed_num in range(NUM_SEEDS):
                        results.append({
                            "data_idx": int(filename_parts[0]),
                            "cut_idx": cut_num,
                            "seed_idx": seed_num,
                            "parse_orig": 0 if filename_parts[1] == "pyparse" else 1,
                            "parse_earley": 0,
                            "successful_generation": 0,
                            "parse_constrained": 0,
                            "parse_unconstrained": 0,
                            "differs": 0
                        })

                return results

            elif len(filename_parts) == 4 and filename_parts[3] == "err":
                return [{
                    "data_idx": int(filename_parts[0]),
                    "cut_idx": int(filename_parts[1]),
                    "seed_idx": int(filename_parts[2]),
                    "parse_orig": 1,
                    "parse_earley": 1,
                    "successful_generation": 1,
                    "parse_constrained": 0,
                    "parse_unconstrained": 0,
                    "differs": 0
                }]
            else:
                raise Exception("Unexpected filename format:  " + top_level_file.name)
        else:
            results = []
            data_idx = int(top_level_file.name)
            for cut_path in top_level_file.iterdir():
                cut_idx = int(cut_path.name)
                for seed_path in cut_path.iterdir():
                    seed_idx = int(seed_path.name)

                    prefix = (seed_path / "prefix").read_text()
                    suffix = (seed_path / "suffix").read_text()

                    constrained = seed_path / "constrained"
                    unconstrained = seed_path / "unconstrained"
                    unconstrained_checked = seed_path / "unconstrained_checked"

                    error_message = None

                    is_complete = unconstrained.exists()
                    if not is_complete:
                        continue  # Allow us to use this script during active generation

                    prefix_suffix_constrained = None
                    if not constrained.exists():
                        # This "full_constrained" refers to constrained generation which might
                        # be cut off at a wrong spot
                        assert (seed_path / "full_constrained").exists()
                        constrained_parse = False
                        constrained_clipped_generation = False
                    else:
                        constrained_text = constrained.read_text()
                        prefix_suffix_constrained = prefix + constrained_text + suffix
                        constrained_clipped_generation = True

                        try:
                            ast.parse(prefix_suffix_constrained)
                            constrained_parse = True
                        except SyntaxError as e:
                            error_message = f"{e.__class__.__name__}: {e.msg}"
                            constrained_parse = False

                    unconstrained_text = unconstrained.read_text()
                    prefix_suffix_unconstrained = prefix + unconstrained_text + suffix

                    try:
                        ast.parse(prefix_suffix_unconstrained)
                        unconstrained_parse = True
                    except (SyntaxError, MemoryError):
                        unconstrained_parse = False

                    unconstrained_checked_exists = unconstrained_checked.exists()

                    if (seed_path / "stats.json").exists():
                        stats = json.loads((seed_path / "stats.json").read_text())
                    else:
                        stats = {}

                    results.append({
                        "data_idx": data_idx,
                        "cut_idx": cut_idx,
                        "seed_idx": seed_idx,
                        "parse_orig": 1,
                        "parse_earley": 1,
                        "successful_generation": 1,
                        "successful_clipped_generation": int(constrained_clipped_generation),
                        "parse_constrained": int(constrained_parse),
                        "parse_unconstrained": int(unconstrained_parse),
                        "parse_unconstrained_checked": int(unconstrained_checked_exists),
                        "differs": int(prefix_suffix_constrained != prefix_suffix_unconstrained),
                        "error_message": error_message,
                        "prefix_size": len(prefix),
                        "suffix_size": len(suffix),
                        **stats
                    })

            return results
    except BaseException as e:
        return traceback.format_exc()


def stack_results(results_dir: Path, summary_csv: Path):
    with summary_csv.open("w") as f, multiprocessing.Pool(20) as pool:
        writer = csv.DictWriter(f, fieldnames=["data_idx", "cut_idx", "seed_idx", "parse_orig", "parse_earley",
                                               "successful_generation", "successful_clipped_generation",
                                               "parse_constrained", "parse_unconstrained",
                                               "parse_unconstrained_checked",
                                               "differs", "num_branches_in_middle",
                                               "num_branches_after_first_suffix_lexeme",
                                               "error_message",
                                               "pre_time", "total_constrained_time", "num_constrained_tokens",
                                               "num_unconstrained_tokens", "unconstrained_generation_time",
                                               "constrained_overhead_p50", "constrained_overhead_p90",
                                               "constrained_overhead_eval_p50", "constrained_overhead_eval_p90",
                                               "unconstrained_checking_overhead_p50",
                                               "unconstrained_checking_overhead_p90",
                                               "prefix_size", "suffix_size"])
        writer.writeheader()

        results = pool.imap_unordered(parse_top_level_file, results_dir.iterdir(), chunksize=20)

        for result in tqdm(results):
            if isinstance(result, str):
                print(result)
                exit(1)

            for r in result:
                result_plus_defaults = {
                    "num_branches_in_middle": None,
                    "num_branches_after_first_suffix_lexeme": None,
                    "successful_clipped_generation": 0,
                    "error_message": None,
                    **r
                }
                writer.writerow(result_plus_defaults)


if __name__ == "__main__":
    stack_results(Path(sys.argv[1]),
                  Path(sys.argv[2]))
