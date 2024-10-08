import datetime
import itertools
from typing import List, Tuple, Optional

import torch
from transformers import LogitsProcessor

from incremental_parsing.generation.lex_earley_worker import LexEarleyWorker


class MaxNumAtempts(Exception):
    pass


class LexEarleyLogitsProcessor(LogitsProcessor):
    def __init__(self, worker: LexEarleyWorker, beam_size: int, eof_id: int, debug: bool = False, max_iter_attempts: Optional[int] = None):
        self.beam_size = beam_size
        self.worker = worker
        self.eof_id = eof_id
        self.debug = debug
        self.overall_token_times: List[datetime.timedelta] = []
        self.per_evaluation_times: List[datetime.timedelta] = []
        self.max_iter_attempts = max_iter_attempts

        # The following hack accounts for santacoder being weird about its pad IDs
        pad_token_input = worker.tokenizer("<fim-pad>").input_ids
        if len(pad_token_input) == 1 and pad_token_input[0] in worker.tokenizer.all_special_ids:
            self.pad_token_id = pad_token_input[0]
        else:
            self.pad_token_id = worker.tokenizer.pad_token_id

    def get_and_reset_times(self) -> Tuple[List[datetime.timedelta], List[datetime.timedelta]]:
        ot, pe = self.overall_token_times, self.per_evaluation_times
        self.overall_token_times = []
        self.per_evaluation_times = []
        return ot, pe

    def __call__(self, input_ids, scores):
        assert len(scores.shape) == 2

        if self.pad_token_id is not None:
            scores[:, self.pad_token_id] = float("-inf")

        scores_log_softmax = torch.log_softmax(scores, dim=1)

        eof_scores = scores[:, self.eof_id].cpu().numpy().tolist()
        eof_scores_log_softmax = scores_log_softmax[:, self.eof_id].cpu().numpy().tolist()
        scores[:, self.eof_id] = float("-inf")

        total_toks = 0
        result = torch.full_like(scores, float("-inf"))

        time_start = datetime.datetime.now()

        for _ in (range(self.max_iter_attempts) if self.max_iter_attempts is not None else itertools.count()):
            # Keep going until we have enough tokens that the parser likes
            top_k = torch.topk(scores, self.beam_size, dim=1)
            top_k_softmax = torch.topk(scores_log_softmax, self.beam_size, dim=1)
            top_k_log_softmax_values_cpu = top_k_softmax.values.cpu().numpy().tolist()

            input_ids_cpu = input_ids.cpu().numpy().tolist()
            top_k_indices_cpu = top_k.indices.cpu().numpy().tolist()

            total_eof_valid = 0

            for i in range(scores.shape[0]):
                per_evaluation_start_time = datetime.datetime.now()
                results, eof_valid, redundant_branches = self.worker.check_toks(prefix=input_ids_cpu[i],
                                                                                possible_generations=top_k_indices_cpu[
                                                                                    i],
                                                                                scores=top_k_log_softmax_values_cpu[i],
                                                                                eof_score=eof_scores_log_softmax[i])

                results_gpu = torch.as_tensor(results, dtype=torch.bool, device=scores.device)
                result[i][top_k.indices[i]] = torch.where(results_gpu, top_k.values[i], float("-inf"))

                if eof_valid:
                    result[i, self.eof_id] = eof_scores[i]
                    total_eof_valid += 1

                total_toks += sum(results)

                self.per_evaluation_times.append(datetime.datetime.now() - per_evaluation_start_time)

                if self.debug:
                    print(self.worker.tokenizer.decode(input_ids[i][self.worker.num_ignored_tokens:]))
                    print(f"({sum(results)})")

            # Hopefully we don't get stuck here?
            if total_toks + total_eof_valid < (1 if self.beam_size == 1 else self.beam_size * 2):
                if self.debug:
                    print("Not enough allowable tokens, generating more")
                for i in range(scores.shape[0]):
                    scores[i][top_k.indices[i]] = float("-inf")
            else:
                break
        else:
            # Max num iterations reached, just send an EOF
            if self.debug:
                raise MaxNumAtempts()
            result[:,self.eof_id] = 1

        if self.debug:
            print("-----")

        self.overall_token_times.append(datetime.datetime.now() - time_start)

        return result
