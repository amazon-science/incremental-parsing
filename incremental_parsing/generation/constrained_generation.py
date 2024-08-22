import datetime
from typing import Optional, Tuple, Union, Sequence

import torch
from transformers import PreTrainedTokenizer, GenerationMixin, LogitsProcessorList

from incremental_parsing.generation.lex_earley_worker import LexEarleyWorker
from incremental_parsing.generation.logits_processor import LexEarleyLogitsProcessor, MaxNumAtempts
from incremental_parsing.generation.utils import tokenizer_int64, create_balanced_context, color_idx
from incremental_parsing.lex_earley.lex_earley import LexEarleyAlgorithmContext, lex_earley_init, lex_earley_run, \
    is_complete, LexEarleyAlgorithmState


def extract_best_middle(earley_worker: LexEarleyWorker,
                        tokenizer: PreTrainedTokenizer,
                        new_output_tokens_constrained: torch.LongTensor,
                        context: LexEarleyAlgorithmContext,
                        prefix_text: str,
                        suffix_text: str,
                        debug: Union[bool, str] = False) -> Tuple[Optional[str], str]:
    """
    Even if the LLM hasn't generated an EOS, we can go back through all the text that was created,
    and pick the token for which an EOS is valid, which has the highest EOS probability.
    Note that the text we return might not be on a token boundary.
    For example, when generating `def foo(|> None: pass\n` where `|` represents the insertion point, you would expect
    the generation to be `) -`.
    However, the LLM will often generate the parentheses, then incorrectly choose the token which represents `->`,
    leading to `def foo() ->> None`.
    Therefore, we check if each prefix of the token is valid as well,
    and just adopt the EOS probability of the whole token.

    Setting debug=True lets you see all the possible stopping points;
    debug="filtered" is slightly less verbose and only includes stopping points which are at the end of an
    increasing run of EOS scores.
    """
    all_stopping_points = earley_worker.get_stopping_points(
        new_output_tokens_constrained.cpu().numpy().tolist())
    full_output_text_constrained = tokenizer.decode(new_output_tokens_constrained, skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False)

    if debug:
        prev_stopping_point = 0
        output_text = []
        score_output = []

        if debug == "filtered":
            filtered_stopping_points = []

            if len(all_stopping_points) > 0:
                prev_run_stopping_point, prev_run_max_score = all_stopping_points[0]
                seen_increase = True

                for stopping_point, score in all_stopping_points[1:]:
                    if score < prev_run_max_score and seen_increase:
                        filtered_stopping_points.append((prev_run_stopping_point, prev_run_max_score))
                        seen_increase = False
                    else:
                        seen_increase = True

                    prev_run_max_score = score
                    prev_run_stopping_point = stopping_point

                if seen_increase:
                    filtered_stopping_points.append((prev_run_stopping_point, prev_run_max_score))
        else:
            filtered_stopping_points = all_stopping_points

        if filtered_stopping_points and filtered_stopping_points[0][0] == 0:
            score_output.append(f"(Empty string) : {filtered_stopping_points[0][1]:.3}")

        for idx, (stopping_point, score) in enumerate(filtered_stopping_points):
            if stopping_point != 0:
                output_text.append(full_output_text_constrained[prev_stopping_point:stopping_point - 1])
                output_text.append(color_idx(idx)(full_output_text_constrained[stopping_point - 1:stopping_point]))

                score_output.append(color_idx(idx)(full_output_text_constrained[prev_stopping_point:stopping_point]) +
                                    f" : {score:.4}")

                prev_stopping_point = stopping_point

        print("".join(output_text) + full_output_text_constrained[prev_stopping_point:])
        print("Scores:")
        print("\n".join(score_output))

    if all_stopping_points:
        # Get the latest match if multiple are equal
        best_stopping_point, _ = max(all_stopping_points, key=lambda x: (x[1], x[0]))

        # There is a syntactically valid stopping point at some point
        # (excluding the last token when generation gets cut off)
        return full_output_text_constrained[:best_stopping_point], full_output_text_constrained
    else:
        if debug:
            print("No stopping point found")
        if int(new_output_tokens_constrained[-1]) == tokenizer.eos_token_id:
            # The LLM didn't get cut off; it just gave us an EOS at an invalid point
            return None, full_output_text_constrained
        else:
            # The LLM got cut off, but by sheer luck, it may have been cut off a syntactically valid point.
            # See if this is the case
            earley_state: LexEarleyAlgorithmState = lex_earley_init(context)
            earley_state = lex_earley_run(context, earley_state,
                                          prefix_text + full_output_text_constrained + suffix_text)
            if is_complete(context, earley_state):
                return full_output_text_constrained, full_output_text_constrained
            else:
                return None, full_output_text_constrained


def prefix_suffix_constrained_generation(tokenizer: PreTrainedTokenizer,
                                         model: GenerationMixin,
                                         context: LexEarleyAlgorithmContext,
                                         prefix_text: str,
                                         suffix_text: str,
                                         beam_size: int,
                                         max_generation_length: int,
                                         device: str,
                                         num_token_finding_attempts: Optional[int] = None,
                                         debug: Union[bool, str] = False) -> \
        Tuple[Optional[str], str, datetime.timedelta, datetime.timedelta, Sequence[datetime.timedelta],
        Sequence[datetime.timedelta]]:
    """
    Helper function that tokenizes everything and _then_ does constrained generation
    (if also doing unconstrained generation, doesn't make sense to repeat work,
    so we don't use this function all the time)
    """
    pre_input_ids, pre_attention_mask = tokenizer_int64(tokenizer, prefix_text)
    post_input_ids, post_attention_mask = tokenizer_int64(tokenizer, suffix_text)

    input_ids, attention_mask = create_balanced_context(
        pre_input_ids=pre_input_ids, pre_attention_mask=pre_attention_mask,
        post_input_ids=post_input_ids, post_attention_mask=post_attention_mask,
        tokenizer=tokenizer, max_generation_length=max_generation_length, device=device
    )

    return do_prefix_suffix_constrained_generation(
        tokenizer=tokenizer, model=model, context=context,
        input_ids=input_ids, attention_mask=attention_mask,
        prefix_text=prefix_text, suffix_text=suffix_text,
        pre_input_ids=pre_input_ids, post_input_ids=post_input_ids,
        beam_size=beam_size, max_generation_length=max_generation_length,
        num_token_finding_attempts=num_token_finding_attempts,
        debug=debug
    )


def do_prefix_suffix_constrained_generation(tokenizer: PreTrainedTokenizer,
                                            model: GenerationMixin,
                                            context: LexEarleyAlgorithmContext,
                                            input_ids: torch.Tensor,
                                            attention_mask: torch.Tensor,
                                            prefix_text: str,
                                            suffix_text: str,
                                            pre_input_ids: torch.Tensor,
                                            post_input_ids: torch.Tensor,
                                            beam_size: int,
                                            max_generation_length: int,
                                            num_token_finding_attempts: Optional[int] = None,
                                            debug: Union[bool, str] = False
                                            ) -> \
        Tuple[Optional[str], str, datetime.timedelta, datetime.timedelta, Sequence[datetime.timedelta],
        Sequence[datetime.timedelta]]:
    """
    The "Core" constrained generation function

    It creates and initializes all necessary parser objects, then calls the model with this parser in tow

    Returns
    - text: code such that if you take prefix_text + text + suffix_text and parse it, it "should" be valid.
      - May not exist.
    - full_text: Regardless of text, the entire model output. May be longer than text, and text is a prefix of full_text
    - after_prefix_suffix_time - start_time: Time it takes to parse the prefix and suffix
    - end_time - after_prefix_suffix_time: Time it takes for constrained generation (including the model evaluation)
    - tok_times: Time to generate each token. Note that many LLM tokens may be evaluated just to generate one
    - eval_times: Time to evaluate each LLM token with the incremental parser.
      - Likely more eval_times then the number of tokens in the output text.
      - Still includes calling the parser on multiple characters per token
    """
    len_input_tokens = input_ids.shape[1]
    earley_worker = LexEarleyWorker(context=context, tokenizer=tokenizer)
    earley_logits_processor = LexEarleyLogitsProcessor(worker=earley_worker, beam_size=beam_size,
                                                       eof_id=tokenizer.eos_token_id,
                                                       max_iter_attempts=num_token_finding_attempts)
    start_time = datetime.datetime.now()
    earley_worker.set_prefix_suffix(pre_input_ids[0].numpy().tolist(), post_input_ids[0].numpy().tolist())
    after_prefix_suffix_time = datetime.datetime.now()
    earley_worker.ignore_k_toks(len_input_tokens)
    try:
        outputs_constrained = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                         max_new_tokens=max_generation_length,
                                         logits_processor=LogitsProcessorList([earley_logits_processor]),
                                         # stopping_criteria=StoppingCriteriaList([stopping_criteria]),
                                         num_beams=beam_size, num_return_sequences=1, early_stopping=True)
        new_output_tokens_constrained = outputs_constrained[0][len_input_tokens:]
        text, full_text = extract_best_middle(earley_worker=earley_worker,
                                              tokenizer=tokenizer,
                                              new_output_tokens_constrained=new_output_tokens_constrained,
                                              context=context,
                                              prefix_text=prefix_text,
                                              suffix_text=suffix_text,
                                              debug=debug)
    except MaxNumAtempts:
        text = None
        full_text = None

    end_time = datetime.datetime.now()
    tok_times, eval_times = earley_logits_processor.get_and_reset_times()

    return (text, full_text, after_prefix_suffix_time - start_time, end_time - after_prefix_suffix_time,
            tok_times, eval_times)


def unconstrained_generation(tokenizer: PreTrainedTokenizer,
                             model: GenerationMixin,
                             prefix_text: str,
                             suffix_text: str,
                             beam_size: int,
                             max_new_tokens: int,
                             device: str) -> str:
    """
    Helper function to do unconstrained generation, even if there are more than the model's max tokens present
    """
    pre_input_ids, pre_attention_mask = tokenizer_int64(tokenizer, prefix_text)
    post_input_ids, post_attention_mask = tokenizer_int64(tokenizer, suffix_text)

    input_ids, attention_mask = create_balanced_context(
        pre_input_ids=pre_input_ids, pre_attention_mask=pre_attention_mask,
        post_input_ids=post_input_ids, post_attention_mask=post_attention_mask,
        tokenizer=tokenizer, max_generation_length=max_new_tokens, device=device
    )
    len_input_tokens = input_ids.shape[1]

    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                             max_new_tokens=max_new_tokens,
                             num_beams=beam_size, num_return_sequences=1, early_stopping=True)

    return tokenizer.decode(outputs[0][len_input_tokens:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
