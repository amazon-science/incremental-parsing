from typing import Optional, List, Dict, Tuple

from transformers import PreTrainedTokenizer

from incremental_parsing.generation.probability_trie import TokenProbabilityTrieNode
from incremental_parsing.lex_earley.lex_earley import LexEarleyAlgorithmState, is_completable, \
    LexEarleyAlgorithmContext, lex_earley_init, \
    LexEarleyAlgorithmPrefixState, lex_earley_step, is_complete, \
    lex_earley_to_middle


class EarleyTrieNode:
    """
    We don't want to do more work than needed, so we save the parse results for each token
    """
    def __init__(self, state: LexEarleyAlgorithmState, length: int, eof_indices: Tuple[int, ...],
                 parent_partial_tokens: Tuple[int, ...],
                 parent_partial_token_valid_idx: int):
        """
        :param state: Algorithm state at this node
        :param length: How many characters made up the most recent LLM token at this node
        :param eof_indices: See below
        :param parent_partial_tokens: See below
        :param parent_partial_token_valid_idx: See below
        """
        self.state = state
        self.valid = (state is not None) and is_completable(state)
        self.children: Dict[int, "EarleyTrieNode"] = {}
        self.to_middle_state_child: Dict[str, EarleyTrieNode] = {}
        self.length = length

        """
        eof_indices = []       this token is not an end-of-file
        eof_indices = [0]      this token is a valid EOF
        eof_indices = [-1, 0]  this token is multiple characters. It is a valid EOF ending on this token, _and_ 
                               a valid EOF if we end one character before
        eof_indices = [-1]     this token is multiple characters. It is _not_ a valid EOF ending on this token, but
                               a valid EOF if we end one character before
        """
        self.eof_indices = eof_indices

        """
        Sometimes a token decoded by itself doesn't yield valid unicode.
        For example, with santacoder, decode([17351]) == " �", while decode([17351, 100]) == " 旧"
        Clearly, 17351 can't stand by itself. 
        We can parse the " ", but when we see unicode FFFD (�), we should wait until the next token to see
        what the validity is. We set parent_partial_tokens = [17351]. 
        Because we did parse one character, the space, we set parent_partial_token_valid_idx = 1.
        """
        self.parent_partial_tokens = parent_partial_tokens
        self.parent_partial_token_valid_idx = parent_partial_token_valid_idx

    def next_token(self, token: int, tokenizer: PreTrainedTokenizer, context: LexEarleyAlgorithmContext) -> Optional[
        "EarleyTrieNode"]:
        """
        If the next token has already been parsed, just return that node.
        Otherwise, invoke the parser.
        This is the main location where the LLM interfaces with the actual parser.
        """
        if not self.valid:
            self.children[token] = self
            return self

        if token in self.children:
            return self.children[token]

        full_token_text = tokenizer.decode(self.parent_partial_tokens + (token,), skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)

        if len(full_token_text) == 0:
            next_valid_idx = 0
            next_partial_tokens = self.parent_partial_tokens + (token,)
            token_text = ""
        elif full_token_text[-1] == chr(0xFFFD):
            next_valid_idx = len(full_token_text) - 1
            token_text = full_token_text[self.parent_partial_token_valid_idx:next_valid_idx]
            next_partial_tokens = self.parent_partial_tokens + (token,)
        else:
            token_text = full_token_text[self.parent_partial_token_valid_idx:]
            next_valid_idx = 0
            next_partial_tokens = ()

        valid_eof_idx = []

        current_state = self.state

        for idx_rel_to_end, char in zip(range(-len(token_text) + 1, 1), token_text):
            current_state = lex_earley_step(context, current_state, char)
            if is_complete(context, current_state):
                valid_eof_idx.append(idx_rel_to_end)

        self.children[token] = EarleyTrieNode(current_state, len(token_text), tuple(valid_eof_idx),
                                              parent_partial_tokens=next_partial_tokens,
                                              parent_partial_token_valid_idx=next_valid_idx)
        return self.children[token]

    def to_middle_state(self, context: LexEarleyAlgorithmContext, suffix: str, lookahead: str = "") -> "EarleyTrieNode":
        """
        If we have computed the quotient language for this suffix before, return that, otherwise, compute the quotient
        language and cache it
        """
        assert isinstance(self.state, LexEarleyAlgorithmPrefixState)
        assert self.parent_partial_tokens == ()
        assert self.parent_partial_token_valid_idx == 0

        if suffix not in self.to_middle_state_child:
            middle_parser_state = lex_earley_to_middle(context=context, state=self.state,
                                                       suffix=suffix, middle_lookahead=lookahead)
            self.to_middle_state_child[suffix] = EarleyTrieNode(middle_parser_state, 0,
                                                                (0,) if is_complete(context,
                                                                                    middle_parser_state) else (),
                                                                parent_partial_tokens=(),
                                                                parent_partial_token_valid_idx=0)
        return self.to_middle_state_child[suffix]


class LexEarleyWorker:
    """
    Holds both the Earley Trie (a view from an incremental parsing perspective) and the Probability Trie
    (a view from a LLM generation perspective).
    Is useful for various operations that require mixing the two
    """
    def __init__(self, context: LexEarleyAlgorithmContext, tokenizer: PreTrainedTokenizer):
        self.init_state = lex_earley_init(context)
        self.root_trie = EarleyTrieNode(self.init_state,
                                        length=0,
                                        eof_indices=(0,) if is_complete(context, self.init_state) else (),
                                        parent_partial_tokens=(),
                                        parent_partial_token_valid_idx=0)
        self.probability_trie = TokenProbabilityTrieNode(0.0, 0)
        self.tokenizer = tokenizer
        self.context = context
        self.num_ignored_tokens = 0

    def check_toks(self, prefix: List[int], possible_generations: List[int],
                   scores: List[float],
                   eof_score: float) -> Tuple[List[bool], bool, List[bool]]:
        """
        :param prefix: Token ids that came before
        :param possible_generations: New leaf tokens to check
        :param scores: From the LLM
        :param eof_score: Also from LLM
        :return: Whether each token is valid, plus if the EOF is valid
        """

        trie_node = self.get_trie_node(prefix[self.num_ignored_tokens:])
        eof_valid = len(trie_node.eof_indices) != 0

        probability_trie_node, max_parent_eof_prob = self.get_probability_trie_node(prefix[self.num_ignored_tokens:])
        probability_trie_node.set_eof_probability(eof_score)

        results = []
        redundant_branches = []

        for possible_tok, score in zip(possible_generations, scores):
            suffix_node = trie_node.next_token(possible_tok, self.tokenizer, self.context)
            next_prob_node = probability_trie_node.add_child(possible_tok,
                                                             score if suffix_node.valid else float("-inf"))
            assert next_prob_node.running_sum_logprob <= probability_trie_node.running_sum_logprob, \
                f"{next_prob_node.running_sum_logprob} <= {probability_trie_node.running_sum_logprob}"

            results.append(suffix_node.valid)
            redundant_branches.append(suffix_node.valid and next_prob_node.running_sum_logprob <= max_parent_eof_prob)

        return results, eof_valid, redundant_branches

    def get_probability_trie_node(self, prefix_tokens) -> Tuple[TokenProbabilityTrieNode, float]:
        """
        :return: The node, plus the max eof probability of a parent token
        """
        trie_node = self.probability_trie
        if trie_node.running_eof_sum_logprob is None:
            assert len(prefix_tokens) == 0
            return trie_node, 0.0

        max_eof_prob = trie_node.running_eof_sum_logprob

        for i, tok in enumerate(prefix_tokens):
            trie_node = trie_node.get_child_or_default(tok)

            if trie_node.running_eof_sum_logprob is None:
                assert i == len(prefix_tokens) - 1
                return trie_node, max_eof_prob

            max_eof_prob = max(max_eof_prob, trie_node.running_eof_sum_logprob)

        return trie_node, max_eof_prob

    def get_trie_node(self, prefix):
        trie_node = self.root_trie
        for i, prefix_tok in enumerate(prefix):
            next_trie_node = trie_node.next_token(prefix_tok, self.tokenizer, self.context)
            if False and ((next_trie_node is None) or not next_trie_node.valid):  # Better to just swallow it
                raise ValueError(f"""
                Invalid prefix: {self.tokenizer.decode(prefix[:i + 1])}
                Remaining: {self.tokenizer.decode(prefix[i + 1:])}""")
            trie_node = next_trie_node  # So that mypy is happy about trie_node's type
        return trie_node

    def get_stopping_points(self, sequence: List[int]) -> List[Tuple[int, float]]:

        # Note that the input is a sequence of token ids, but returned int is a string index

        node = self.probability_trie
        parsing_node = self.root_trie

        stopping_points = []

        # assert parsing_node.length == 0
        # assert parsing_node.eof_indices == () or parsing_node.eof_indices == (0,)

        sequence = list(sequence)

        assert node.running_eof_sum_logprob is not None
        if 0 in parsing_node.eof_indices:
            stopping_points.append((0, node.running_eof_sum_logprob))

        end_string_idx = 0

        for tok in sequence:
            if tok == self.tokenizer.eos_token_id:
                break

            assert tok in node.children
            node = node.children[tok]
            parsing_node = parsing_node.children[tok]

            end_string_idx += parsing_node.length

            if node.running_eof_sum_logprob is None:
                # If the LLM was cut off due to some stopping criteria (such as max length,
                # or due to the running probability getting too low),
                # the last token (or sometimes two) won't have an EOF probability.
                # It is a pain to actually calculate that probability at this point, so we treat it as -inf
                break

            if parsing_node.eof_indices != () and node.eof_score is not None:
                stopping_points.extend((end_string_idx + eof_idx, node.eof_score)
                                       for eof_idx in parsing_node.eof_indices)

        return stopping_points

    def clear(self):
        self.root_trie = EarleyTrieNode(self.init_state, 0, (0,) if is_complete(self.context, self.init_state) else (),
                                        parent_partial_tokens=(),
                                        parent_partial_token_valid_idx=0)
        self.probability_trie = TokenProbabilityTrieNode(0.0, 0)

    def set_prefix(self, prefix: List[int]):
        self.root_trie = EarleyTrieNode(
            lex_earley_init(self.context, self.tokenizer.decode(prefix, skip_special_tokens=True,
                                                                clean_up_tokenization_spaces=False)),
            length=0,
            eof_indices=(),
            parent_partial_tokens=(),
            parent_partial_token_valid_idx=0)
        self.root_trie = self.get_trie_node(prefix)

    def set_prefix_suffix(self, prefix: List[int], suffix: List[int]):
        self.set_prefix(prefix)
        suffix_text = self.tokenizer.decode(suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        self.root_trie = self.root_trie.to_middle_state(context=self.context,
                                                        suffix=suffix_text,
                                                        lookahead="")

    def ignore_k_toks(self, k: int):
        self.num_ignored_tokens = k
