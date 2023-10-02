from typing import Optional, Dict


class TokenProbabilityTrieNode:
    """
    Keep track of cumulative probability, given marginal probabilities of each token
    """
    def __init__(self, running_sum_logprob: float, hypothesis_length: int):
        self.running_sum_logprob = running_sum_logprob
        self.children: Dict[int, TokenProbabilityTrieNode] = {}
        self.running_eof_sum_logprob: Optional[float] = None
        self.hypothesis_length = hypothesis_length

    def add_child(self, child_token: int, child_logprob: float) -> "TokenProbabilityTrieNode":
        self.children[child_token] = TokenProbabilityTrieNode(
            self.running_sum_logprob + child_logprob, self.hypothesis_length + 1)
        return self.children[child_token]

    def set_eof_probability(self, eof_sum_logprob: float):
        self.running_eof_sum_logprob = eof_sum_logprob + self.running_sum_logprob

    @property
    def eof_score(self) -> Optional[float]:
        if self.running_eof_sum_logprob is None:
            return None
        return self.running_eof_sum_logprob / (self.hypothesis_length + 1)
