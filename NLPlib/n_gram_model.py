from collections import defaultdict, Counter
from typing import Iterable, Tuple
import math

class NGramModel:
    
    def __init__(self, n: int, k: float, sentences: Iterable[Tuple[str]]):
        """
        Constructor n-gram model.
        Parameters:
        n: The order of the n-gram
        k: The smoothing parameter
        sentences: A collection of sentences (each sentence is tuple of words)
        """
        self.n = n  # The order of the n-gram
        self.k = k  # The smoothing parameter
        self.ngrams = defaultdict(Counter)  # dict: keys (tuple of words) - values (Counters of words)
        self.context_counts = defaultdict(int)  # dict: keys (tuples of words) - values (frequency of words)
        self.vocabulary = set()  # Unique words

        for sentence in sentences:
            padded_sentence = (n - 1) * ['<s>'] + list(sentence) + ['</s>']
            for i in range(len(padded_sentence) - n + 1):
                context = tuple(padded_sentence[i:i + n - 1])
                word = padded_sentence[i + n - 1]
                self.ngrams[context][word] += 1
                self.context_counts[context] += 1
                # Add word to vocabulary
                self.vocabulary.add(word) 

    def p(self, word: str, context: Tuple[str]) -> float:
        """
        Calculate the probability of a word given its context using k smoothing.
        Args:
            word (str): The word for which the probability is to be calculated.
            context (Tuple[str]): The context in which the word appears.
        Returns:
            float: The probability of the word given the context.
        """
        context_count = self.context_counts[context]
        word_count = self.ngrams[context][word]
        vocabulary_size = len(self.vocabulary)
        return (word_count + self.k) / (context_count + self.k * vocabulary_size)

    def score(self, sentence: Tuple[str]) -> float:
        """
        Calculate the log-probability of a given sentence using an n-gram model.
        Args:
            sentence (Tuple[str]): A tuple of words representing the sentence to score.
        Returns:
            float: The log-prob of the sentence.
        """
        padded_sentence = (self.n - 1) * ['<s>'] + list(sentence) + ['</s>']
        log_prob = 0.0
        for i in range(len(padded_sentence) - self.n + 1):
            context = tuple(padded_sentence[i:i + self.n - 1])
            word = padded_sentence[i + self.n - 1]
            log_prob += math.log(self.p(word, context))
        return log_prob