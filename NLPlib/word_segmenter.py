import json
import math
import sys

class WordSegmenter:
    """
    A word segmenter using a unigram language model with fallback
    to single-character segmentation for unknown words.
    """

    def __init__(self, lexicon_file):
        """
        Initializes the WordSegmenter with a given lexicon JSON file.

        Args:
            lexicon_file (str): The path to the lexicon JSON file that contains
                                word frequencies ({"a": 100, "b": 90, ...}).
        """
        with open(lexicon_file, 'r') as f:
            self.lexicon = json.load(f)

        # Calculate total word frequency and store log value for reuse
        self.total_words = sum(self.lexicon.values()) or 1
        self.log_total_words = math.log(self.total_words)
        self.memo = {}

        # Maximum length of substring to consider as a single word
        self.max_word_length = 5

    def segment(self, text):
        """
        Segments the input text into words using a dynamic programming approach.

        For each position i in the text, it stores the highest log probability
        of any segmentation up to that position. If a substring (text[j:i]) is
        in the lexicon, it uses its frequency from the lexicon; otherwise, it
        splits that substring into single-character words.

        Args:
            text (str): The input text to be segmented, with no whitespace.

        Returns:
            list: A list of words that form the highest log-probability segmentation.
        """
        # Use memoized result if available
        if text in self.memo:
            return self.memo[text]

        n = len(text)
        best_segment = [None] * (n + 1)
        best_score = [float('-inf')] * (n + 1)
        best_score[0] = 0  # Base case: empty string has log-prob 0

        for i in range(1, n + 1):
            # Check substrings ending at i, up to max_word_length in length
            for j in range(max(0, i - self.max_word_length), i):
                word = text[j:i]

                if word in self.lexicon:
                    # Word is known: use its frequency
                    word_freq = self.lexicon[word]
                    score = best_score[j] + math.log(word_freq) - self.log_total_words
                else:
                    # Word is unknown: treat it as multiple single-char words
                    score = best_score[j]
                    for k in range(j, i):
                        char = text[k]
                        char_freq = self.lexicon.get(char, 1)
                        score += math.log(char_freq) - self.log_total_words

                if score > best_score[i]:
                    best_score[i] = score
                    best_segment[i] = (j, word)

        # Reconstruct the segmentation from the back
        words = []
        i = n
        while i > 0:
            if best_segment[i] is None:
                # If no segment found, take one character
                words.append(text[i-1:i])
                i -= 1
            else:
                j, word = best_segment[i]
                if word in self.lexicon:
                    words.append(word)
                else:
                    # Split unknown substring into individual characters
                    for char in word:
                        words.append(char)
                i = j

        words.reverse()
        self.memo[text] = words
        return words

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 word_segmenter.py lexicon.json input.txt >segmented.txt")
        sys.exit(1)

    lexicon_file = sys.argv[1]
    input_file = sys.argv[2]

    with open(input_file, 'r') as f:
        text = f.read().strip()

    segmenter = WordSegmenter(lexicon_file)
    segmented_text = segmenter.segment(text)

    with open('output.txt', 'w') as f:
        f.write(" ".join(segmented_text))