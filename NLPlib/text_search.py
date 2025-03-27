import os
import gzip
import numpy as np

class WordVectors:
    """
    Represents a collection of word vectors loaded from a gzip-compressed file

    In the first line of the file should be the vocabulary size and the word vector
        lang/word  v1 v2 ... v_dim
    """

    def __init__(self, filename: str):
        """
        Initializes the WordVectors by reading the specified gzip-compressed embedding file

        Parameters:
            filename (str): The path to the embeddings file

        Raises:
            ValueError: If the header of the file does not contain exactly two tokens
        """
        self.word2vec = {}
        self.dim = None
        # Extract language code from filename
        lang_code = os.path.basename(filename).split('.')[0]
        with gzip.open(filename, 'rt', encoding='utf-8') as f:
            header = f.readline().strip().split()
            if len(header) != 2:
                raise ValueError("Invalid header in vector file")
            # Ignore the first line (header)
            _, dim_str = header
            self.dim = int(dim_str)
            for line in f:
                parts = line.strip().split()
                if len(parts) != self.dim + 1:
                    # Skip invalid lines
                    continue  
                token = parts[0]
                # Remove the language prefix if present
                if token.startswith(lang_code + "/"):
                    word = token[len(lang_code) + 1:]
                else:
                    word = token
                word = word.lower()
                vector = np.array(parts[1:], dtype=np.float32)
                self.word2vec[word] = vector

    def make_sentence_vector(self, words: list[str]) -> np.ndarray:
        """
        Computes the mean vector for a sentence.

        The sentence is represented by the average of the word vectors for each unique,
        word in the vocabulary. If none of the words are found, a ValueError is raised

        Parameters:
            words (list[str]): A list of tokens in the sentence

        Returns:
            np.ndarray: The mean vector of shape (dim,), where dim is the dimensionality of the embeddings

        Raises:
            ValueError: If no word in the list is found in the vocabulary.
        """
        unique_words = set(word.lower() for word in words)
        vectors = [self.word2vec[w] for w in unique_words if w in self.word2vec]
        if not vectors:
            raise ValueError("None of the words are in the vocabulary")
        return np.mean(vectors, axis=0, dtype=np.float32)


class TextSearch:
    """
    Implements a text search system using multilingual word vectors

    It supports loading word vectors from multiple languages,indexing a 
    gzipped text file containing tokenized sentences, and searching for the sentence
    that best matches a given query using cosine similarity
    """

    def __init__(self, word_vectors_filenames: list[str]):
        """
        Initializes the TextSearch system by loading the word vectors for each language

        Parameters:
            word_vectors_filenames (list[str]): A list of file paths to embedding files
        """
        self.vectors_by_language = {}
        # For each language code, store a list of tuples (sentence_vector, filename, sentence)
        self.indexed_sentences = {}
        for filename in word_vectors_filenames:
            lang_code = os.path.basename(filename).split('.')[0]
            self.vectors_by_language[lang_code] = WordVectors(filename)
            self.indexed_sentences[lang_code] = []

    def index_text(self, filename: str, language_code: str, min_words: int = 1, max_words: int = None) -> None:
        """
        Reads a gzip-compressed text file and indexes its sentences.

        Each line in the file is assumed to be a tokenized sentence (words separated by whitespace).
        Only sentences that meet the minimum and (if provided) maximum word count criteria and have at least one
        word in the vocabulary are indexed.

        Parameters:
            filename (str): Path to the gzipped text file
            language_code (str): The language code corresponding to the word vectors to use
            min_words (int): Minimum number of words required for a sentence to be indexed
            max_words (int, optional): Maximum number of words allowed in a sentence to be indexed. 
            If None, no upper limit is applied

        Raises:
            ValueError: If the specified language code is not loaded
        """
        if language_code not in self.vectors_by_language:
            raise ValueError(f"Language code {language_code} not loaded")
        word_vectors = self.vectors_by_language[language_code]
        with gzip.open(filename, 'rt', encoding='utf-8') as f:
            for line in f:
                sentence = line.strip()
                if not sentence:
                    continue
                tokens = sentence.split()
                if len(tokens) < min_words:
                    continue
                if max_words is not None and len(tokens) > max_words:
                    continue
                try:
                    vec = word_vectors.make_sentence_vector(tokens)
                    # Store the sentence vector along with the text file name and sentence.
                    self.indexed_sentences[language_code].append((vec, filename, sentence))
                except ValueError:
                    continue

    def search(self, query: list[str], language_code: str, n_matches: int = 1) -> list[tuple[float, str, str]]:
        """
        Searches for the sentences with the highest cosine similarity to the query

        The query is converted to a vector using the specified language's word vectors.
        Then, the cosine similarity is computed between the query vector and every indexed sentence vector. 
        The top n_matches (sorted in descending order of similarity) are returned.

        Parameters:
            query (list[str]): A list of tokens forming the query
            language_code (str): The language code for the query
            n_matches (int): The number of top matching sentences to return.

        Returns:
            list[tuple[float, str, str]]: A list of tuples where each tuple contains:
                - cosine similarity (float)
                - filename (str) from which the sentence was indexed
                - the sentence (str) itself

        Raises:
            ValueError: If the specified language code is not loaded or the query vector has zero norm
        """
        if language_code not in self.vectors_by_language:
            raise ValueError(f"Language code {language_code} not loaded")
        query_vector = self.vectors_by_language[language_code].make_sentence_vector(query)
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            raise ValueError("Query vector has zero norm")
        candidates = []
        # Check all indexed sentences from all languages
        for sentences in self.indexed_sentences.values():
            for sent_vec, fname, sent in sentences:
                sent_norm = np.linalg.norm(sent_vec)
                if sent_norm == 0:
                    continue
                similarity = float(np.dot(query_vector, sent_vec) / (query_norm * sent_norm))
                candidates.append((similarity, fname, sent))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[:n_matches]
