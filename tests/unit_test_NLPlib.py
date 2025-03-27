import unittest
from NLPlib import n_gram_model, essay_score_prediction, POS_tagger, word_segmenter, text_search

class TestNGramModel(unittest.TestCase):
    def setUp(self):
        # Create a small corpus of sentences
        self.sentences = [
            ('this', 'is', 'a', 'test'),
            ('another', 'test', 'sentence'),
        ]
        # n=2 with a smoothing parameter, k=1.0
        self.model = n_gram_model.NGramModel(n=2, k=1.0, sentences=self.sentences)

    def test_probability(self):
        # Pick a context and test that probability is within a valid range
        prob = self.model.p('test', ('this',))
        self.assertGreaterEqual(prob, 0)
        self.assertLessEqual(prob, 1)

class TestEssayScorePrediction(unittest.TestCase):
    def test_length_4th_root(self):
        essay = "This is a simple essay."
        result = essay_score_prediction.EssayScorePredictorBase.calculate_length_4th_root(essay)
        expected = len(essay.split()) ** 0.25
        self.assertAlmostEqual(result, expected)

class TestPOSTagger(unittest.TestCase):
    def setUp(self):
        # Assuming POS_tagger has a class POS_Tagger that loads a model file
        # And a method tag(sentence) that returns list of tuples (word, tag)
        self.tagger = POS_tagger.POS_Tagger(model_path='models/pos_tagger.h5')
    
    def test_tagging(self):
        sentence = "This is a test.".split()
        tagged = self.tagger.tag(sentence)
        # simple test: we expect a list of same length as input sentence
        self.assertEqual(len(tagged), len(sentence))

class TestWordSegmenter(unittest.TestCase):
    def setUp(self):
        # Assuming word_segmenter has a function segment(text)
        self.text = "wordsegmenterexample"
    
    def test_segmentation(self):
        segments = word_segmenter.segment(self.text)
        # Example test: no segment should be empty and rejoining equals original text
        self.assertTrue(all(segments))
        self.assertEqual("".join(segments), self.text)

class TestTextSearch(unittest.TestCase):
    def setUp(self):
        # Assuming text_search has a function search(query, documents)
        self.documents = [
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
        ]
    
    def test_search(self):
        results = text_search.search("document", self.documents)
        # Test: at least two documents containing the keyword "document" are returned
        self.assertGreaterEqual(len(results), 2)

if __name__ == '__main__':
    unittest.main()