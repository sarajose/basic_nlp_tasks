import unittest
import json
import os
import numpy as np
import pickle
import tempfile
import tensorflow as tf

from NLPlib import n_gram_model, essay_score_prediction, POS_tagger, word_segmenter, text_search

class TestNGramModel(unittest.TestCase):
    def setUp(self):
        # Create a small corpus of sentences.
        self.sentences = [
            ('this', 'is', 'a', 'test'),
            ('another', 'test', 'sentence'),
        ]
        # n=2 with a smoothing parameter, k=1.0
        self.model = n_gram_model.NGramModel(n=2, k=1.0, sentences=self.sentences)

    def test_probability(self):
        # Pick a context and test that probability is within a valid range.
        prob = self.model.p('test', ('this',))
        self.assertGreaterEqual(prob, 0)
        self.assertLessEqual(prob, 1)
    
    def test_score(self):
        # Test the score method on a sentence
        score = self.model.score(('this', 'is', 'a', 'test'))
        self.assertIsInstance(score, float)

class TestEssayScorePrediction(unittest.TestCase):
    def test_length_4th_root(self):
        essay = "This is a simple essay."
        result = essay_score_prediction.EssayScorePredictorBase.calculate_length_4th_root(essay)
        expected = len(essay.split()) ** 0.25
        self.assertAlmostEqual(result, expected)
    
    def test_ovix(self):
        essay = "This is a simple essay with repeated words. This is a test."
        result = essay_score_prediction.EssayScorePredictorBase.calculate_ovix(essay)
        # OVIX can be negative for certain inputs, so we just check it returns a float
        self.assertIsInstance(result, float)

class TestPOSTagger(unittest.TestCase):
    def setUp(self):
        # Create paths for model and vocabulary
        self.model_path = os.path.join("models", "english_pos_tagger.h5")
        self.vocab_path = os.path.join("data", "english_vocab.pkl")
        
        # Skip if model or vocabulary doesn't exist
        if not os.path.exists(self.model_path):
            self.skipTest(f"Model file not available for testing: {self.model_path}")
        if not os.path.exists(self.vocab_path):
            self.skipTest(f"Vocabulary file not available for testing: {self.vocab_path}")
        
        # Initialize the tagger
        self.tagger = POS_tagger.POSTagger()
        
        # Load the model manually
        self.tagger.model = tf.keras.models.load_model(self.model_path)
        
        # Load the vocabulary
        with open(self.vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
            self.tagger.word2idx = vocab_data['word2idx']
            self.tagger.tag2idx = vocab_data['tag2idx']
            self.tagger.max_len = vocab_data['max_len']

    def test_tagging(self):
        # Skip if the model isn't loaded
        if not hasattr(self, 'tagger') or self.tagger.model is None:
            self.skipTest("POS tagger model not available")

        # Create a simple model for testing
        sentence = "This is a test"
        # We're just testing the interface works, not the accuracy
        try:
            tags = self.tagger.predict_sentence(sentence)
            # Just check we get some output, don't validate the specific tags
            self.assertEqual(len(tags), len(sentence.split()))
        except Exception as e:
            self.fail(f"POS tagger prediction failed: {str(e)}")
    
    def tearDown(self):
        # No need to clean up as we're using the existing vocab file
        pass

class TestWordSegmenter(unittest.TestCase):
    def setUp(self):
        # Use a fixture file stored in tests/fixtures.
        self.lexicon_file = os.path.join("tests", "fixtures", "dummy_lexicon.json")
        # Instantiate the segmenter using the lexicon file.
        self.segmenter = word_segmenter.WordSegmenter(self.lexicon_file)
        self.text = "wordsegmenterexample"
    
    def test_segmentation(self):
        segments = self.segmenter.segment(self.text)
        # Check that the combined segments contain the same characters as the original text
        self.assertEqual(sorted("".join(segments)), sorted(self.text))
        # Check that there are fewer segments than characters (some segmentation happened)
        self.assertLess(len(segments), len(self.text))
        # Check that all segments are non-empty
        self.assertTrue(all(segments))

class TestTextSearch(unittest.TestCase):
    def test_search_function(self):
        # Test the simple search function
        documents = [
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one."
        ]
        
        # Use the actual implementation from the module
        results = text_search.search("document", documents)
        
        # Verify that two documents containing "document" are returned
        self.assertEqual(len(results), 2)
        for doc in results:
            self.assertIn("document", doc.lower())

if __name__ == '__main__':
    # Allow running specific test classes or methods from command line
    unittest.main(verbosity=2)