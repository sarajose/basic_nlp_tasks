# NLPlib - A collection of basic NLP tools and models

# Import main components for easier access
from NLPlib.n_gram_model import NGramModel
from NLPlib.essay_score_prediction import EssayScorePredictorBase, EssayScorePredictorSKLearn, EssayScorePredictorTorch
from NLPlib.POS_tagger import POSTagger
from NLPlib.word_segmenter import WordSegmenter
from NLPlib.text_search import search, TextSearch, WordVectors

# Version
__version__ = '0.1.0'