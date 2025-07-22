# NLPlib Examples

This directory contains examples demonstrating how to use the different modules in the NLPlib package.

## Running the Examples

Make sure you have installed the NLPlib package:

```
pip install -e .
```

Then you can run any example with Python:

```
python examples/ngram_model_example.py
```

## Available Examples

1. **ngram_model_example.py** - Demonstrates building a language model, calculating word probabilities, and scoring sentences.

2. **essay_scoring_example.py** - Shows how to extract features from essays and train models to predict essay scores using scikit-learn and PyTorch.

3. **pos_tagging_example.py** - Illustrates part-of-speech tagging with pre-trained models (requires model files in the `models/` directory).

4. **word_segmentation_example.py** - Demonstrates word segmentation using a lexicon-based approach.

5. **text_search_example.py** - Shows basic text search functionality and advanced search using word embeddings.

## Model Files

Some examples (like POS tagging) require pre-trained model files. These should be placed in the `models/` directory. The model files are:

- `english_pos_tagger.h5` - Pre-trained model for English POS tagging
- `catalan_pos_tagger.h5` - Pre-trained model for Catalan POS tagging

## Word Embeddings

The advanced text search example requires word embedding files. These are not included with the package, but you can use embeddings from sources like:

- [FastText](https://fasttext.cc/docs/en/english-vectors.html)
- [GloVe](https://nlp.stanford.edu/projects/glove/)
- [Word2Vec](https://code.google.com/archive/p/word2vec/)

Download the embeddings and adjust the paths in the example code as needed.
