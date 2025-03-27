# NLPlib

NLPlib is a Python library that provides basic NLP tools and models including:

- **Essay Score Prediction:** Estimates an essay score using a basic statistical method.
- **POS Tagging:** Tags each word in a sentence with its part of speech using a pre-trained model.
- **N-Gram Language Modeling:** Builds language models based on n-gram probabilities.
- **Word Segmentation:** Breaks down continuous text into individual words.
- **Text Search:** Searches text using word embeddings.

## Installation

Clone the repository and install the package in editable mode:

```
pip install -e .
```

## Usage

Below are examples demonstrating how to use each of the modules:

### N-Gram Model

```python
from NLPlib import n_gram_model

sentences = [
    ('this', 'is', 'a', 'sentence'),
    ('another', 'example', 'sentence'),
]
model = n_gram_model.NGramModel(n=2, k=1.0, sentences=sentences)
print(model.score(('this', 'is', 'a', 'sentence')))
```

### Essay Score Prediction

```python
from NLPlib import essay_score_prediction

essay = "Your essay text goes here."
score = essay_score_prediction.EssayScorePredictorBase.calculate_length_4th_root(essay)
print("Predicted Essay Score:", score)
```

### Part-of-Speech Tagging

```python
from NLPlib import POS_tagger

# Ensure the model file is located in the 'models' directory.
pos_tagger = POS_tagger.POS_Tagger(model_path='models/pos_tagger.h5')
sentence = "This is a test sentence.".split()
tags = pos_tagger.tag(sentence)
print("POS Tags:", tags)
```

### Word Segmentation

```python
from NLPlib import word_segmenter

text = "wordsegmenterexample"
segments = word_segmenter.segment(text)
print("Segments:", segments)
```

### Text Search

```python
from NLPlib import text_search

documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
]
results = text_search.search("document", documents)
print("Search Results:", results)
```

## Running Tests

All modules include unit tests. To run the tests, execute the following command from the root of the project:

```
python -m unittest discover -s tests
```

This command will automatically discover and run tests in the `tests` directory.

## Modules Overview

- **N-Gram Model:** Constructs language models using n-gram probabilities. Useful for predictive text and analysis.
- **Essay Score Prediction:** Computes a score for essays based on text length and other features.
- **POS Tagging:** Uses a pre-trained model (provided as a `.h5` file in the `models/` directory) to assign part-of-speech tags. The accompanying confusion matrix (see below) illustrates its performance.
- **Word Segmentation:** Segments continuous text into individual words.
- **Text Search:** Implements search functionality based on word embeddings.

### Confusion Matrix Example

Below is an example confusion matrix for the English vocabulary of the POS tagger:

![Confusion Matrix](images/Figure_conf_matrix_eng.png)