"""
N-Gram Model Example

This example shows how to use the NGramModel class to:
1. Build a simple language model from a collection of sentences
2. Calculate probabilities of words given their context
3. Score complete sentences based on the model
"""

from NLPlib.n_gram_model import NGramModel

def main():
    # Create a small corpus of sentences
    sentences = [
        ('this', 'is', 'a', 'sample', 'sentence'),
        ('another', 'example', 'sentence'),
        ('language', 'models', 'predict', 'the', 'next', 'word'),
        ('predict', 'the', 'next', 'word', 'in', 'a', 'sequence'),
        ('this', 'is', 'another', 'example')
    ]
    
    print("Building a bigram model with k=1.0 smoothing...")
    # Create a bigram model with k=1.0 for smoothing
    model = NGramModel(n=2, k=1.0, sentences=sentences)
    
    # Print some statistics
    print(f"Vocabulary size: {len(model.vocabulary)}")
    
    # Test some probabilities
    contexts_to_test = [
        ('this',),
        ('example',),
        ('the',)
    ]
    
    print("\nTesting word probabilities given context:")
    for context in contexts_to_test:
        print(f"\nContext: {context}")
        # Find words that follow this context
        if context in model.ngrams:
            word_counts = model.ngrams[context]
            for word, count in word_counts.items():
                prob = model.p(word, context)
                print(f"  P({word} | {context}) = {prob:.4f}")
    
    # Test sentence scoring
    test_sentences = [
        ('this', 'is', 'a', 'sample', 'sentence'),  # Seen before
        ('this', 'is', 'another', 'sample', 'sentence'),  # Combination of seen n-grams
        ('completely', 'new', 'text')  # Unseen n-grams
    ]
    
    print("\nScoring test sentences:")
    for sentence in test_sentences:
        score = model.score(sentence)
        print(f"  Log probability of '{' '.join(sentence)}': {score:.4f}")

if __name__ == "__main__":
    main()
