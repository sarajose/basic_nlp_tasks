"""
N-Gram Model Usage Example

This example demonstrates how to use the NGramModel class from NLPlib
to create a language model that can calculate probabilities and score sentences.
"""
from NLPlib.n_gram_model import NGramModel

def main():
    # Sample corpus for training the N-gram model
    corpus = [
        "I love programming in Python",
        "Python is a versatile programming language",
        "Language models are useful for many NLP tasks",
        "I am learning about natural language processing",
        "Many modern applications use machine learning"
    ]
    
    # Process corpus into token lists
    token_corpus = [sentence.split() for sentence in corpus]
    
    # Create N-gram models with different n values
    print("Creating bigram model...")
    bigram_model = NGramModel(n=2, k=1.0, sentences=token_corpus)  # bigram model with smoothing
    
    print("Creating trigram model...")
    trigram_model = NGramModel(n=3, k=1.0, sentences=token_corpus)  # trigram model with smoothing
    
    # Example sentences to score
    test_sentences = [
        "I love programming in Python",  # In the corpus
        "Python is versatile for programming",  # Similar to corpus
        "Zebras enjoy dancing tango underwater"  # Not similar to corpus
    ]
    
    # Calculate scores and probabilities with bigram model
    print("\nBigram Model Results:")
    print("--------------------")
    for sentence in test_sentences:
        # Convert to tokens and calculate sentence score (log probability)
        tokens = sentence.split()
        score = bigram_model.score(tuple(tokens))
        print(f"Sentence: '{sentence}'")
        print(f"Bigram score: {score:.4f}")
        
        # Calculate probability of specific words following others
        words = sentence.split()
        for i in range(len(words) - 1):
            prob = bigram_model.p(words[i+1], (words[i],))
            print(f"P({words[i+1]} | {words[i]}) = {prob:.4f}")
        print()
    
    # Calculate scores with trigram model
    print("\nTrigram Model Results:")
    print("--------------------")
    for sentence in test_sentences:
        # Convert to tokens and calculate sentence score (log probability)
        tokens = sentence.split()
        score = trigram_model.score(tuple(tokens))
        print(f"Sentence: '{sentence}'")
        print(f"Trigram score: {score:.4f}")
        
        # Calculate probability of specific words following others
        words = sentence.split()
        for i in range(len(words) - 2):
            prob = trigram_model.p(words[i+2], (words[i], words[i+1]))
            print(f"P({words[i+2]} | {words[i]} {words[i+1]}) = {prob:.4f}")
        print()

if __name__ == "__main__":
    main()
