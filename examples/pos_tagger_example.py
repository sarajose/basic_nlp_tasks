"""
POS Tagger Usage Example

This example demonstrates how to use the POSTagger class from NLPlib to tag sentences.
It shows how to load an existing model and vocabulary, and then tag English sentences.
"""
import os
import pickle
import tensorflow as tf
from NLPlib.POS_tagger import POSTagger

def main():
    # Create instance of POS tagger
    tagger = POSTagger()
    
    # Paths to model and vocabulary files
    model_path = os.path.join("models", "english_pos_tagger.h5")
    vocab_path = os.path.join("data", "english_vocab.pkl")
    
    # Check if model and vocabulary files exist
    if not os.path.exists(model_path) or not os.path.exists(vocab_path):
        print(f"Error: Model file {model_path} or vocabulary file {vocab_path} not found")
        print("Make sure you have the required model and vocabulary files")
        return
    
    # Load the model
    print("Loading POS tagger model...")
    tagger.model = tf.keras.models.load_model(model_path)
    
    # Load the vocabulary
    print("Loading vocabulary...")
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)
        tagger.word2idx = vocab_data['word2idx']
        tagger.tag2idx = vocab_data['tag2idx']
        tagger.max_len = vocab_data['max_len']
    
    # Define some example sentences
    sentences = [
        "The cat sat on the mat.",
        "She walked quickly to the store.",
        "Python programming is fun and educational."
    ]
    
    # Tag each sentence and print the results
    print("\nPOS Tagging Results:")
    print("--------------------")
    for i, sentence in enumerate(sentences, 1):
        print(f"\nSentence {i}: {sentence}")
        
        # Tag the sentence
        tags = tagger.predict_sentence(sentence)
        
        # Print words with their tags
        words = sentence.split()
        for word, tag in zip(words, tags):
            print(f"{word}: {tag}")

if __name__ == "__main__":
    main()
