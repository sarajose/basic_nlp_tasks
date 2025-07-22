import unittest
from NLPlib.POS_tagger import POSTagger

def simple_pos_example():
    """
    A simple example showing how to use the POSTagger for basic part-of-speech tagging.
    
    This example assumes you have downloaded the model files from the repository
    and they are located in the 'models' directory.
    """
    # For English language
    english_model_path = 'models/english_pos_tagger.h5'
    
    # Initialize the tagger
    pos_tagger = POSTagger()
    
    # Load the model - model will be loaded when first prediction is made
    # The model file should be in the specified path
    
    # Example sentence for tagging
    sentence = "This is a simple example for part-of-speech tagging."
    
    try:
        # Get the tags
        tags = pos_tagger.predict_sentence(sentence)
        
        # Print the words and their tags
        words = sentence.split()
        print("Word\tTag")
        print("-" * 20)
        for word, tag in zip(words, tags):
            print(f"{word}\t{tag}")
            
        return True
    except Exception as e:
        print(f"Error predicting tags: {e}")
        return False

if __name__ == "__main__":
    success = simple_pos_example()
    if success:
        print("\nPOS tagging example completed successfully.")
    else:
        print("\nPOS tagging example failed. Make sure model files are available.")
