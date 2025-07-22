"""
Word Segmentation Example

This example demonstrates how to use the WordSegmenter class to:
1. Segment continuous text into individual words using a lexicon
2. Handle unknown words by falling back to character-level segmentation
"""

from NLPlib.word_segmenter import WordSegmenter
import json
import os

def create_sample_lexicon():
    """Create a sample lexicon file for demonstration purposes"""
    lexicon = {
        "word": 100,
        "segmenter": 80,
        "example": 60,
        "hello": 50,
        "world": 45,
        "machine": 40,
        "learning": 35,
        "natural": 30,
        "language": 25,
        "processing": 20,
        "text": 15,
        "python": 10,
        "program": 5,
        "a": 90,
        "the": 85,
        "is": 75,
        "this": 70,
        "an": 65
    }
    
    # Add single characters for fallback
    for c in "abcdefghijklmnopqrstuvwxyz":
        lexicon[c] = 2
    
    # Save lexicon to file
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    lexicon_path = os.path.join(examples_dir, "sample_lexicon.json")
    
    with open(lexicon_path, "w") as f:
        json.dump(lexicon, f, indent=2)
    
    return lexicon_path

def main():
    # Create a sample lexicon
    lexicon_path = create_sample_lexicon()
    print(f"Created sample lexicon at: {lexicon_path}")
    
    # Initialize the segmenter
    segmenter = WordSegmenter(lexicon_path)
    
    # Sample texts to segment
    sample_texts = [
        "wordsegmenterexample",  # Contains words in the lexicon
        "thisisanexample",       # Contains words in the lexicon
        "helloworldpython",      # Contains words in the lexicon
        "machinelearning",       # Contains words in the lexicon
        "naturallanguageprocessing",  # Contains words in the lexicon
        "xyzabc"                 # Contains only characters, no words in lexicon
    ]
    
    # Segment each sample text
    print("\nSegmenting sample texts:")
    for text in sample_texts:
        segments = segmenter.segment(text)
        print(f"\nOriginal: {text}")
        print(f"Segmented: {' '.join(segments)}")
    
    # Clean up the sample lexicon file
    os.remove(lexicon_path)
    print(f"\nRemoved sample lexicon file: {lexicon_path}")

if __name__ == "__main__":
    main()
