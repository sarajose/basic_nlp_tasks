"""
Word Segmentation Example

This example demonstrates how to use the WordSegmenter class from NLPlib
to segment text that doesn't have spaces between words (like in some Asian languages).
"""
import json
import os
from NLPlib.word_segmenter import WordSegmenter

def main():
    # Create a simple lexicon for testing
    lexicon = {
        "apple": 0.02,
        "banana": 0.01,
        "orange": 0.015,
        "pear": 0.005,
        "grape": 0.01,
        "fruit": 0.03,
        "salad": 0.008,
        "mix": 0.02,
        "fresh": 0.025,
        "juice": 0.018,
        "sweet": 0.015,
        "sour": 0.01,
        "taste": 0.02,
        "good": 0.03,
        "bad": 0.01,
        "a": 0.1,
        "the": 0.12,
        "is": 0.08,
        "are": 0.06,
        "with": 0.05,
        "and": 0.09,
        "or": 0.04,
        "in": 0.07,
        "on": 0.05,
        "to": 0.08,
        "of": 0.09,
        "for": 0.06
    }
    
    # Save lexicon to a temporary file
    lexicon_path = os.path.join("examples", "temp_lexicon.json")
    os.makedirs(os.path.dirname(lexicon_path), exist_ok=True)
    with open(lexicon_path, 'w') as f:
        json.dump(lexicon, f)
    
    # Create word segmenter with the lexicon
    segmenter = WordSegmenter(lexicon_path)
    
    # Example texts to segment
    texts = [
        "applesandoranges",
        "freshfruitjuice",
        "abananaisagoodfruit",
        "orangeandapplemix",
        "thesweetandsourtaste"
    ]
    
    # Segment and print results
    print("Word Segmentation Results:")
    print("--------------------------")
    for text in texts:
        segments = segmenter.segment(text)
        print(f"Original: {text}")
        print(f"Segmented: {' '.join(segments)}")
        print()
    
    # Clean up the temporary lexicon file
    if os.path.exists(lexicon_path):
        os.remove(lexicon_path)
        print(f"Cleaned up temporary file: {lexicon_path}")

if __name__ == "__main__":
    main()
