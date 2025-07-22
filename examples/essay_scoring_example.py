"""
Essay Score Prediction Example

This example demonstrates how to:
1. Extract features from essays using EssayScorePredictorBase
2. Create and train a simple essay scoring model using both scikit-learn and PyTorch backends
3. Predict scores for new essays
"""

from NLPlib.essay_score_prediction import (
    EssayScorePredictorBase, 
    EssayScorePredictorSKLearn,
    EssayScorePredictorTorch
)

def main():
    # Sample essays with corresponding scores
    training_data = [
        ("This is a short essay about topic A. It has few words and limited vocabulary.", 2),
        ("This essay discusses topic A in more depth. It includes several sentences and has a bit more vocabulary. The ideas are expressed clearly.", 3),
        ("In this comprehensive essay about topic A, I will explore multiple aspects. The essay contains substantial content with varied vocabulary. It presents arguments, examples, and thoughtful analysis of the subject matter.", 4),
        ("Short text about B.", 1),
        ("Longer text about topic B with a few more words in it.", 2),
        ("Detailed exploration of topic B with rich vocabulary and insightful commentary throughout the entire text.", 4)
    ]
    
    # Validation data
    validation_data = [
        ("An essay on topic A with moderate length and vocabulary.", 3),
        ("Brief comment on B.", 1)
    ]
    
    # Extract and display features using the base class
    print("Feature extraction demonstration:")
    for essay, score in training_data[:2]:  # Just show the first two for brevity
        length_feature = EssayScorePredictorBase.calculate_length_4th_root(essay)
        ovix_feature = EssayScorePredictorBase.calculate_ovix(essay)
        print(f"\nEssay: \"{essay[:30]}...\"")
        print(f"  Actual score: {score}")
        print(f"  Length (4th root): {length_feature:.4f}")
        print(f"  OVIX: {ovix_feature:.4f}")
    
    # Train scikit-learn model
    print("\nTraining scikit-learn model...")
    sklearn_predictor = EssayScorePredictorSKLearn(learning_rate=0.01, epochs=500)
    sklearn_predictor.train(training_data, validation_data)
    
    # Train PyTorch model
    print("\nTraining PyTorch model...")
    pytorch_predictor = EssayScorePredictorTorch(learning_rate=0.01, epochs=500)
    pytorch_predictor.train(training_data, validation_data)
    
    # Test predictions on new essays
    test_essays = [
        "This is a new short essay that hasn't been seen before.",
        "This is a much longer essay that provides detailed information about the topic. It uses varied vocabulary and presents multiple viewpoints on the subject matter. The arguments are structured logically, and the essay includes examples to support the main points."
    ]
    
    print("\nPredicting scores for new essays:")
    for i, essay in enumerate(test_essays):
        sklearn_score = sklearn_predictor.predict(essay)
        pytorch_score = pytorch_predictor.predict(essay)
        
        print(f"\nEssay {i+1}: \"{essay[:30]}...\"")
        print(f"  scikit-learn prediction: {sklearn_score:.2f}")
        print(f"  PyTorch prediction: {pytorch_score:.2f}")

if __name__ == "__main__":
    main()
