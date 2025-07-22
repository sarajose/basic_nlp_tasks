"""
Essay Score Prediction Example

This example demonstrates how to use the EssayScorePredictor classes
from NLPlib to calculate features from essays and make predictions.
"""
from NLPlib.essay_score_prediction import EssayScorePredictorBase, EssayScorePredictorSKLearn

def main():
    # Sample essays with different characteristics
    essays = [
        # Short, simple essay
        "This is a short essay. It has few words and simple sentences.",
        
        # Medium length, more varied vocabulary
        "The application of natural language processing techniques to educational assessment has gained significant traction in recent years. " +
        "Automated essay scoring systems utilize various linguistic features to evaluate the quality of student writing. " +
        "These systems analyze aspects such as grammatical correctness, vocabulary diversity, and coherence.",
        
        # Long essay with rich vocabulary
        "The implementation of computational methodologies for evaluating written discourse presents numerous advantages " +
        "in educational contexts. Sophisticated algorithms can now analyze multifaceted aspects of composition, including " +
        "lexical diversity, syntactic complexity, coherence, and adherence to rhetorical conventions. The evolution of " +
        "these technologies has been facilitated by advancements in machine learning and natural language processing. " +
        "Educational institutions increasingly recognize the potential of these systems to provide consistent, objective " +
        "assessment metrics while simultaneously reducing the workload of instructors. Nevertheless, critics maintain " +
        "that automated evaluation systems may fail to appreciate the nuanced creativity and unconventional brilliance " +
        "that characterizes exceptional writing. The ongoing refinement of these systems seeks to address such limitations " +
        "through increasingly sophisticated feature extraction techniques and hybrid models that incorporate human oversight."
    ]
    
    # Create base feature calculator
    base_calculator = EssayScorePredictorBase()
    
    # Print feature calculations for each essay
    print("Essay Features:")
    print("--------------")
    
    for i, essay in enumerate(essays, 1):
        print(f"\nEssay {i}:")
        print(f"Length: {len(essay.split())} words")
        
        # Calculate features
        length_root = base_calculator.calculate_length_4th_root(essay)
        ovix = base_calculator.calculate_ovix(essay)
        
        print(f"Length 4th root: {length_root:.4f}")
        print(f"OVIX (vocabulary diversity): {ovix:.4f}")
    
    # Create a simple prediction model using default features
    print("\nTraining a simple prediction model...")
    
    # Training data (sample essays and scores)
    training_essays = essays
    training_scores = [2.5, 3.8, 4.7]  # Example scores on a 5-point scale
    
    # Create and train the model
    model = EssayScorePredictorSKLearn()
    model.train(training_essays, training_scores)
    
    # Make predictions on new essays
    test_essay = "The analysis of linguistic features in academic writing reveals patterns that correlate with quality assessments."
    
    print(f"\nTest Essay: {test_essay}")
    prediction = model.predict([test_essay])[0]
    print(f"Predicted score: {prediction:.2f} / 5.0")

if __name__ == "__main__":
    main()
