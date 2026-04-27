import os
import joblib
from preprocess import clean_text

class ContentModerator:
    def __init__(self, models_dir):
        # We will load the Logistic Regression model as default for inference
        self.label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
        model_path = os.path.join(models_dir, 'logistic_regression_ovr.joblib')
        
        if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"Models not found in {models_dir}. Please run train.py first.")
        
        self.vectorizer = joblib.load(vectorizer_path)
        self.model = joblib.load(model_path)
        
    def predict(self, raw_text):
        """
        Takes raw text, cleans it, extracts features, and outputs predictions
        across all 6 toxic comment categories.
        """
        # Preprocess text
        cleaned_text = clean_text(raw_text)
        
        # Extract features
        features = self.vectorizer.transform([cleaned_text])
        
        # Predict class flags (0 or 1 for each label)
        predictions = self.model.predict(features)[0]
        
        # Probabilities
        probabilities = self.model.predict_proba(features)[0]
        
        result_dict = {}
        for idx, label in enumerate(self.label_cols):
            result_dict[label] = {
                'flag': bool(predictions[idx]),
                'probability': float(probabilities[idx])
            }
            
        is_safe = not any(predictions)
            
        return {
            'text': raw_text,
            'is_safe': is_safe,
            'details': result_dict
        }

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    models_dir = os.path.join(project_root, 'models')
    
    moderator = ContentModerator(models_dir)
    
    test_sentences = [
        "This is a wonderful and beautiful day!",
        "You are an idiot and I hope you suffer.",
        "Can you please explain this concept to me?",
        "I will kill you if you post that again."
    ]
    
    for sentence in test_sentences:
        result = moderator.predict(sentence)
        print(f"Text: '{result['text']}'")
        print(f"Safe: {result['is_safe']}")
        
        print("Triggers:")
        for label, data in result['details'].items():
            if data['flag'] or data['probability'] > 0.1:
                print(f"  - {label.upper()}: {data['probability']:.2%} ({'YES' if data['flag'] else 'NO'})")
        print("-" * 50)
