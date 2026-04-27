import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from preprocess import clean_text
data_path = "C:/Users/student/Downloads/archive/train.csv"
def train_pipeline(data_path, models_dir):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)        
    
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    if 'comment_text' not in df.columns:
        raise ValueError("Dataset must contain 'comment_text' column.")
    
    for col in label_cols:
        if col not in df.columns:
            raise ValueError(f"Dataset missing label column: {col}")
            
    print("Preprocessing text (this might take a minute)...")
    df['clean_text'] = df['comment_text'].apply(clean_text)
    
    X = df['clean_text']
    y = df[label_cols]
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Extracting features using TF-IDF (10,000 features)...")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', lowercase=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("\nTraining Logistic Regression (One-Vs-Rest)...")
    lr_model = OneVsRestClassifier(LogisticRegression(solver='liblinear', random_state=42))
    lr_model.fit(X_train_tfidf, y_train)
    
    y_pred_lr = lr_model.predict(X_test_tfidf)
    y_prob_lr = lr_model.predict_proba(X_test_tfidf)
    
    print("\n--- Logistic Regression Performance ---")
    print("Exact Match Ratio (Subset Accuracy):", accuracy_score(y_test, y_pred_lr))
    print("ROC AUC (Macro):", roc_auc_score(y_test, y_prob_lr, average='macro'))
    print(classification_report(y_test, y_pred_lr, target_names=label_cols))
    
    print("\nTraining Naive Bayes (One-Vs-Rest)...")
    nb_model = OneVsRestClassifier(MultinomialNB())
    nb_model.fit(X_train_tfidf, y_train)
    
    y_pred_nb = nb_model.predict(X_test_tfidf)
    y_prob_nb = nb_model.predict_proba(X_test_tfidf)
    
    print("\n--- Naive Bayes Performance ---")
    print("Exact Match Ratio (Subset Accuracy):", accuracy_score(y_test, y_pred_nb))
    print("ROC AUC (Macro):", roc_auc_score(y_test, y_prob_nb, average='macro'))
    print(classification_report(y_test, y_pred_nb, target_names=label_cols))
    
    # Save models
    print("\nSaving models and vectorizer...")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(models_dir, 'tfidf_vectorizer.joblib'))
    joblib.dump(lr_model, os.path.join(models_dir, 'logistic_regression_ovr.joblib'))
    joblib.dump(nb_model, os.path.join(models_dir, 'naive_bayes_ovr.joblib'))
    print("Saved successfully to", models_dir)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    # Target the new Jigsaw dataset format
    default_data_path = os.path.join(project_root, 'data', 'train.csv')
    default_models_dir = os.path.join(project_root, 'models')
    
    print("Project Root:", project_root)
    train_pipeline(default_data_path, default_models_dir)
