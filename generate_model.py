#!/usr/bin/env python
import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.models.model import FakeNewsClassifier
from src.data.preprocess import TextPreprocessor

def main():
    """
    Train and save a fake news detection model.
    """
    print("Loading training data...")
    # Load the training data
    train_data_path = os.path.join('data', 'processed', 'synthetic_train.csv')
    train_df = pd.read_csv(train_data_path)
    
    # Preprocess the text data
    print("Preprocessing text data...")
    preprocessor = TextPreprocessor()
    train_df['processed_text'] = train_df.apply(
        lambda row: preprocessor.preprocess_text(f"{row['title']}. {row['text']}"), 
        axis=1
    )
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        train_df['processed_text'], 
        train_df['label'], 
        test_size=0.2, 
        random_state=42
    )
    
    # Create and fit the vectorizer
    print("Vectorizing text data...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    
    # Save the vectorizer
    vectorizer_path = os.path.join('models', 'vectorizer.joblib')
    print(f"Saving vectorizer to {vectorizer_path}...")
    joblib.dump(vectorizer, vectorizer_path)
    
    # Create and train the model
    print("Training the model...")
    model = FakeNewsClassifier(model_type='random_forest', vectorizer=vectorizer)
    model.fit(X_train_vec, y_train)
    
    # Evaluate the model
    print("Evaluating the model...")
    metrics = model.evaluate(X_val_vec, y_val)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    # Set feature names for feature importance
    model.set_feature_names(vectorizer.get_feature_names_out())
    
    # Save the model
    model_path = 'fake_news_model.pkl'
    model_full_path = os.path.join('models', model_path)
    print(f"Saving model to {model_full_path}...")
    model.save(model_path)
    print(f"Model saved successfully to {model_full_path}")

if __name__ == "__main__":
    main()