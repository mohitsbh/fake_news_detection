import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.dataset import get_dataset
from src.data.preprocess import TextPreprocessor
from src.models.model import FakeNewsModelTrainer

# Set paths
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def train_models(dataset_name='synthetic', model_types=None, vectorizer_type='tfidf', max_features=10000):
    """
    Train fake news detection models.
    
    Args:
        dataset_name (str): Name of the dataset to use
        model_types (list): List of model types to train
        vectorizer_type (str): Type of vectorizer to use
        max_features (int): Maximum number of features for vectorization
        
    Returns:
        tuple: (best_model, vectorizer, metrics)
    """
    print(f"Loading {dataset_name} dataset...")
    X_train, X_test, y_train, y_test = get_dataset(dataset_name)
    
    if X_train is None:
        print("Failed to load dataset. Using synthetic dataset instead.")
        X_train, X_test, y_train, y_test = get_dataset('synthetic')
    
    print(f"Dataset loaded. Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    
    # Preprocess text
    print("Preprocessing text data...")
    preprocessor = TextPreprocessor()
    
    # Determine text columns based on dataset structure
    if isinstance(X_train, pd.DataFrame):
        text_columns = X_train.columns.tolist()
    else:
        text_columns = ['text']
        X_train = pd.DataFrame({'text': X_train})
        X_test = pd.DataFrame({'text': X_test})
    
    # Extract features
    print(f"Extracting features using {vectorizer_type} vectorizer...")
    X_train_vec, feature_names = preprocessor.extract_features(
        X_train, text_columns, vectorizer_type=vectorizer_type, max_features=max_features
    )
    X_test_vec = preprocessor.transform_texts([preprocessor.preprocess_text(text) for text in X_test.values.flatten()])
    
    # Train models
    print("Training models...")
    if model_types is None:
        model_types = ['random_forest', 'logistic_regression', 'svm', 'naive_bayes']
    
    trainer = FakeNewsModelTrainer(vectorizer=preprocessor.vectorizer)
    trainer.train_models(X_train_vec, y_train, X_test_vec, y_test, model_types=model_types)
    
    # Get best model
    best_model_type, best_model = trainer.get_best_model(metric='f1')
    print(f"\nBest model: {best_model_type}")
    print(f"Best model F1 score: {trainer.results[best_model_type]['f1']:.4f}")
    
    # Save all models
    saved_paths = trainer.save_models()
    print("\nSaved models:")
    for model_type, path in saved_paths.items():
        print(f"  {model_type}: {path}")
    
    # Save vectorizer separately
    vectorizer_path = os.path.join(MODELS_DIR, 'vectorizer.joblib')
    joblib.dump(preprocessor.vectorizer, vectorizer_path)
    print(f"Vectorizer saved to: {vectorizer_path}")
    
    return best_model, preprocessor.vectorizer, trainer.results[best_model_type]

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train fake news detection models')
    
    parser.add_argument('--dataset', type=str, default='synthetic',
                        choices=['synthetic', 'kaggle', 'liar'],
                        help='Dataset to use for training')
    
    parser.add_argument('--models', type=str, nargs='+',
                        default=['random_forest', 'logistic_regression', 'svm', 'naive_bayes'],
                        choices=['random_forest', 'logistic_regression', 'svm', 'naive_bayes', 'gradient_boosting', 'lstm'],
                        help='Models to train')
    
    parser.add_argument('--vectorizer', type=str, default='tfidf',
                        choices=['tfidf', 'count'],
                        help='Type of vectorizer to use')
    
    parser.add_argument('--max-features', type=int, default=10000,
                        help='Maximum number of features for vectorization')
    
    parser.add_argument('--download', action='store_true',
                        help='Download dataset if not available locally')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Train models
    best_model, vectorizer, metrics = train_models(
        dataset_name=args.dataset,
        model_types=args.models,
        vectorizer_type=args.vectorizer,
        max_features=args.max_features
    )
    
    # Print evaluation metrics
    print("\nBest model evaluation metrics:")
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            print(f"  {metric}: {value:.4f}")
    
    # Print feature importance if available
    try:
        feature_importance = best_model.get_feature_importance(top_n=20)
        if feature_importance:
            print("\nTop 20 features:")
            for feature, importance in feature_importance:
                print(f"  {feature}: {importance:.4f}")
    except Exception as e:
        print(f"Could not get feature importance: {e}")