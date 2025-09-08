#!/usr/bin/env python
"""
Evaluation utilities for the fake news detection system.

This module provides functions to evaluate the performance of
fake news detection models using various metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class ModelEvaluator:
    """Class for evaluating fake news detection models."""
    
    def __init__(self, model_name=""):
        """Initialize the evaluator.
        
        Args:
            model_name (str): Name of the model being evaluated.
        """
        self.model_name = model_name
        self.metrics = {}
    
    def evaluate(self, y_true, y_pred, y_prob=None):
        """Evaluate model performance.
        
        Args:
            y_true (array-like): True labels.
            y_pred (array-like): Predicted labels.
            y_prob (array-like, optional): Predicted probabilities for the positive class.
                                          Required for ROC and PR curves.
        
        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        # Basic metrics
        self.metrics['accuracy'] = accuracy_score(y_true, y_pred)
        self.metrics['precision'] = precision_score(y_true, y_pred, average='binary', pos_label='fake')
        self.metrics['recall'] = recall_score(y_true, y_pred, average='binary', pos_label='fake')
        self.metrics['f1'] = f1_score(y_true, y_pred, average='binary', pos_label='fake')
        
        # Confusion matrix
        self.metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        # Classification report
        self.metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
        
        # ROC and PR curves if probabilities are provided
        if y_prob is not None:
            # Convert labels to binary (0/1) for ROC curve calculation
            y_true_binary = np.array([1 if label == 'fake' else 0 for label in y_true])
            
            # ROC curve
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob)
            self.metrics['roc_auc'] = auc(fpr, tpr)
            self.metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
            
            # Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_true_binary, y_prob)
            self.metrics['average_precision'] = average_precision_score(y_true_binary, y_prob)
            self.metrics['pr_curve'] = {'precision': precision.tolist(), 'recall': recall.tolist()}
        
        return self.metrics
    
    def print_summary(self):
        """Print a summary of the evaluation metrics."""
        if not self.metrics:
            print("No evaluation metrics available. Run evaluate() first.")
            return
        
        print(f"\n{'=' * 50}")
        print(f"Model Evaluation Summary: {self.model_name}")
        print(f"{'=' * 50}\n")
        
        print(f"Accuracy: {self.metrics['accuracy']:.4f}")
        print(f"Precision: {self.metrics['precision']:.4f}")
        print(f"Recall: {self.metrics['recall']:.4f}")
        print(f"F1 Score: {self.metrics['f1']:.4f}")
        
        if 'roc_auc' in self.metrics:
            print(f"ROC AUC: {self.metrics['roc_auc']:.4f}")
        
        if 'average_precision' in self.metrics:
            print(f"Average Precision: {self.metrics['average_precision']:.4f}")
        
        print("\nConfusion Matrix:")
        cm = np.array(self.metrics['confusion_matrix'])
        print(f"\t\tPredicted Real\tPredicted Fake")
        print(f"Actual Real\t{cm[0][0]}\t\t{cm[0][1]}")
        print(f"Actual Fake\t{cm[1][0]}\t\t{cm[1][1]}")
    
    def plot_confusion_matrix(self, save_path=None):
        """Plot confusion matrix.
        
        Args:
            save_path (str, optional): Path to save the plot. If None, the plot is displayed.
        """
        if 'confusion_matrix' not in self.metrics:
            print("Confusion matrix not available. Run evaluate() first.")
            return
        
        cm = np.array(self.metrics['confusion_matrix'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Real', 'Fake'],
                   yticklabels=['Real', 'Fake'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - {self.model_name}')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_roc_curve(self, save_path=None):
        """Plot ROC curve.
        
        Args:
            save_path (str, optional): Path to save the plot. If None, the plot is displayed.
        """
        if 'roc_curve' not in self.metrics:
            print("ROC curve data not available. Run evaluate() with y_prob parameter.")
            return
        
        fpr = np.array(self.metrics['roc_curve']['fpr'])
        tpr = np.array(self.metrics['roc_curve']['tpr'])
        roc_auc = self.metrics['roc_auc']
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_name}')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_precision_recall_curve(self, save_path=None):
        """Plot precision-recall curve.
        
        Args:
            save_path (str, optional): Path to save the plot. If None, the plot is displayed.
        """
        if 'pr_curve' not in self.metrics:
            print("Precision-Recall curve data not available. Run evaluate() with y_prob parameter.")
            return
        
        precision = np.array(self.metrics['pr_curve']['precision'])
        recall = np.array(self.metrics['pr_curve']['recall'])
        avg_precision = self.metrics['average_precision']
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f'Precision-Recall Curve - {self.model_name}')
        plt.legend(loc="lower left")
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def save_metrics(self, file_path):
        """Save metrics to a JSON file.
        
        Args:
            file_path (str): Path to save the metrics.
        """
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        metrics_json = {}
        for key, value in self.metrics.items():
            if isinstance(value, np.ndarray):
                metrics_json[key] = value.tolist()
            else:
                metrics_json[key] = value
        
        with open(file_path, 'w') as f:
            json.dump(metrics_json, f, indent=4)

def compare_models(evaluators, metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'], save_path=None):
    """Compare multiple models based on selected metrics.
    
    Args:
        evaluators (list): List of ModelEvaluator instances.
        metrics (list): List of metrics to compare.
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    # Prepare data for plotting
    model_names = [evaluator.model_name for evaluator in evaluators]
    data = {}
    
    for metric in metrics:
        data[metric] = []
        for evaluator in evaluators:
            if metric in evaluator.metrics:
                data[metric].append(evaluator.metrics[metric])
            else:
                data[metric].append(0)  # Default value if metric not available
    
    # Create DataFrame
    df = pd.DataFrame(data, index=model_names)
    
    # Plot
    plt.figure(figsize=(12, 8))
    ax = df.plot(kind='bar', figsize=(12, 8))
    plt.title('Model Comparison')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def evaluate_model_on_dataset(model, X_test, y_test, model_name=""):
    """Evaluate a model on a test dataset.
    
    Args:
        model: Trained model with predict and predict_proba methods.
        X_test: Test features.
        y_test: True labels.
        model_name (str): Name of the model.
    
    Returns:
        ModelEvaluator: Evaluator with computed metrics.
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Get probabilities for the positive class (fake news)
    y_prob = None
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]  # Assuming second column is for 'fake'
    
    # Create evaluator and compute metrics
    evaluator = ModelEvaluator(model_name)
    evaluator.evaluate(y_test, y_pred, y_prob)
    
    return evaluator

if __name__ == "__main__":
    # Example usage
    print("This module provides utilities for evaluating fake news detection models.")
    print("Import and use the ModelEvaluator class in your evaluation scripts.")