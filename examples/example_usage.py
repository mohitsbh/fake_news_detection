#!/usr/bin/env python
"""
Example usage of the Fake News Detection System.

This script demonstrates how to use the FakeNewsPredictor class
to analyze news articles for fake news detection.
"""

import os
import sys
import json

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import FakeNewsPredictor

def print_separator():
    """Print a separator line."""
    print("\n" + "-" * 80 + "\n")

def example_text_prediction():
    """Example of predicting fake news from text."""
    print("Example 1: Analyzing text input")
    print_separator()
    
    # Initialize predictor
    predictor = FakeNewsPredictor()
    
    # Example text with fake news characteristics
    fake_text = """
    BREAKING: Scientists shocked as miracle cure for all diseases discovered! 
    Big Pharma doesn't want you to know about this amazing natural remedy that 
    cures cancer, diabetes, and heart disease overnight! Share this before it gets deleted!
    """
    
    # Example text with real news characteristics
    real_text = """
    A recent study published in the Journal of Medical Research found that regular exercise 
    may reduce the risk of heart disease by up to 30 percent. The research, conducted over 
    a five-year period with 2,000 participants, suggests that even moderate physical activity 
    can have significant health benefits. Dr. Sarah Johnson, lead researcher, recommends at 
    least 150 minutes of exercise per week.
    """
    
    # Analyze fake news example
    print("Analyzing potential fake news:")
    fake_result = predictor.predict(text=fake_text, explain=True)
    print(f"Prediction: {fake_result['prediction'].upper()}")
    print(f"Confidence: {fake_result['confidence']} ({fake_result['probability']:.2%})")
    print("\nExplanation:")
    print(fake_result['explanation']['summary'])
    
    print_separator()
    
    # Analyze real news example
    print("Analyzing potential real news:")
    real_result = predictor.predict(text=real_text, explain=True)
    print(f"Prediction: {real_result['prediction'].upper()}")
    print(f"Confidence: {real_result['confidence']} ({real_result['probability']:.2%})")
    print("\nExplanation:")
    print(real_result['explanation']['summary'])

def example_url_prediction():
    """Example of predicting fake news from a URL."""
    print("\nExample 2: Analyzing news from URL")
    print_separator()
    
    # Initialize predictor
    predictor = FakeNewsPredictor()
    
    # Example URL (replace with an actual news URL)
    url = "https://www.reuters.com/world/us/"
    
    print(f"Analyzing news from URL: {url}")
    try:
        result = predictor.predict(url=url, explain=True)
        print(f"Prediction: {result['prediction'].upper()}")
        print(f"Confidence: {result['confidence']} ({result['probability']:.2%})")
        print("\nExplanation:")
        print(result['explanation']['summary'])
    except Exception as e:
        print(f"Error analyzing URL: {str(e)}")
        print("Note: URL analysis requires internet connection and may not work with all websites.")

def example_linguistic_analysis():
    """Example of detailed linguistic analysis."""
    print("\nExample 3: Detailed Linguistic Analysis")
    print_separator()
    
    # Initialize predictor
    predictor = FakeNewsPredictor()
    
    # Example text
    text = """
    SHOCKING TRUTH REVEALED: Top scientists admit global warming is a HOAX designed 
    to control the population! Anonymous insider leaks SECRET documents proving 
    the conspiracy! The mainstream media is HIDING this from you!!!
    """
    
    print("Analyzing text for linguistic markers:")
    print(text)
    print()
    
    # Get prediction with explanation
    result = predictor.predict(text=text, explain=True)
    
    # Print linguistic analysis
    print("Linguistic Analysis:")
    linguistic = result['explanation']['linguistic_analysis']
    
    print("\nFake News Markers:")
    for category, terms in linguistic['fake_news_markers'].items():
        category_name = " ".join(word.capitalize() for word in category.split("_"))
        print(f"  {category_name}: {', '.join(terms)}")
    
    print("\nCredibility Indicators:")
    if linguistic['credibility_markers']:
        for category, terms in linguistic['credibility_markers'].items():
            category_name = " ".join(word.capitalize() for word in category.split("_"))
            print(f"  {category_name}: {', '.join(terms)}")
    else:
        print("  None detected")
    
    print("\nVerification Tips:")
    for tip in result['explanation']['verification_tips']:
        print(f"  • {tip}")

def example_json_output():
    """Example of getting results in JSON format."""
    print("\nExample 4: JSON Output Format")
    print_separator()
    
    # Initialize predictor
    predictor = FakeNewsPredictor()
    
    # Example text
    text = "Scientists have discovered a new species of deep-sea fish in the Mariana Trench."
    
    # Get prediction with explanation
    result = predictor.predict(text=text, explain=True)
    
    # Print as formatted JSON
    print("Result as JSON:")
    print(json.dumps(result, indent=2))

def main():
    """Run all examples."""
    print("Fake News Detection System - Usage Examples")
    print("===========================================\n")
    
    try:
        # Run examples
        example_text_prediction()
        example_linguistic_analysis()
        example_json_output()
        
        # URL example may fail if no internet connection
        # or if the website blocks scraping
        try:
            example_url_prediction()
        except Exception as e:
            print(f"\nURL example failed: {str(e)}")
    
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        print("\nNote: Make sure you have trained models available or use the default models.")
        print("You can train models using the src/models/train.py script.")

if __name__ == "__main__":
    # Create examples directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    main()