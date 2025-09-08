#!/usr/bin/env python
import argparse
import sys
import json
import textwrap
from colorama import init, Fore, Style

from src.predict import FakeNewsPredictor

# Initialize colorama for cross-platform colored terminal output
init()

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Fake News Detection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python cli.py --text "This is a news article to analyze"
          python cli.py --url "https://example.com/news-article"
          python cli.py --file news_article.txt
        """)
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", type=str, help="News article text to analyze")
    input_group.add_argument("--url", type=str, help="URL of news article to analyze")
    input_group.add_argument("--file", type=str, help="Path to file containing news article text")
    
    parser.add_argument("--title", type=str, help="Title of the news article (optional)")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("--model", type=str, default="best", 
                        help="Model to use for prediction (default: best)")
    
    return parser.parse_args()

def print_header():
    """
    Print CLI header.
    """
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'Fake News Detection System':^60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print()

def print_prediction(result):
    """
    Print prediction results in a formatted way.
    """
    # Print prediction
    prediction = result["prediction"]
    probability = result["probability"]
    confidence = result["confidence"]
    
    if prediction == "real":
        prediction_color = Fore.GREEN
    else:
        prediction_color = Fore.RED
    
    print(f"Prediction: {prediction_color}{prediction.upper()}{Style.RESET_ALL}")
    print(f"Confidence: {prediction_color}{confidence}{Style.RESET_ALL} ({probability:.2%})")
    print()
    
    # Print explanation
    if "explanation" in result:
        explanation = result["explanation"]
        print(f"{Fore.YELLOW}Summary:{Style.RESET_ALL}")
        print(explanation["summary"])
        print()
        
        # Print linguistic analysis
        print(f"{Fore.YELLOW}Linguistic Analysis:{Style.RESET_ALL}")
        
        # Fake news markers
        print(f"{Fore.RED}Potential Fake News Markers:{Style.RESET_ALL}")
        fake_markers = explanation["linguistic_analysis"]["fake_news_markers"]
        if fake_markers:
            for category, terms in fake_markers.items():
                category_name = " ".join(word.capitalize() for word in category.split("_"))
                print(f"  {category_name}: {', '.join(terms)}")
        else:
            print("  None detected")
        print()
        
        # Credibility markers
        print(f"{Fore.GREEN}Credibility Indicators:{Style.RESET_ALL}")
        real_markers = explanation["linguistic_analysis"]["credibility_markers"]
        if real_markers:
            for category, terms in real_markers.items():
                category_name = " ".join(word.capitalize() for word in category.split("_"))
                print(f"  {category_name}: {', '.join(terms)}")
        else:
            print("  None detected")
        print()
        
        # Verification tips
        print(f"{Fore.YELLOW}Verification Tips:{Style.RESET_ALL}")
        for tip in explanation["verification_tips"]:
            print(f"  • {tip}")

def main():
    """
    Main function for the CLI.
    """
    args = parse_arguments()
    
    # Initialize predictor
    try:
        predictor = FakeNewsPredictor(model_type=args.model)
    except Exception as e:
        print(f"{Fore.RED}Error initializing predictor: {str(e)}{Style.RESET_ALL}")
        sys.exit(1)
    
    # Get input text
    text = None
    url = None
    
    if args.text:
        text = args.text
    elif args.url:
        url = args.url
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"{Fore.RED}Error reading file: {str(e)}{Style.RESET_ALL}")
            sys.exit(1)
    
    # Make prediction
    try:
        result = predictor.predict(
            text=text,
            url=url,
            title=args.title,
            explain=True
        )
    except Exception as e:
        print(f"{Fore.RED}Error making prediction: {str(e)}{Style.RESET_ALL}")
        sys.exit(1)
    
    # Output results
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_header()
        print_prediction(result)

if __name__ == "__main__":
    main()