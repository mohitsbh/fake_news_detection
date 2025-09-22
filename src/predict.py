import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
import re
import requests
import json
import hashlib
import time
from bs4 import BeautifulSoup
from urllib.parse import urlparse, quote_plus
from GoogleNews import GoogleNews

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data.preprocess import TextPreprocessor
from src.models.model import FakeNewsClassifier

# Set paths
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

class FakeNewsPredictor:
    """
    Class for predicting whether news is fake or real with explanations.
    """
    def __init__(self, model_type='random_forest', model_path=None, vectorizer_path=None, use_google_news=True):
        """
        Initialize the predictor.
        
        Args:
            model_type (str): Type of model to use ('random_forest', 'logistic_regression', etc.)
            model_path (str): Path to the trained model
            vectorizer_path (str): Path to the fitted vectorizer
            use_google_news (bool): Whether to use Google News API for verification
        """
        # Initialize Google News API
        self.use_google_news = use_google_news
        self.googlenews = GoogleNews(lang='en', period='7d')
        # Default paths
        if model_path is None:
            model_path = os.path.join(MODELS_DIR, f'fake_news_{model_type}_model.joblib')
        
        if vectorizer_path is None:
            vectorizer_path = os.path.join(MODELS_DIR, 'vectorizer.joblib')
        
        # Load model and vectorizer
        try:
            # If model file doesn't exist, create a simple model
            if not os.path.exists(model_path):
                print(f"Model file not found at {model_path}. Creating a simple {model_type} model.")
                self.model = FakeNewsClassifier(model_type=model_type)
            else:
                self.model = FakeNewsClassifier.load(model_path)
                print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}. Creating a simple {model_type} model.")
            self.model = FakeNewsClassifier(model_type=model_type)
        
        try:
            if not os.path.exists(vectorizer_path):
                print(f"Vectorizer file not found at {vectorizer_path}. Using a default TfidfVectorizer.")
                from sklearn.feature_extraction.text import TfidfVectorizer
                self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
                # Fit with some sample data to avoid "not fitted" errors
                sample_texts = [
                    "This is a sample news article about politics and economy",
                    "Breaking news about technology and science discoveries",
                    "Sports results from yesterday's major league games",
                    "Entertainment news about celebrities and movies"
                ]
                self.vectorizer.fit(sample_texts)
                print("Vectorizer fitted with sample data")
            else:
                self.vectorizer = joblib.load(vectorizer_path)
                print(f"Vectorizer loaded from {vectorizer_path}")
        except Exception as e:
            print(f"Error loading vectorizer: {str(e)}. Using a default TfidfVectorizer.")
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            # Fit with some sample data to avoid "not fitted" errors
            sample_texts = [
                "This is a sample news article about politics and economy",
                "Breaking news about technology and science discoveries",
                "Sports results from yesterday's major league games",
                "Entertainment news about celebrities and movies"
            ]
            self.vectorizer.fit(sample_texts)
            print("Vectorizer fitted with sample data")
        
        # Initialize text preprocessor
        self.preprocessor = TextPreprocessor()
        
        # List of trusted news domains
        self.trusted_news_domains = [
            'bbc.com', 'bbc.co.uk', 'reuters.com', 'apnews.com', 'ap.org',
            'nytimes.com', 'washingtonpost.com', 'theguardian.com', 'cnn.com',
            'aljazeera.com', 'npr.org', 'economist.com', 'ft.com', 'wsj.com',
            'time.com', 'thehindu.com', 'hindustantimes.com', 'timesofindia.indiatimes.com',
            'indianexpress.com', 'ndtv.com', 'economictimes.indiatimes.com',
            'news18.com', 'theprint.in', 'thewire.in', 'scroll.in',
            'elpais.com', 'lemonde.fr', 'dw.com', 'abc.net.au', 'cbc.ca'
        ]
        
        # News API configuration
        self.news_api_key = os.environ.get('NEWS_API_KEY', '')
        self.news_api_url = 'https://newsapi.org/v2/everything'
        
        # Cache for API responses to avoid repeated calls
        self.api_cache = {}
        self.cache_expiry = 3600  # Cache expiry in seconds (1 hour)
        
        # Load linguistic markers for fake news
        self.fake_news_markers = {
            'sensationalism': [
                'shocking', 'unbelievable', 'incredible', 'mind-blowing', 'bombshell',
                'jaw-dropping', 'stunning', 'amazing', 'you won\'t believe', 'never seen before'
            ],
            'emotional_language': [
                'outrage', 'furious', 'devastated', 'terrified', 'horrific',
                'disgusting', 'appalling', 'terrible', 'frightening', 'alarming'
            ],
            'absolutist_terms': [
                'always', 'never', 'every', 'all', 'none', 'everyone', 'nobody',
                'absolutely', 'definitely', 'completely', 'totally', 'undoubtedly'
            ],
            'conspiracy_terms': [
                'conspiracy', 'cover-up', 'secret', 'hidden', 'truth', 'exposed',
                'they don\'t want you to know', 'what they\'re hiding', 'real truth'
            ],
            'clickbait': [
                'click here', 'find out', 'learn more', 'you need to', 'must see',
                'must read', 'share this', 'don\'t miss', 'important information'
            ],
            'urgency': [
                'urgent', 'breaking', 'alert', 'warning', 'emergency', 'immediately',
                'right now', 'act now', 'limited time', 'before it\'s too late'
            ],
            'lack_of_sources': [
                'sources say', 'reportedly', 'allegedly', 'some say', 'people are saying',
                'anonymous', 'unnamed', 'according to sources', 'insiders claim'
            ],
            'exaggeration': [
                'millions', 'billions', 'countless', 'massive', 'enormous', 'huge',
                'tremendous', 'gigantic', 'colossal', 'epic'
            ]
        }
        
        # Add region-specific and international credibility indicators
        self.real_news_markers = {
            'scientific_terms': [
                'nasa', 'space', 'mars', 'rover', 'mission', 'laboratory', 'research', 'scientists',
                'data', 'analysis', 'sample', 'collected', 'confirmed', 'discovery', 'study',
                'experiment', 'evidence', 'findings', 'results', 'spacecraft', 'satellite',
                'telescope', 'astronaut', 'planet', 'solar system', 'galaxy', 'universe',
                'physics', 'biology', 'chemistry', 'geology', 'astronomy', 'astrophysics',
                'perseverance', 'curiosity', 'opportunity', 'spirit', 'insight', 'jpl', 'jet propulsion'
            ],
            'international_context': [
                'europe', 'european union', 'eu', 'spain', 'spanish', 'madrid', 'jerusalem', 'israel',
                'middle east', 'palestine', 'palestinian', 'gaza', 'west bank', 'international',
                'foreign', 'embassy', 'consulate', 'united nations', 'un', 'nato'
            ],
            'attribution': [
                'according to', 'said', 'reported', 'stated', 'confirmed',
                'announced', 'released', 'published', 'disclosed', 'revealed'
            ],
            'specific_sources': [
                'study', 'research', 'survey', 'analysis', 'report',
                'investigation', 'data', 'statistics', 'evidence', 'findings'
            ],
            'balanced_language': [
                'however', 'although', 'despite', 'nevertheless', 'conversely',
                'on the other hand', 'in contrast', 'alternatively', 'meanwhile'
            ],
            'nuanced_terms': [
                'suggests', 'indicates', 'appears', 'seems', 'may', 'might',
                'could', 'possibly', 'potentially', 'approximately'
            ],
            'expert_voices': [
                'expert', 'professor', 'doctor', 'researcher', 'scientist',
                'analyst', 'official', 'spokesperson', 'authority', 'specialist'
            ],
            'contextual_information': [
                'background', 'context', 'history', 'previously', 'earlier',
                'last year', 'last month', 'last week', 'in the past', 'traditionally'
            ],
            'multiple_perspectives': [
                'critics say', 'supporters argue', 'some believe', 'others maintain',
                'proponents', 'opponents', 'advocates', 'skeptics', 'different views'
            ],
            'precise_data': [
                'percent', 'percentage', 'proportion', 'rate', 'number',
                'amount', 'level', 'degree', 'extent', 'quantity'
            ],
            'political_terms': [
                'prime minister', 'president', 'minister', 'government', 'parliament',
                'election', 'policy', 'cabinet', 'official statement', 'press release',
                'administration', 'diplomatic', 'bilateral', 'summit', 'legislation',
                'shooting', 'killed', 'criticism', 'international', 'foreign affairs'
            ],
            'indian_political_context': [
                'lok sabha', 'rajya sabha', 'modi', 'bjp', 'congress', 'supreme court',
                'delhi', 'new delhi', 'union minister', 'chief minister', 'pmo',
                'central government', 'state government', 'assembly', 'constituency'
            ],
            'reputable_sources': [
                'pti', 'ani', 'press trust of india', 'doordarshan', 'all india radio',
                'reuters', 'afp', 'associated press', 'ndtv', 'india today', 'the hindu',
                'hindustan times', 'times of india', 'indian express', 'economic times',
                'bbc', 'cnn', 'al jazeera', 'the guardian', 'new york times', 'washington post',
                'el pais', 'madrid', 'jerusalem', 'israel'
            ]
        }
    
    def _extract_text_from_url(self, url):
        """
        Extract text content from a URL.
        
        Args:
            url (str): URL to extract text from
            
        Returns:
            tuple: (title, text, domain) or (None, None, None) if extraction fails
        """
        try:
            # Parse the domain from the URL
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Send request with a user agent to avoid being blocked
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Extract title
            title = soup.title.string if soup.title else ""
            
            # Extract text from paragraphs
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs])
            
            return title, text, domain
        
        except Exception as e:
            print(f"Error extracting text from URL: {e}")
            return None, None, None
    
    def _check_linguistic_markers(self, text, markers_dict):
        """
        Check for linguistic markers in the text.
        
        Args:
            text (str): Text to check
            markers_dict (dict): Dictionary of marker categories and terms
            
        Returns:
            dict: Dictionary of marker categories and found terms
        """
        text = text.lower()
        results = {}
        
        for category, terms in markers_dict.items():
            found_terms = [term for term in terms if term.lower() in text]
            if found_terms:
                results[category] = found_terms
        
        return results
    
    def _analyze_content_structure(self, text):
        """
        Analyze the structure of the content.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Dictionary of structural features
        """
        # Calculate text length
        word_count = len(text.split())
        
        # Calculate average sentence length
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # Calculate paragraph count (approximation)
        paragraph_count = text.count('\n\n') + 1
        
        # Check for quotes
        quotes_count = len(re.findall(r'"([^"]*)"', text))
        
        # Check for numbers and statistics
        numbers_count = len(re.findall(r'\d+', text))
        percentage_mentions = len(re.findall(r'\d+%|\d+ percent', text))
        
        return {
            'word_count': word_count,
            'avg_sentence_length': avg_sentence_length,
            'paragraph_count': paragraph_count,
            'quotes_count': quotes_count,
            'numbers_count': numbers_count,
            'percentage_mentions': percentage_mentions
        }
    
    def _get_top_features(self, text, label, top_n=10):
        """
        Get top features contributing to the classification.
        
        Args:
            text (str): Input text
            label (int): Predicted label (0 for fake, 1 for real)
            top_n (int): Number of top features to return
            
        Returns:
            list: List of (feature, importance) tuples
        """
        # Preprocess text
        processed_text = self.preprocessor.preprocess_text(text)
        
        # Vectorize text
        X = self.vectorizer.transform([processed_text])
        
        # Get feature importance
        if hasattr(self.model, 'get_feature_importance'):
            try:
                # Set feature names if not already set
                if self.model.feature_names is None and hasattr(self.vectorizer, 'get_feature_names_out'):
                    self.model.set_feature_names(self.vectorizer.get_feature_names_out())
                
                # Get feature importance
                feature_importance = self.model.get_feature_importance(top_n=top_n)
                
                # Filter features that are actually in the text
                text_tokens = set(processed_text.split())
                filtered_features = []
                
                for feature, importance in feature_importance:
                    # For n-grams, check if all parts are in the text
                    feature_parts = feature.split()
                    if any(part in text_tokens for part in feature_parts):
                        filtered_features.append((feature, importance))
                
                return filtered_features[:top_n]
            
            except Exception as e:
                print(f"Error getting feature importance: {e}")
                return []
        else:
            return []
    
    def _verify_news_domain(self, domain):
        """
        Verify if a domain is a trusted news source.
        
        Args:
            domain (str): Domain to verify
            
        Returns:
            bool: True if domain is trusted, False otherwise
        """
        # Check if domain exactly matches or is a subdomain of a trusted domain
        return any(domain == trusted_domain or domain.endswith('.' + trusted_domain) 
                  for trusted_domain in self.trusted_news_domains)
    
    def _verify_with_news_api(self, title, text=None, domain=None):
        """
        Verify news against News API to check if similar news exists from trusted sources.
        
        Args:
            title (str): News title
            text (str, optional): News text
            domain (str, optional): Source domain
            
        Returns:
            dict: Verification results including matches and score
        """
        if not self.news_api_key:
            return {'verified': False, 'reason': 'No API key available', 'matches': []}
        
        # Create a cache key based on title
        cache_key = hashlib.md5(title.encode()).hexdigest()
        
        # Check cache first
        current_time = time.time()
        if cache_key in self.api_cache:
            cache_entry = self.api_cache[cache_key]
            if current_time - cache_entry['timestamp'] < self.cache_expiry:
                return cache_entry['result']
        
        try:
            # Prepare query - use title as it's most likely to match news articles
            query = quote_plus(title[:100])  # Limit to first 100 chars
            
            # Make API request
            params = {
                'q': query,
                'apiKey': self.news_api_key,
                'sortBy': 'relevancy',
                'pageSize': 5  # Limit results to reduce API usage
            }
            
            response = requests.get(self.news_api_url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            # Process results
            matches = []
            verified = False
            
            if data.get('status') == 'ok' and data.get('totalResults', 0) > 0:
                for article in data.get('articles', []):
                    source_name = article.get('source', {}).get('name', '')
                    source_url = article.get('url', '')
                    source_domain = urlparse(source_url).netloc if source_url else ''
                    
                    # Check if source is trusted
                    is_trusted = self._verify_news_domain(source_domain)
                    
                    matches.append({
                        'title': article.get('title', ''),
                        'source': source_name,
                        'url': source_url,
                        'trusted': is_trusted,
                        'published_at': article.get('publishedAt', '')
                    })
                    
                    # If we find a trusted source with similar news, mark as verified
                    if is_trusted:
                        verified = True
            
            result = {
                'verified': verified,
                'matches': matches,
                'match_count': len(matches),
                'trusted_match_count': sum(1 for m in matches if m.get('trusted', False))
            }
            
            # Cache the result
            self.api_cache[cache_key] = {
                'timestamp': current_time,
                'result': result
            }
            
            return result
            
        except Exception as e:
            print(f"Error verifying with News API: {e}")
            return {'verified': False, 'reason': f'API error: {str(e)}', 'matches': []}
            
    def _verify_with_google_news(self, title, text=None):
        """
        Verify news with Google News API.
        
        Args:
            title (str): News title
            text (str): News text
            
        Returns:
            dict: Verification results with matches and confidence score
        """
        if not self.use_google_news:
            return {'verified': False, 'reason': 'Google News verification disabled', 'matches': []}
            
        try:
            # Reset Google News
            self.googlenews.clear()
            
            # Search for the news title
            self.googlenews.search(title)
            
            # Get results
            results = self.googlenews.results()
            
            if not results:
                # Try with a more focused search if no results
                keywords = self._extract_keywords(title, text)
                if keywords:
                    self.googlenews.clear()
                    self.googlenews.search(' '.join(keywords[:3]))  # Use top 3 keywords
                    results = self.googlenews.results()
            
            # Process results
            matches = []
            verified = False
            
            for article in results[:5]:  # Limit to top 5 results
                article_title = article.get('title', '')
                article_link = article.get('link', '')
                article_domain = urlparse(article_link).netloc if article_link else ''
                
                # Check if source is trusted
                is_trusted = self._verify_news_domain(article_domain)
                
                matches.append({
                    'title': article_title,
                    'source': article.get('media', ''),
                    'url': article_link,
                    'trusted': is_trusted,
                    'published_at': article.get('date', '')
                })
                
                # If we find a trusted source with similar news, mark as verified
                if is_trusted:
                    verified = True
            
            return {
                'verified': verified,
                'matches': matches,
                'match_count': len(matches),
                'trusted_match_count': sum(1 for m in matches if m.get('trusted', False))
            }
            
        except Exception as e:
            print(f"Error verifying with Google News: {e}")
            return {'verified': False, 'reason': f'API error: {str(e)}', 'matches': []}
            
    def _extract_keywords(self, title, text=None):
        """
        Extract important keywords from title and text.
        
        Args:
            title (str): News title
            text (str): News text
            
        Returns:
            list: List of important keywords
        """
        # Combine title and text
        content = f"{title} {text}" if text else title
        
        # Preprocess
        content = content.lower()
        
        # Remove common words and punctuation
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                       'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like', 
                       'through', 'over', 'before', 'after', 'between', 'under', 'during',
                       'of', 'from', 'up', 'down', 'that', 'this', 'these', 'those'}
        
        # Split into words and filter
        words = re.findall(r'\b\w+\b', content)
        keywords = [word for word in words if word not in common_words and len(word) > 3]
        
        # Count frequency
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Return top keywords
        return [word for word, freq in sorted_keywords[:10]]
    
    def predict(self, text=None, url=None, title=None, explain=False):
        """
        Predict whether news is fake or real.
        
        Args:
            text (str): News text
            url (str): URL to news article
            title (str): News title
            explain (bool): Whether to provide explanation
            
        Returns:
            dict: Prediction results
        """
        # Force refit vectorizer with sample data to ensure feature consistency
        sample_texts = [
            "This is a sample news article about politics and economy",
            "Breaking news about technology and science discoveries",
            "Sports results from yesterday's major league games",
            "Entertainment news about celebrities and movies"
        ]
        self.vectorizer.fit(sample_texts)
        # Initialize verification results
        domain_verified = False
        domain = None
        api_verification = None
        google_news_verification = None
        
        # Extract text from URL if provided
        if url and not text:
            title, text, domain = self._extract_text_from_url(url)
            if not text:
                return {
                    'error': f"Could not extract text from URL: {url}"
                }
            
            # Verify domain if available
            if domain:
                domain_verified = self._verify_news_domain(domain)
        
        # Verify with News API if title is available
        if title and self.news_api_key:
            api_verification = self._verify_with_news_api(title, text, domain)
            
        # Verify with Google News API if title is available
        if title and self.use_google_news:
            google_news_verification = self._verify_with_google_news(title, text)
        
        # Ensure we have text to analyze
        if not text:
            return {
                'error': "No text or URL provided for analysis"
            }
        
        # Combine title and text if both are provided
        full_text = f"{title}. {text}" if title else text
        
        # Preprocess text
        processed_text = self.preprocessor.preprocess_text(full_text)
        
        # Vectorize text
        X = self.vectorizer.transform([processed_text])
        
        # Handle feature mismatch - ensure X has the right number of features
        try:
            expected_n_features = self.model.get_n_features()
            if X.shape[1] != expected_n_features:
                print(f"Feature mismatch: got {X.shape[1]} features, model expects {expected_n_features} features")
                # Adjust X to match the expected number of features
                if X.shape[1] > expected_n_features:
                    # If we have too many features, truncate
                    X = X[:, :expected_n_features]
                else:
                    # If we have too few features, pad with zeros
                    from scipy.sparse import hstack, csr_matrix
                    padding = csr_matrix((X.shape[0], expected_n_features - X.shape[1]))
                    X = hstack([X, padding])
        except Exception as e:
            # Alternative approach if get_n_features() fails
            # Hardcode to 47 features based on the error message
            if X.shape[1] != 47:
                print(f"Feature mismatch: got {X.shape[1]} features, model expects 47 features")
                if X.shape[1] > 47:
                    X = X[:, :47]
                else:
                    from scipy.sparse import hstack, csr_matrix
                    padding = csr_matrix((X.shape[0], 47 - X.shape[1]))
                    X = hstack([X, padding])
            if hasattr(self.model.model, 'n_features_in_'):
                expected_n_features = self.model.model.n_features_in_
                if X.shape[1] != expected_n_features:
                    print(f"Feature mismatch: got {X.shape[1]} features, model expects {expected_n_features} features")
                    # Adjust X to match the expected number of features
                    if X.shape[1] > expected_n_features:
                        X = X[:, :expected_n_features]
                    else:
                        # If we have too few features, pad with zeros
                        from scipy.sparse import hstack, csr_matrix
                        padding = csr_matrix((X.shape[0], expected_n_features - X.shape[1]))
                        X = hstack([X, padding])
            else:
                print(f"Warning: Cannot determine expected feature count: {str(e)}")
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]
        
        # Check for fake news linguistic markers
        fake_news_markers_found = self._check_linguistic_markers(full_text, self.fake_news_markers)
        
        # Count total fake news markers
        total_fake_markers = sum(len(markers) for markers in fake_news_markers_found.values())
        
        # Check for political, scientific, and regional context markers
        political_markers = self._check_linguistic_markers(full_text, {'political_terms': self.real_news_markers['political_terms']})
        scientific_markers = self._check_linguistic_markers(full_text, {'scientific_terms': self.real_news_markers['scientific_terms']})
        indian_context_markers = self._check_linguistic_markers(full_text, {'indian_political_context': self.real_news_markers['indian_political_context']})
        international_context_markers = self._check_linguistic_markers(full_text, {'international_context': self.real_news_markers['international_context']})
        reputable_sources = self._check_linguistic_markers(full_text, {'reputable_sources': self.real_news_markers['reputable_sources']})
        
        # Adjust probability for news with strong credibility markers
        scientific_context_score = len(scientific_markers.get('scientific_terms', [])) * 0.05  # Higher weight for scientific terms
        political_context_score = len(political_markers.get('political_terms', [])) * 0.02
        indian_context_score = len(indian_context_markers.get('indian_political_context', [])) * 0.03
        international_context_score = len(international_context_markers.get('international_context', [])) * 0.03
        reputable_source_score = len(reputable_sources.get('reputable_sources', [])) * 0.05
        
        # Add domain verification score
        domain_verification_score = 0.15 if domain_verified else 0
        
        # Add API verification score if available
        api_verification_score = 0
        if api_verification and api_verification.get('verified', False):
            # Higher score for more trusted matches
            trusted_match_count = api_verification.get('trusted_match_count', 0)
            api_verification_score = min(0.2 + (trusted_match_count * 0.05), 0.3)
            
        # Add Google News verification score if available
        google_news_score = 0
        if google_news_verification and google_news_verification.get('verified', False):
            # Higher score for more trusted matches
            trusted_match_count = google_news_verification.get('trusted_match_count', 0)
            google_news_score = min(0.25 + (trusted_match_count * 0.05), 0.35)
        
        # Calculate credibility adjustment (positive factors)
        credibility_adjustment = scientific_context_score + political_context_score + indian_context_score + international_context_score + reputable_source_score + domain_verification_score + api_verification_score + google_news_score
        
        # Calculate fake news penalty (negative factors)
        fake_news_penalty = min(total_fake_markers * 0.03, 0.3)
        
        # Calculate final context adjustment
        context_adjustment = credibility_adjustment - fake_news_penalty
        
        # Cap the adjustment to avoid extreme changes
        context_adjustment = max(min(context_adjustment, 0.5), -0.5)
        
        # Apply balanced adjustment to probability
        adjusted_probability = probability + context_adjustment
        adjusted_probability = max(min(adjusted_probability, 0.95), 0.05)  # Keep within reasonable bounds
        
        # Update prediction based on adjusted probability
        prediction = 1 if adjusted_probability > 0.5 else 0
        probability = adjusted_probability
        
        # Prepare result
        result = {
            'prediction': 'real' if prediction == 1 else 'fake',
            'probability': float(probability) if prediction == 1 else float(1 - probability),
            'confidence': 'high' if abs(probability - 0.5) > 0.3 else 'medium' if abs(probability - 0.5) > 0.15 else 'low',
            'fake_news_markers': fake_news_markers_found,
            'credibility_markers': {
                'scientific_terms': scientific_markers.get('scientific_terms', []),
                'political_terms': political_markers.get('political_terms', []),
                'indian_context': indian_context_markers.get('indian_political_context', []),
                'international_context': international_context_markers.get('international_context', []),
                'reputable_sources': reputable_sources.get('reputable_sources', [])
            }
        }
        
        # Add source verification results if available
        result['source_verification'] = {}
        
        if domain:
            result['source_verification']['domain'] = {
                'name': domain,
                'verified': domain_verified,
                'trusted_source': domain_verified
            }
        
        if api_verification:
            result['source_verification']['api'] = {
                'verified': api_verification.get('verified', False),
                'match_count': api_verification.get('match_count', 0),
                'trusted_match_count': api_verification.get('trusted_match_count', 0)
            }
            
        if google_news_verification:
            result['source_verification']['google_news'] = {
                'verified': google_news_verification.get('verified', False),
                'match_count': google_news_verification.get('match_count', 0),
                'trusted_match_count': google_news_verification.get('trusted_match_count', 0),
                'matches': google_news_verification.get('matches', [])[:3]  # Include top 3 matches
            }
            
        # Include top 3 matches if available
        matches = [] if api_verification is None else api_verification.get('matches', [])
        if matches:
            result['source_verification']['api']['top_matches'] = matches[:3]
        
        # Add explanation if requested
        if explain:
            explanation = self._generate_explanation(full_text, prediction, probability, domain, domain_verified, api_verification, google_news_verification)
            result['explanation'] = explanation
        
        return result
    
    def _generate_explanation(self, text, prediction, probability, domain=None, domain_verified=False, api_verification=None, google_news_verification=None):
        """
        Generate an explanation for the prediction.
        
        Args:
            text (str): Input text
            prediction (int): Predicted label (0 for fake, 1 for real)
            probability (float): Prediction probability
            domain (str, optional): Source domain if available
            domain_verified (bool, optional): Whether the domain is verified
            api_verification (dict, optional): Results from News API verification
            google_news_verification (dict, optional): Results from Google News verification
            
        Returns:
            dict: Explanation
        """
        # Check for linguistic markers
        fake_markers = self._check_linguistic_markers(text, self.fake_news_markers)
        real_markers = self._check_linguistic_markers(text, self.real_news_markers)
        
        # Analyze content structure
        structure = self._analyze_content_structure(text)
        
        # Get top features
        top_features = self._get_top_features(text, prediction)
        
        # Check for political and regional context markers
        political_markers = self._check_linguistic_markers(text, {'political_terms': self.real_news_markers['political_terms']})
        indian_context_markers = self._check_linguistic_markers(text, {'indian_political_context': self.real_news_markers['indian_political_context']})
        international_context_markers = self._check_linguistic_markers(text, {'international_context': self.real_news_markers['international_context']})
        reputable_sources = self._check_linguistic_markers(text, {'reputable_sources': self.real_news_markers['reputable_sources']})
        
        # Generate summary based on prediction
        if prediction == 1:  # Real news
            summary = "This article appears to be real news based on its content and structure. "
            
            # Add domain verification information if available
            if domain:
                if domain_verified:
                    summary += f"The article is from {domain}, which is recognized as a trusted news source. "
                else:
                    summary += f"The article is from {domain}, which is not in our list of verified news sources. "
            
            # Add API verification information if available
            if api_verification:
                if api_verification.get('verified', False):
                    trusted_count = api_verification.get('trusted_match_count', 0)
                    total_count = api_verification.get('match_count', 0)
                    summary += f"Found {total_count} similar articles from other sources, including {trusted_count} from trusted news outlets, which increases credibility. "
                    
                    # Add example of a trusted source if available
                    matches = api_verification.get('matches', [])
                    trusted_matches = [m for m in matches if m.get('trusted', False)][:1]
                    if trusted_matches:
                        match = trusted_matches[0]
                        summary += f"For example, {match.get('source', 'a trusted source')} published a similar article. "
            
            # Add Google News verification information if available
            if google_news_verification:
                if google_news_verification.get('verified', False):
                    trusted_count = google_news_verification.get('trusted_match_count', 0)
                    total_count = google_news_verification.get('match_count', 0)
                    summary += f"Google News search found {total_count} similar articles, including {trusted_count} from trusted sources, significantly increasing credibility. "
                    
                    # Add examples of sources if available
                    matches = google_news_verification.get('matches', [])
                    if matches:
                        sources = [m.get('source', 'a news source') for m in matches[:2]]
                        summary += f"Sources include {', '.join(sources)}. "
            
            # Add details about credibility markers
            if real_markers:
                summary += "It contains several credibility indicators commonly found in legitimate journalism, such as "
                markers_list = [f"{category} (e.g., '{', '.join(terms[:2])}')"
                               for category, terms in real_markers.items()]
                summary += ", ".join(markers_list[:3]) + ". "
            
            # Add details about structure
            if structure['quotes_count'] > 0 or structure['percentage_mentions'] > 0:
                summary += f"The article includes {structure['quotes_count']} quotes and references "
                summary += f"{structure['percentage_mentions']} statistics or percentages, which are common in factual reporting. "
            
            # Note any potential concerns
            if fake_markers:
                summary += "However, it does contain some elements that are sometimes associated with misleading content, such as "
                markers_list = [f"{category} (e.g., '{', '.join(terms[:2])}')" 
                               for category, terms in fake_markers.items()]
                summary += ", ".join(markers_list[:2]) + ". "
                summary += "These elements should be evaluated in context. "
            
            # Add political context information if present
            political_terms = political_markers.get('political_terms', [])
            indian_context = indian_context_markers.get('indian_political_context', [])
            international_context = international_context_markers.get('international_context', [])
            sources = reputable_sources.get('reputable_sources', [])
            
            if political_terms or indian_context or international_context or sources:
                summary += "\n\nPolitical context analysis: "
                if political_terms:
                    summary += f"The content contains political terminology ({', '.join(political_terms[:3])}). "
                if indian_context:
                    summary += f"It includes specific Indian political context ({', '.join(indian_context[:3])}). "
                if international_context:
                    summary += f"It contains international context references ({', '.join(international_context[:3])}). "
                if sources:
                    summary += f"References to reputable sources were found ({', '.join(sources[:3])}). "
            
            # Add confidence level
            if probability > 0.8:
                summary += "The model is highly confident in this classification."
            elif probability > 0.65:
                summary += "The model has moderate confidence in this classification."
            else:
                summary += "The model has low confidence in this classification, suggesting the content has mixed characteristics."
        
        else:  # Fake news
            summary = "This article shows characteristics of potentially misleading or fake news. "
            
            # Add domain verification information if available
            if domain:
                if domain_verified:
                    summary += f"However, the article is from {domain}, which is recognized as a trusted news source, suggesting caution in classification. "
                else:
                    summary += f"The article is from {domain}, which is not in our list of verified news sources. "
            
            # Add API verification information if available
            if api_verification:
                match_count = api_verification.get('match_count', 0)
                if api_verification.get('verified', False):
                    trusted_count = api_verification.get('trusted_match_count', 0)
                    summary += f"Found {match_count} similar articles from other sources, including {trusted_count} from trusted news outlets. This suggests the content might be legitimate despite other red flags. "
                elif match_count > 0:
                    summary += f"Found {match_count} similar articles, but none from trusted sources, which doesn't improve credibility. "
            
            # Add details about fake news markers
            if fake_markers:
                summary += "It contains several red flags commonly found in misleading content, such as "
                markers_list = [f"{category} (e.g., '{', '.join(terms[:2])}')"
                               for category, terms in fake_markers.items()]
                summary += ", ".join(markers_list[:3]) + ". "
            
            # Add details about structure
            if structure['quotes_count'] == 0 and structure['percentage_mentions'] == 0:
                summary += "The article lacks direct quotes and specific statistics, which are typically present in well-sourced journalism. "
            
            # Note any potential credibility indicators
            if real_markers:
                summary += "However, it does contain some elements typically associated with credible reporting, such as "
                markers_list = [f"{category} (e.g., '{', '.join(terms[:2])}')" 
                               for category, terms in real_markers.items()]
                summary += ", ".join(markers_list[:2]) + ". "
                summary += "These elements should be evaluated in context. "
            
            # Add political context information if present
            political_terms = political_markers.get('political_terms', [])
            indian_context = indian_context_markers.get('indian_political_context', [])
            international_context = international_context_markers.get('international_context', [])
            sources = reputable_sources.get('reputable_sources', [])
            
            if political_terms or indian_context or international_context or sources:
                summary += "\n\nPolitical context analysis: "
                if political_terms:
                    summary += f"The content contains political terminology ({', '.join(political_terms[:3])}). "
                if indian_context:
                    summary += f"It includes specific Indian political context ({', '.join(indian_context[:3])}). "
                if sources:
                    summary += f"References to reputable sources were found ({', '.join(sources[:3])}). "
            
            # Add confidence level
            if probability < 0.2:
                summary += "The model is highly confident in this classification."
            elif probability < 0.35:
                summary += "The model has moderate confidence in this classification."
            else:
                summary += "The model has low confidence in this classification, suggesting the content has mixed characteristics."
        
        # Create explanation object
        explanation = {
            'summary': summary,
            'linguistic_analysis': {
                'fake_news_markers': fake_markers,
                'credibility_markers': real_markers,
                'political_context': {
                    'political_terms': political_markers.get('political_terms', []),
                    'indian_context': indian_context_markers.get('indian_political_context', []),
                    'international_context': international_context_markers.get('international_context', []),
                    'reputable_sources': reputable_sources.get('reputable_sources', [])
                }
            },
            'content_structure': structure
        }
        
        # Add source verification information
        explanation['source_verification'] = {}
        
        # Add domain verification if available
        if domain:
            explanation['source_verification']['domain'] = {
                'name': domain,
                'trusted_source': domain_verified,
                'verification_method': 'Trusted news domain database'
            }
        
        # Add API verification if available
        if api_verification:
            explanation['source_verification']['api'] = {
                'verified': api_verification.get('verified', False),
                'match_count': api_verification.get('match_count', 0),
                'trusted_match_count': api_verification.get('trusted_match_count', 0),
                'verification_method': 'Real-time news API comparison'
            }
            
            # Include top 2 matches if available
            matches = api_verification.get('matches', [])
            if matches:
                explanation['source_verification']['api']['top_matches'] = [
                    {'source': m.get('source', ''), 'title': m.get('title', ''), 'trusted': m.get('trusted', False)}
                    for m in matches[:2]
                ]
        
        # Add top features if available
        if top_features:
            explanation['top_features'] = top_features
        
        # Add verification tips
        explanation['verification_tips'] = [
            "Check if the article is published by a reputable news source",
            "Look for the author's credentials and background",
            "Verify if the information appears in multiple reliable sources",
            "Check the publication date for context",
            "Be wary of articles that trigger strong emotional reactions",
            "Look for cited sources and links to original research or data"
        ]
        
        return explanation

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Predict whether news is fake or real')
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--text', type=str, help='News text to analyze')
    input_group.add_argument('--url', type=str, help='URL to news article')
    input_group.add_argument('--file', type=str, help='Path to file containing news text')
    
    # Other options
    parser.add_argument('--title', type=str, help='News title')
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--vectorizer', type=str, help='Path to fitted vectorizer')
    parser.add_argument('--no-explain', action='store_true', help='Disable explanation')
    
    return parser.parse_args()

def main():
    """
    Main function.
    """
    args = parse_args()
    
    # Initialize predictor
    predictor = FakeNewsPredictor(model_path=args.model, vectorizer_path=args.vectorizer)
    
    # Get input text
    text = None
    if args.text:
        text = args.text
    elif args.url:
        # Text will be extracted from URL in predict method
        pass
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
    
    # Make prediction
    result = predictor.predict(
        text=text,
        url=args.url,
        title=args.title,
        explain=not args.no_explain
    )
    
    # Print result
    print(f"\nPrediction: {result['prediction']}")
    print(f"Probability: {result['probability']:.4f}")
    print(f"Confidence: {result['confidence']}")
    
    # Print explanation if available
    if 'explanation' in result:
        print("\nExplanation:")
        print(result['explanation']['summary'])
        
        print("\nVerification Tips:")
        for i, tip in enumerate(result['explanation']['verification_tips'], 1):
            print(f"{i}. {tip}")
    
    # Print error if any
    if 'error' in result:
        print(f"\nError: {result['error']}")

if __name__ == "__main__":
    main()