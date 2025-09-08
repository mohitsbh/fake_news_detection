import re
import string
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class TextPreprocessor:
    """
    Class for preprocessing text data for fake news detection.
    """
    def __init__(self, remove_stopwords=True, lemmatize=True, lowercase=True):
        """
        Initialize the text preprocessor.
        
        Args:
            remove_stopwords (bool): Whether to remove stopwords
            lemmatize (bool): Whether to lemmatize words
            lowercase (bool): Whether to convert text to lowercase
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.lowercase = lowercase
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = None
    
    def clean_text(self, text):
        """
        Clean text by removing special characters, URLs, etc.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # Convert to lowercase if specified
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Preserve important punctuation but remove others
        # Keep periods, commas, hyphens, and apostrophes as they're important for names and dates
        punctuation_to_keep = '.,-\''
        punctuation_to_remove = ''.join([p for p in string.punctuation if p not in punctuation_to_keep])
        text = text.translate(str.maketrans('', '', punctuation_to_remove))
        
        # Preserve numbers and dates as they're important for political news verification
        # Instead of removing all numbers, we'll keep them
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """
        Tokenize text into words.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of tokens
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords if specified
        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in self.stop_words]
        
        # Lemmatize if specified
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        return tokens
    
    def preprocess_text(self, text):
        """
        Preprocess text by cleaning and tokenizing.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize(cleaned_text)
        return ' '.join(tokens)
    
    def preprocess_df(self, df, text_columns):
        """
        Preprocess text columns in a DataFrame.
        
        Args:
            df (pandas.DataFrame): Input DataFrame
            text_columns (list): List of column names containing text
            
        Returns:
            pandas.DataFrame: DataFrame with preprocessed text
        """
        df_processed = df.copy()
        
        for col in text_columns:
            if col in df.columns:
                df_processed[f'{col}_processed'] = df[col].apply(self.preprocess_text)
        
        return df_processed
    
    def fit_vectorizer(self, texts, vectorizer_type='tfidf', max_features=10000, ngram_range=(1, 2)):
        """
        Fit a vectorizer on the preprocessed texts.
        
        Args:
            texts (list): List of preprocessed texts
            vectorizer_type (str): Type of vectorizer ('tfidf' or 'count')
            max_features (int): Maximum number of features
            ngram_range (tuple): Range of n-grams
            
        Returns:
            self: The fitted vectorizer
        """
        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        else:  # count vectorizer
            self.vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
        
        self.vectorizer.fit(texts)
        return self
    
    def transform_texts(self, texts):
        """
        Transform texts using the fitted vectorizer.
        
        Args:
            texts (list): List of preprocessed texts
            
        Returns:
            scipy.sparse.csr.csr_matrix: Vectorized texts
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_vectorizer first.")
        
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        """
        Get feature names from the vectorizer.
        
        Returns:
            list: List of feature names
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_vectorizer first.")
        
        return self.vectorizer.get_feature_names_out()
    
    def extract_features(self, df, text_columns, vectorizer_type='tfidf', max_features=10000, ngram_range=(1, 2)):
        """
        Extract features from text columns in a DataFrame.
        
        Args:
            df (pandas.DataFrame): Input DataFrame
            text_columns (list): List of column names containing text
            vectorizer_type (str): Type of vectorizer ('tfidf' or 'count')
            max_features (int): Maximum number of features
            ngram_range (tuple): Range of n-grams
            
        Returns:
            tuple: (X, feature_names)
        """
        # Preprocess text columns
        df_processed = self.preprocess_df(df, text_columns)
        
        # Combine processed text columns
        combined_texts = []
        for _, row in df_processed.iterrows():
            combined_text = ' '.join([row[f'{col}_processed'] for col in text_columns if f'{col}_processed' in row])
            combined_texts.append(combined_text)
        
        # Fit and transform vectorizer
        self.fit_vectorizer(combined_texts, vectorizer_type, max_features, ngram_range)
        X = self.transform_texts(combined_texts)
        feature_names = self.get_feature_names()
        
        return X, feature_names

# Additional utility functions for text analysis

def get_top_n_words(corpus, n=10):
    """
    Get the top N most frequent words in a corpus.
    
    Args:
        corpus (list): List of texts
        n (int): Number of top words to return
        
    Returns:
        list: List of (word, count) tuples
    """
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

def get_ngrams(corpus, n=2, top_k=10):
    """
    Get the top K most frequent n-grams in a corpus.
    
    Args:
        corpus (list): List of texts
        n (int): N-gram size
        top_k (int): Number of top n-grams to return
        
    Returns:
        list: List of (ngram, count) tuples
    """
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:top_k]

def calculate_text_stats(texts):
    """
    Calculate various statistics about texts.
    
    Args:
        texts (list): List of texts
        
    Returns:
        dict: Dictionary of statistics
    """
    stats = {}
    
    # Calculate average text length
    text_lengths = [len(text.split()) for text in texts]
    stats['avg_length'] = np.mean(text_lengths)
    stats['median_length'] = np.median(text_lengths)
    stats['min_length'] = np.min(text_lengths)
    stats['max_length'] = np.max(text_lengths)
    stats['std_length'] = np.std(text_lengths)
    
    return stats

if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "This is a sample text about fake news that contains misleading information.",
        "Here is a real news article with factual information about current events.",
        "Another example of fake news with exaggerated claims and sensationalism."
    ]
    
    # Create preprocessor
    preprocessor = TextPreprocessor()
    
    # Preprocess texts
    processed_texts = [preprocessor.preprocess_text(text) for text in sample_texts]
    print("Processed texts:")
    for text in processed_texts:
        print(f"- {text}")
    
    # Extract features
    df = pd.DataFrame({'text': sample_texts})
    X, feature_names = preprocessor.extract_features(df, ['text'])
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    # Get top words
    top_words = get_top_n_words(processed_texts, n=5)
    print("\nTop 5 words:")
    for word, count in top_words:
        print(f"- {word}: {count}")