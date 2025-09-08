import os
import pandas as pd
import numpy as np
import requests
import zipfile
import io
from sklearn.model_selection import train_test_split

# URLs for fake news datasets
FAKE_NEWS_KAGGLE_URL = "https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/download"
LIAR_DATASET_URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"

# Local paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def download_kaggle_dataset(kaggle_api_token=None):
    """
    Download the Fake and Real News dataset from Kaggle.
    Requires Kaggle API token to be set up.
    
    Args:
        kaggle_api_token (dict): Kaggle API credentials with username and key
    
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        # If kaggle_api_token is provided, save it to ~/.kaggle/kaggle.json
        if kaggle_api_token:
            import json
            os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
            with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as f:
                json.dump(kaggle_api_token, f)
            os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)
        
        # Use Kaggle API to download dataset
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files('clmentbisaillon/fake-and-real-news-dataset', 
                                  path=RAW_DATA_DIR, unzip=True)
        print("Kaggle dataset downloaded successfully.")
        return True
    except Exception as e:
        print(f"Error downloading Kaggle dataset: {e}")
        print("Please manually download the dataset from Kaggle and place it in the data/raw directory.")
        print("Dataset URL: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        return False

def download_liar_dataset():
    """
    Download the LIAR dataset.
    
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        response = requests.get(LIAR_DATASET_URL)
        if response.status_code == 200:
            z = zipfile.ZipFile(io.BytesIO(response.content))
            z.extractall(RAW_DATA_DIR)
            print("LIAR dataset downloaded successfully.")
            return True
        else:
            print(f"Failed to download LIAR dataset. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading LIAR dataset: {e}")
        return False

def load_fake_real_news_dataset():
    """
    Load the Fake and Real News dataset.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test) or None if dataset not found
    """
    try:
        # Check if files exist
        fake_path = os.path.join(RAW_DATA_DIR, 'Fake.csv')
        true_path = os.path.join(RAW_DATA_DIR, 'True.csv')
        
        if not (os.path.exists(fake_path) and os.path.exists(true_path)):
            print("Dataset files not found. Please download the dataset first.")
            return None
        
        # Load datasets
        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)
        
        # Add labels
        fake_df['label'] = 0  # 0 for fake
        true_df['label'] = 1  # 1 for real
        
        # Combine datasets
        df = pd.concat([fake_df, true_df], ignore_index=True)
        
        # Shuffle the data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split features and target
        X = df[['title', 'text']]
        y = df['label']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Save processed data
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        train_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'test.csv'), index=False)
        
        print(f"Dataset loaded and processed. Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def load_liar_dataset():
    """
    Load the LIAR dataset.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test) or None if dataset not found
    """
    try:
        # Check if files exist
        train_path = os.path.join(RAW_DATA_DIR, 'train.tsv')
        test_path = os.path.join(RAW_DATA_DIR, 'test.tsv')
        
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            print("LIAR dataset files not found. Please download the dataset first.")
            return None
        
        # Column names for the TSV files
        columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job_title', 'state_info', 
                  'party_affiliation', 'barely_true_counts', 'false_counts', 'half_true_counts', 
                  'mostly_true_counts', 'pants_on_fire_counts', 'context']
        
        # Load datasets
        train_df = pd.read_csv(train_path, sep='\t', names=columns)
        test_df = pd.read_csv(test_path, sep='\t', names=columns)
        
        # Map text labels to numeric
        label_map = {
            'pants-fire': 0,  # Completely false
            'false': 0,      # False
            'barely-true': 0, # Mostly false
            'half-true': 1,  # Mixed
            'mostly-true': 1, # Mostly true
            'true': 1        # True
        }
        
        train_df['numeric_label'] = train_df['label'].map(label_map)
        test_df['numeric_label'] = test_df['label'].map(label_map)
        
        # Split features and target
        X_train = train_df[['statement', 'subject', 'speaker', 'context']]
        y_train = train_df['numeric_label']
        
        X_test = test_df[['statement', 'subject', 'speaker', 'context']]
        y_test = test_df['numeric_label']
        
        # Save processed data
        train_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'liar_train.csv'), index=False)
        test_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'liar_test.csv'), index=False)
        
        print(f"LIAR dataset loaded and processed. Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        print(f"Error loading LIAR dataset: {e}")
        return None

def create_sample_dataset(size=1000, save=True):
    """
    Create a synthetic dataset for testing when real datasets are not available.
    
    Args:
        size (int): Number of samples to generate
        save (bool): Whether to save the dataset to disk
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Sample fake news headlines and content
    fake_titles = [
        "BREAKING: Scientist discovers cure for all diseases",
        "Government confirms aliens living among us",
        "Celebrity secretly a robot, insider reveals",
        "New study shows chocolate is healthier than vegetables",
        "World leaders agree to cancel all debt worldwide"
    ]
    
    fake_content = [
        "A scientist working alone in a small lab has discovered a miracle cure that big pharma doesn't want you to know about...",
        "Top secret documents reveal that the government has been hiding aliens for decades...",
        "Anonymous sources close to the celebrity have confirmed that they are actually an advanced AI...",
        "Researchers found that eating chocolate three times a day leads to weight loss and improved health...",
        "In a historic summit, world leaders have unanimously agreed to forgive all international debt..."
    ]
    
    # Sample real news headlines and content
    real_titles = [
        "Study shows modest benefits of new treatment for specific condition",
        "Government releases annual budget report",
        "Celebrity announces new charitable foundation",
        "Research indicates balanced diet remains best for overall health",
        "International talks continue on economic cooperation"
    ]
    
    real_content = [
        "A peer-reviewed study published in the Journal of Medicine found that a new treatment showed a 15% improvement for patients with specific conditions...",
        "The Treasury Department released its annual budget report today, showing both areas of growth and concern in the economy...",
        "The celebrity announced the launch of a new foundation aimed at addressing childhood education in underserved communities...",
        "A comprehensive review of dietary studies confirms that a balanced approach to nutrition continues to show the best long-term health outcomes...",
        "Representatives from 12 countries met this week to discuss ongoing economic cooperation and trade agreements..."
    ]
    
    # Add political news examples (real)
    political_real_titles = [
        "Prime Minister Modi inaugurates new infrastructure project",
        "Parliament passes key legislation on economic reforms",
        "Election Commission announces dates for upcoming state elections",
        "Supreme Court delivers verdict on constitutional matter",
        "India and US sign new bilateral trade agreement"
    ]
    
    political_real_content = [
        "Prime Minister Narendra Modi inaugurated the new highway project on Wednesday, stating that it would boost connectivity and economic growth in the region...",
        "The Lok Sabha passed the Economic Reforms Bill with a majority vote after a six-hour debate. Finance Minister highlighted that the reforms would attract foreign investment...",
        "The Election Commission of India today announced the schedule for assembly elections in five states. Voting will take place between October and November...",
        "In a landmark judgment, the Supreme Court of India ruled in favor of the constitutional validity of the new legislation. Chief Justice stated that the law does not violate fundamental rights...",
        "India and the United States signed a new trade agreement during the bilateral summit in New Delhi yesterday. Commerce Minister said this would boost exports significantly..."
    ]
    
    # Add political fake news examples
    political_fake_titles = [
        "PM Modi resigns suddenly without explanation",
        "Government secretly planning to change constitution",
        "Opposition leader caught accepting billions in foreign money",
        "Secret deal between India and enemy nation exposed",
        "Ruling party to ban all opposition parties next month"
    ]
    
    political_fake_content = [
        "Sources close to the PMO claim that Prime Minister Modi has submitted his resignation letter last night without any explanation. The news is being suppressed...",
        "Insiders reveal that the government is planning to completely rewrite the constitution in secret meetings. The changes would eliminate democratic processes...",
        "Shocking documents show that the opposition leader has received billions in illegal funds from foreign governments to destabilize India...",
        "A whistleblower has exposed a secret agreement between Indian officials and an enemy nation that would compromise national security...",
        "Anonymous sources within the ruling party confirm plans to ban all opposition parties next month using emergency powers..."
    ]
    
    # Add political news to existing collections
    real_titles.extend(political_real_titles)
    real_content.extend(political_real_content)
    fake_titles.extend(political_fake_titles)
    fake_content.extend(political_fake_content)
    
    # Generate synthetic data
    np.random.seed(42)
    data = []
    
    for _ in range(size):
        if np.random.random() > 0.5:  # Real news
            title_idx = np.random.randint(0, len(real_titles))
            title = real_titles[title_idx]
            content = real_content[title_idx]
            # Add some variation
            content = content + " " + np.random.choice(real_content)
            label = 1
        else:  # Fake news
            title_idx = np.random.randint(0, len(fake_titles))
            title = fake_titles[title_idx]
            content = fake_content[title_idx]
            # Add some variation
            content = content + " " + np.random.choice(fake_content)
            label = 0
            
        data.append({
            'title': title,
            'text': content,
            'label': label
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Split into train and test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    if save:
        # Save to disk
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        train_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'synthetic_train.csv'), index=False)
        test_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'synthetic_test.csv'), index=False)
    
    # Return split data
    X_train = train_df[['title', 'text']]
    y_train = train_df['label']
    X_test = test_df[['title', 'text']]
    y_test = test_df['label']
    
    print(f"Synthetic dataset created. Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test

def get_dataset(dataset_name='synthetic', download=False):
    """
    Main function to get a dataset by name.
    
    Args:
        dataset_name (str): Name of the dataset ('kaggle', 'liar', or 'synthetic')
        download (bool): Whether to attempt downloading the dataset
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if dataset_name == 'kaggle':
        if download:
            download_kaggle_dataset()
        return load_fake_real_news_dataset()
    
    elif dataset_name == 'liar':
        if download:
            download_liar_dataset()
        return load_liar_dataset()
    
    elif dataset_name == 'synthetic':
        return create_sample_dataset(size=1000, save=True)
    
    else:
        print(f"Unknown dataset: {dataset_name}. Using synthetic dataset instead.")
        return create_sample_dataset(size=1000, save=True)

if __name__ == "__main__":
    # Example usage
    print("Creating synthetic dataset for testing...")
    X_train, X_test, y_train, y_test = get_dataset('synthetic')
    print(f"Dataset shape - X_train: {X_train.shape}, y_train: {y_train.shape}")