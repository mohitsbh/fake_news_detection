import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import FakeNewsPredictor

class TestFakeNewsPredictor(unittest.TestCase):
    """Test cases for the FakeNewsPredictor class."""
    
    @patch('src.predict.FakeNewsPredictor._load_model')
    def setUp(self, mock_load_model):
        """Set up test fixtures."""
        # Mock the model loading to avoid actual file operations during tests
        mock_load_model.return_value = (MagicMock(), MagicMock())
        self.predictor = FakeNewsPredictor()
        
        # Mock the classifier's predict_proba method
        self.predictor.classifier.predict_proba = MagicMock()
        
    def test_predict_with_text_fake_news(self):
        """Test prediction with fake news text."""
        # Mock the prediction to return fake news probability
        self.predictor.classifier.predict_proba.return_value = [[0.2, 0.8]]  # [real, fake]
        
        # Test text with fake news characteristics
        text = "SHOCKING: Scientists discover miracle cure that Big Pharma doesn't want you to know about!!!"
        
        # Call predict method
        result = self.predictor.predict(text=text, explain=True)
        
        # Assertions
        self.assertEqual(result['prediction'], 'fake')
        self.assertGreater(result['probability'], 0.5)
        self.assertIn('confidence', result)
        self.assertIn('explanation', result)
        self.assertIn('summary', result['explanation'])
        self.assertIn('linguistic_analysis', result['explanation'])
        self.assertIn('verification_tips', result['explanation'])
    
    def test_predict_with_text_real_news(self):
        """Test prediction with real news text."""
        # Mock the prediction to return real news probability
        self.predictor.classifier.predict_proba.return_value = [[0.9, 0.1]]  # [real, fake]
        
        # Test text with real news characteristics
        text = "According to a study published in the Journal of Medicine, researchers found a correlation between exercise and improved cardiovascular health."
        
        # Call predict method
        result = self.predictor.predict(text=text, explain=True)
        
        # Assertions
        self.assertEqual(result['prediction'], 'real')
        self.assertGreater(result['probability'], 0.5)
        self.assertIn('confidence', result)
        self.assertIn('explanation', result)
    
    @patch('src.predict.FakeNewsPredictor._extract_text_from_url')
    def test_predict_with_url(self, mock_extract):
        """Test prediction with URL."""
        # Mock URL text extraction
        mock_extract.return_value = {
            'title': 'Test News Article',
            'text': 'This is a test news article with factual content and proper citations.'
        }
        
        # Mock the prediction to return real news probability
        self.predictor.classifier.predict_proba.return_value = [[0.85, 0.15]]  # [real, fake]
        
        # Call predict method with URL
        result = self.predictor.predict(url='https://example.com/news', explain=True)
        
        # Assertions
        self.assertEqual(result['prediction'], 'real')
        self.assertGreater(result['probability'], 0.5)
        self.assertIn('confidence', result)
        self.assertIn('explanation', result)
        
        # Verify URL extraction was called
        mock_extract.assert_called_once_with('https://example.com/news')
    
    def test_analyze_linguistic_markers(self):
        """Test linguistic marker analysis."""
        # Test text with various linguistic markers
        text = "BREAKING NEWS!!! Scientists SHOCKED by this MIRACLE cure! The mainstream media won't tell you this SECRET!"
        
        # Call the analysis method
        markers = self.predictor._analyze_linguistic_markers(text)
        
        # Assertions
        self.assertIsInstance(markers, dict)
        self.assertIn('fake_news_markers', markers)
        self.assertIn('credibility_markers', markers)
        
        # Check for specific fake news markers
        fake_markers = markers['fake_news_markers']
        self.assertIsInstance(fake_markers, dict)
        
        # The text contains sensationalist language and excessive punctuation
        has_sensationalist = any('sensationalist' in key.lower() for key in fake_markers.keys())
        has_punctuation = any('punctuation' in key.lower() for key in fake_markers.keys())
        
        self.assertTrue(has_sensationalist or has_punctuation)
    
    def test_generate_verification_tips(self):
        """Test generation of verification tips."""
        # Call the method
        tips = self.predictor._generate_verification_tips('fake')
        
        # Assertions
        self.assertIsInstance(tips, list)
        self.assertGreater(len(tips), 0)
        
        # Tips should be different for real news
        real_tips = self.predictor._generate_verification_tips('real')
        self.assertNotEqual(tips, real_tips)

if __name__ == '__main__':
    unittest.main()