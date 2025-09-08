# Fake News Detection System

This project implements a machine learning system to detect whether a given news article is fake or real. It provides detailed explanations for the classification and can handle real-time news input.

## Features

- Fake news detection with machine learning
- Detailed explanation for classification results
- Real-time news input processing
- Web interface for easy interaction

## Project Structure

```
├── data/               # Dataset storage
├── models/             # Trained models
├── notebooks/          # Jupyter notebooks for exploration
├── src/                # Source code
│   ├── data/           # Data processing scripts
│   ├── models/         # Model implementation
│   ├── utils/          # Utility functions
│   ├── app.py          # Flask web application
│   └── predict.py      # Prediction script
├── tests/              # Unit tests
├── requirements.txt    # Dependencies
└── README.md           # Project documentation
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## Usage

### Web Interface

Run the Flask application:
```
python src/app.py
```
Then open your browser and navigate to `http://localhost:5000`.

### Command Line

For direct prediction:
```
python src/predict.py --text "Your news article text here"
```

## Model Training

To train the model with your own dataset:
```
python src/models/train.py --data path/to/your/dataset
```

## License

MIT