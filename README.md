# Fake News Detection System

This project implements a machine learning system to detect whether a given news article is fake or real. It provides detailed explanations for the classification and can handle real-time news input.

## Features

- Fake news detection with machine learning
- Detailed explanation for classification results
- Real-time news input processing
- Web interface for easy interaction
- Responsive and modern UI design

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
└── README.md          # Project documentation
```

## Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fake_news_detection.git
   cd fake_news_detection
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Unix/MacOS
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

5. Set up environment variables:
   - Create a `.env` file in the root directory
   - Add required environment variables:
     ```
     API_KEY=your_api_key_here
     FLASK_ENV=development
     ```

## Usage

### Web Interface

Run the Flask application:
```bash
# Development
python -m flask --app src.app run

# Production
gunicorn src.app:app
```
Then open your browser and navigate to `http://localhost:5000`.

### Command Line

For direct prediction:
```bash
python src/predict.py --text "Your news article text here"
```

## Model Training

To train the model with your own dataset:
```bash
python src/models/train.py --data path/to/your/dataset
```

## Deployment

### Deploying to Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Use the following settings:
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn src.app:app`
4. Add environment variables:
   - `PYTHON_VERSION`: 3.12.0
   - `API_KEY`: Your API key

The application will be automatically deployed when you push changes to your repository.

### Health Check

The application includes a health check endpoint at `/health` that returns a 200 OK status when the service is running properly.

## Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/new-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature/new-feature`
5. Submit a pull request

## License

MIT