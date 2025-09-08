import os
import sys
import json
from flask import Flask, render_template, request, jsonify

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.predict import FakeNewsPredictor

app = Flask(__name__)

# Initialize predictor
predictor = None
# Flag to use simple model without TensorFlow dependencies
use_simple_model = True

@app.route('/')
def index():
    """
    Render the main page.
    """
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze news text or URL.
    """
    global predictor
    
    # Initialize predictor if not already initialized
    if predictor is None:
        try:
            if use_simple_model:
                # Use a simple model that doesn't depend on TensorFlow
                predictor = FakeNewsPredictor(model_type='random_forest')
            else:
                predictor = FakeNewsPredictor()
        except Exception as e:
            return jsonify({
                'error': f"Error initializing predictor: {str(e)}"
            })
    
    # Get input data
    data = request.get_json()
    
    # Check if input is provided
    if not data or (not data.get('text') and not data.get('url')):
        return jsonify({
            'error': "Please provide either text or URL"
        })
    
    # Make prediction
    try:
        result = predictor.predict(
            text=data.get('text'),
            url=data.get('url'),
            title=data.get('title'),
            explain=True
        )
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': f"Error making prediction: {str(e)}"
        })

@app.route('/health')
def health():
    """
    Health check endpoint.
    """
    return jsonify({
        'status': 'ok'
    })

def create_app():
    """
    Create and configure the Flask app.
    """
    # Create templates and static directories if they don't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'static'), exist_ok=True)
    
    # Create HTML templates
    create_templates()
    
    return app

def create_templates():
    """
    Create HTML templates.
    """
    # Create index.html
    index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fake News Detector</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <div class="container">
        <header class="text-center my-5">
            <h1>Fake News Detector</h1>
            <p class="lead">Analyze news articles to determine if they are real or fake</p>
        </header>
        
        <div class="row justify-content-center">
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header">
                        <ul class="nav nav-tabs card-header-tabs" id="input-tabs" role="tablist">
                            <li class="nav-item" role="presentation">
                                <button class="nav-link active" id="text-tab" data-bs-toggle="tab" data-bs-target="#text-input" type="button" role="tab" aria-controls="text-input" aria-selected="true">Text Input</button>
                            </li>
                            <li class="nav-item" role="presentation">
                                <button class="nav-link" id="url-tab" data-bs-toggle="tab" data-bs-target="#url-input" type="button" role="tab" aria-controls="url-input" aria-selected="false">URL Input</button>
                            </li>
                        </ul>
                    </div>
                    <div class="card-body">
                        <div class="tab-content" id="input-tab-content">
                            <!-- Text Input Tab -->
                            <div class="tab-pane fade show active" id="text-input" role="tabpanel" aria-labelledby="text-tab">
                                <form id="text-form">
                                    <div class="mb-3">
                                        <label for="news-title" class="form-label">News Title (Optional)</label>
                                        <input type="text" class="form-control" id="news-title" placeholder="Enter the title of the news article">
                                    </div>
                                    <div class="mb-3">
                                        <label for="news-text" class="form-label">News Text</label>
                                        <textarea class="form-control" id="news-text" rows="10" placeholder="Paste the full text of the news article here" required></textarea>
                                    </div>
                                    <button type="submit" class="btn btn-primary">Analyze</button>
                                </form>
                            </div>
                            
                            <!-- URL Input Tab -->
                            <div class="tab-pane fade" id="url-input" role="tabpanel" aria-labelledby="url-tab">
                                <form id="url-form">
                                    <div class="mb-3">
                                        <label for="news-url" class="form-label">News URL</label>
                                        <input type="url" class="form-control" id="news-url" placeholder="Enter the URL of the news article" required>
                                    </div>
                                    <button type="submit" class="btn btn-primary">Analyze</button>
                                </form>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Loading Spinner -->
                <div id="loading" class="text-center my-4 d-none">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p>Analyzing... This may take a moment.</p>
                </div>
                
                <!-- Results Section -->
                <div id="results" class="my-4 d-none">
                    <div class="card">
                        <div class="card-header d-flex justify-content-between align-items-center">
                            <h5 class="mb-0">Analysis Results</h5>
                            <span id="result-badge" class="badge"></span>
                        </div>
                        <div class="card-body">
                            <div class="row mb-4">
                                <div class="col-md-4">
                                    <div class="text-center">
                                        <h2 id="prediction-text"></h2>
                                        <div class="progress">
                                            <div id="prediction-bar" class="progress-bar" role="progressbar" style="width: 0%" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100"></div>
                                        </div>
                                        <p class="mt-2">Confidence: <span id="confidence-text"></span></p>
                                    </div>
                                </div>
                                <div class="col-md-8">
                                    <h5>Explanation</h5>
                                    <p id="explanation-text"></p>
                                </div>
                            </div>
                            
                            <div class="row">
                                <div class="col-md-6">
                                    <h5>Linguistic Analysis</h5>
                                    <div class="accordion" id="linguistic-accordion">
                                        <div class="accordion-item">
                                            <h2 class="accordion-header" id="fake-markers-heading">
                                                <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#fake-markers-collapse" aria-expanded="false" aria-controls="fake-markers-collapse">
                                                    Potential Fake News Markers
                                                </button>
                                            </h2>
                                            <div id="fake-markers-collapse" class="accordion-collapse collapse" aria-labelledby="fake-markers-heading" data-bs-parent="#linguistic-accordion">
                                                <div class="accordion-body" id="fake-markers-content">
                                                    <!-- Content will be inserted here -->
                                                </div>
                                            </div>
                                        </div>
                                        <div class="accordion-item">
                                            <h2 class="accordion-header" id="real-markers-heading">
                                                <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#real-markers-collapse" aria-expanded="false" aria-controls="real-markers-collapse">
                                                    Credibility Indicators
                                                </button>
                                            </h2>
                                            <div id="real-markers-collapse" class="accordion-collapse collapse" aria-labelledby="real-markers-heading" data-bs-parent="#linguistic-accordion">
                                                <div class="accordion-body" id="real-markers-content">
                                                    <!-- Content will be inserted here -->
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <h5>Verification Tips</h5>
                                    <ul id="verification-tips" class="list-group">
                                        <!-- Tips will be inserted here -->
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Error Alert -->
                <div id="error-alert" class="alert alert-danger my-4 d-none" role="alert">
                    <strong>Error:</strong> <span id="error-message"></span>
                </div>
            </div>
        </div>
    </div>
    
    <footer class="text-center py-4 mt-5 bg-light">
        <p>Fake News Detection System</p>
    </footer>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="{{ url_for('static', filename='script.js') }}"></script>
</body>
</html>
    """
    
    # Create CSS file
    css = """
/* Custom styles for the Fake News Detector */

body {
    background-color: #f8f9fa;
}

.card {
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

#results .card-header {
    background-color: #f8f9fa;
}

.badge.badge-real {
    background-color: #28a745;
    color: white;
}

.badge.badge-fake {
    background-color: #dc3545;
    color: white;
}

.progress {
    height: 10px;
    margin-top: 10px;
}

.progress-bar-real {
    background-color: #28a745;
}

.progress-bar-fake {
    background-color: #dc3545;
}

.marker-category {
    font-weight: bold;
    margin-top: 10px;
}

.marker-terms {
    margin-left: 15px;
    margin-bottom: 10px;
}

.verification-tip {
    padding: 8px 15px;
}

#loading {
    margin: 30px 0;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    #results .row {
        flex-direction: column;
    }
    
    #results .col-md-4,
    #results .col-md-8,
    #results .col-md-6 {
        width: 100%;
        margin-bottom: 20px;
    }
}
    """
    
    # Create JavaScript file
    js = """
// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Form submission handlers
    document.getElementById('text-form').addEventListener('submit', function(e) {
        e.preventDefault();
        analyzeText();
    });
    
    document.getElementById('url-form').addEventListener('submit', function(e) {
        e.preventDefault();
        analyzeUrl();
    });
    
    // Function to analyze text input
    function analyzeText() {
        const title = document.getElementById('news-title').value;
        const text = document.getElementById('news-text').value;
        
        if (!text) {
            showError('Please enter the news text.');
            return;
        }
        
        // Show loading spinner
        showLoading();
        
        // Send request to backend
        fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                title: title,
                text: text
            })
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            if (data.error) {
                showError(data.error);
            } else {
                showResults(data);
            }
        })
        .catch(error => {
            hideLoading();
            showError('An error occurred while analyzing the text. Please try again.');
            console.error('Error:', error);
        });
    }
    
    // Function to analyze URL input
    function analyzeUrl() {
        const url = document.getElementById('news-url').value;
        
        if (!url) {
            showError('Please enter a URL.');
            return;
        }
        
        // Show loading spinner
        showLoading();
        
        // Send request to backend
        fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                url: url
            })
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            if (data.error) {
                showError(data.error);
            } else {
                showResults(data);
            }
        })
        .catch(error => {
            hideLoading();
            showError('An error occurred while analyzing the URL. Please try again.');
            console.error('Error:', error);
        });
    }
    
    // Function to show loading spinner
    function showLoading() {
        document.getElementById('loading').classList.remove('d-none');
        document.getElementById('results').classList.add('d-none');
        document.getElementById('error-alert').classList.add('d-none');
    }
    
    // Function to hide loading spinner
    function hideLoading() {
        document.getElementById('loading').classList.add('d-none');
    }
    
    // Function to show error message
    function showError(message) {
        const errorAlert = document.getElementById('error-alert');
        const errorMessage = document.getElementById('error-message');
        
        errorMessage.textContent = message;
        errorAlert.classList.remove('d-none');
        document.getElementById('results').classList.add('d-none');
    }
    
    // Function to show results
    function showResults(data) {
        const results = document.getElementById('results');
        results.classList.remove('d-none');
        document.getElementById('error-alert').classList.add('d-none');
        
        // Update prediction badge and text
        const resultBadge = document.getElementById('result-badge');
        const predictionText = document.getElementById('prediction-text');
        const predictionBar = document.getElementById('prediction-bar');
        const confidenceText = document.getElementById('confidence-text');
        
        resultBadge.textContent = data.prediction.toUpperCase();
        resultBadge.className = 'badge ' + (data.prediction === 'real' ? 'badge-real' : 'badge-fake');
        
        predictionText.textContent = data.prediction.toUpperCase();
        
        const probability = Math.round(data.probability * 100);
        predictionBar.style.width = probability + '%';
        predictionBar.setAttribute('aria-valuenow', probability);
        predictionBar.className = 'progress-bar ' + (data.prediction === 'real' ? 'progress-bar-real' : 'progress-bar-fake');
        
        confidenceText.textContent = data.confidence.toUpperCase();
        
        // Update explanation
        if (data.explanation) {
            document.getElementById('explanation-text').textContent = data.explanation.summary;
            
            // Update linguistic analysis
            updateLinguisticMarkers('fake-markers-content', data.explanation.linguistic_analysis.fake_news_markers);
            updateLinguisticMarkers('real-markers-content', data.explanation.linguistic_analysis.credibility_markers);
            
            // Update verification tips
            const tipsList = document.getElementById('verification-tips');
            tipsList.innerHTML = '';
            
            data.explanation.verification_tips.forEach(tip => {
                const li = document.createElement('li');
                li.className = 'list-group-item verification-tip';
                li.textContent = tip;
                tipsList.appendChild(li);
            });
        }
    }
    
    // Function to update linguistic markers
    function updateLinguisticMarkers(elementId, markers) {
        const container = document.getElementById(elementId);
        container.innerHTML = '';
        
        if (Object.keys(markers).length === 0) {
            container.textContent = 'None detected';
            return;
        }
        
        for (const category in markers) {
            const categoryDiv = document.createElement('div');
            categoryDiv.className = 'marker-category';
            categoryDiv.textContent = formatCategoryName(category) + ':';
            container.appendChild(categoryDiv);
            
            const termsList = document.createElement('div');
            termsList.className = 'marker-terms';
            termsList.textContent = markers[category].join(', ');
            container.appendChild(termsList);
        }
    }
    
    // Function to format category name
    function formatCategoryName(name) {
        return name.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
    }
});
    """
    
    # Write files
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    
    with open(os.path.join(templates_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(index_html)
    
    with open(os.path.join(static_dir, 'style.css'), 'w', encoding='utf-8') as f:
        f.write(css)
    
    with open(os.path.join(static_dir, 'script.js'), 'w', encoding='utf-8') as f:
        f.write(js)

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)