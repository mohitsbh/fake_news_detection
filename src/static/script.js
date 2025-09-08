
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
    