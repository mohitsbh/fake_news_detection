
// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Add smooth scroll behavior
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // Form submission handlers with input validation
    const textForm = document.getElementById('text-form');
    const urlForm = document.getElementById('url-form');
    
    textForm.addEventListener('submit', function(e) {
        e.preventDefault();
        if (validateTextForm()) {
            analyzeText();
        }
    });
    
    urlForm.addEventListener('submit', function(e) {
        e.preventDefault();
        if (validateUrlForm()) {
            analyzeUrl();
        }
    });

    // Real-time character count for text input
    const newsText = document.getElementById('news-text');
    const charCount = document.createElement('small');
    charCount.className = 'text-muted ms-2';
    newsText.parentNode.appendChild(charCount);

    newsText.addEventListener('input', function() {
        const remaining = 5000 - this.value.length;
        charCount.textContent = `${remaining} characters remaining`;
        charCount.className = `text-muted ms-2 ${remaining < 100 ? 'text-danger' : ''}`;
    });

    // Form validation functions
    function validateTextForm() {
        const title = document.getElementById('news-title').value;
        const text = document.getElementById('news-text').value;
        
        if (!text.trim()) {
            showError('Please enter the news text.', 'text-error');
            return false;
        }
        return true;
    }

    function validateUrlForm() {
        const url = document.getElementById('news-url').value;
        
        if (!url.trim()) {
            showError('Please enter a URL.', 'url-error');
            return false;
        }

        try {
            new URL(url);
            return true;
        } catch {
            showError('Please enter a valid URL.', 'url-error');
            return false;
        }
    }
    
    // Function to analyze text input
    function analyzeText() {
        const title = document.getElementById('news-title').value;
        const text = document.getElementById('news-text').value;
        
        showLoading();
        
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
        .then(handleResponse)
        .catch(handleError);
    }
    
    // Function to analyze URL input
    function analyzeUrl() {
        const url = document.getElementById('news-url').value;
        
        showLoading();
        
        fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                url: url
            })
        })
        .then(handleResponse)
        .catch(handleError);
    }

    // Response handlers
    async function handleResponse(response) {
        const data = await response.json();
        hideLoading();
        if (data.error) {
            showError(data.error);
        } else {
            showResults(data);
            scrollToResults();
        }
    }

    function handleError(error) {
        hideLoading();
        showError('An error occurred while analyzing. Please try again.');
        console.error('Error:', error);
    }
    
    // UI feedback functions
    function showLoading() {
        const loading = document.getElementById('loading');
        loading.classList.remove('d-none');
        loading.classList.add('fade-in');
        document.getElementById('results').classList.add('d-none');
        document.getElementById('error-alert').classList.add('d-none');
    }
    
    function hideLoading() {
        const loading = document.getElementById('loading');
        loading.classList.add('fade-out');
        setTimeout(() => {
            loading.classList.add('d-none');
            loading.classList.remove('fade-out');
        }, 300);
    }
    
    function showError(message, elementId = 'error-alert') {
        const errorAlert = document.getElementById(elementId);
        const errorMessage = errorAlert.querySelector('.error-message') || errorAlert;
        
        errorMessage.textContent = message;
        errorAlert.classList.remove('d-none');
        errorAlert.classList.add('fade-in');
        document.getElementById('results').classList.add('d-none');

        setTimeout(() => {
            errorAlert.classList.remove('fade-in');
        }, 300);
    }
    
    // Function to show results with animations
    function showResults(data) {
        const results = document.getElementById('results');
        results.classList.remove('d-none');
        results.classList.add('fade-in');
        document.getElementById('error-alert').classList.add('d-none');
        
        // Update prediction elements with animations
        updatePredictionElements(data);
        
        // Update explanation and analysis with animations
        if (data.explanation) {
            updateExplanation(data.explanation);
        }

        setTimeout(() => {
            results.classList.remove('fade-in');
        }, 300);
    }

    function updatePredictionElements(data) {
        const resultBadge = document.getElementById('result-badge');
        const predictionText = document.getElementById('prediction-text');
        const predictionBar = document.getElementById('prediction-bar');
        const confidenceText = document.getElementById('confidence-text');
        
        // Update badge and text with animation
        resultBadge.className = 'badge slide-in ' + (data.prediction === 'real' ? 'badge-real' : 'badge-fake');
        resultBadge.textContent = data.prediction.toUpperCase();
        
        predictionText.textContent = data.prediction.toUpperCase();
        
        // Animate probability bar
        const probability = Math.round(data.probability * 100);
        predictionBar.style.width = '0%';
        predictionBar.className = 'progress-bar ' + (data.prediction === 'real' ? 'progress-bar-real' : 'progress-bar-fake');
        
        setTimeout(() => {
            predictionBar.style.width = probability + '%';
            predictionBar.setAttribute('aria-valuenow', probability);
        }, 100);
        
        confidenceText.textContent = data.confidence.toUpperCase();
    }

    function updateExplanation(explanation) {
        // Update explanation text with fade effect
        const explanationText = document.getElementById('explanation-text');
        explanationText.style.opacity = '0';
        setTimeout(() => {
            explanationText.textContent = explanation.summary;
            explanationText.style.opacity = '1';
        }, 150);
        
        // Update linguistic analysis with slide effect
        updateLinguisticMarkers('fake-markers-content', explanation.linguistic_analysis.fake_news_markers);
        updateLinguisticMarkers('real-markers-content', explanation.linguistic_analysis.credibility_markers);
        
        // Update verification tips with fade-in effect
        updateVerificationTips(explanation.verification_tips);
    }
    
    function updateLinguisticMarkers(elementId, markers) {
        const container = document.getElementById(elementId);
        container.innerHTML = '';
        container.style.opacity = '0';
        
        if (Object.keys(markers).length === 0) {
            container.textContent = 'None detected';
        } else {
            for (const category in markers) {
                const categoryDiv = document.createElement('div');
                categoryDiv.className = 'marker-category mb-2';
                categoryDiv.textContent = formatCategoryName(category) + ':';
                container.appendChild(categoryDiv);
                
                const termsList = document.createElement('div');
                termsList.className = 'marker-terms';
                termsList.innerHTML = markers[category].map(term => 
                    `<span class="badge bg-light text-dark me-2 mb-2">${term}</span>`
                ).join('');
                container.appendChild(termsList);
            }
        }
        
        setTimeout(() => {
            container.style.opacity = '1';
        }, 150);
    }
    
    function updateVerificationTips(tips) {
        const tipsList = document.getElementById('verification-tips');
        tipsList.innerHTML = '';
        
        tips.forEach((tip, index) => {
            const li = document.createElement('li');
            li.className = 'list-group-item verification-tip fade-in-delay';
            li.style.animationDelay = `${index * 100}ms`;
            li.innerHTML = `<i class="bi bi-check-circle-fill me-2 text-success"></i>${tip}`;
            tipsList.appendChild(li);
        });
    }
    
    function formatCategoryName(name) {
        return name.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }

    function scrollToResults() {
        const results = document.getElementById('results');
        results.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
});
    