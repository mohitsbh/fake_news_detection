
// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Add animation classes to elements
    document.querySelector('header').classList.add('animate-fade-in');
    document.querySelector('.card').classList.add('animate-slide-up');
    
    // Form submission handlers
    document.getElementById('text-form').addEventListener('submit', function(e) {
        e.preventDefault();
        analyzeText();
    });
    
    document.getElementById('url-form').addEventListener('submit', function(e) {
        e.preventDefault();
        analyzeUrl();
    });
    
    // Add animation to tab switching
    const tabButtons = document.querySelectorAll('.nav-link');
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const targetId = this.getAttribute('data-bs-target');
            const targetPane = document.querySelector(targetId);
            
            // Add animation class
            targetPane.classList.add('animate-fade-in');
            
            // Remove animation class after animation completes
            setTimeout(() => {
                targetPane.classList.remove('animate-fade-in');
            }, 500);
        });
    });
    
    // Add CSS animations
    const style = document.createElement('style');
    style.textContent = 
        "@keyframes fadeIn {" +
        "    from { opacity: 0; }" +
        "    to { opacity: 1; }" +
        "}" +
        
        "@keyframes slideUp {" +
        "    from { transform: translateY(20px); opacity: 0; }" +
        "    to { transform: translateY(0); opacity: 1; }" +
        "}" +
        
        "@keyframes slideIn {" +
        "    from { transform: translateX(-20px); opacity: 0; }" +
        "    to { transform: translateX(0); opacity: 1; }" +
        "}" +
        
        "@keyframes pulse {" +
        "    0% { transform: scale(1); }" +
        "    50% { transform: scale(1.05); }" +
        "    100% { transform: scale(1); }" +
        "}" +
        
        ".animate-fade-in {" +
        "    animation: fadeIn 0.5s ease-out;" +
        "}" +
        
        ".animate-slide-up {" +
        "    animation: slideUp 0.5s ease-out;" +
        "}" +
        
        ".animate-slide-in {" +
        "    animation: slideIn 0.5s ease-out;" +
        "}" +
        
        ".animate-pulse {" +
        "    animation: pulse 0.5s ease-out;" +
        "}";
    document.head.appendChild(style);
    
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
    
    // Function to show loading spinner with animation
    function showLoading() {
        const loadingElement = document.getElementById('loading');
        loadingElement.classList.remove('d-none');
        loadingElement.classList.add('animate-fade-in');
        document.getElementById('results').classList.add('d-none');
        document.getElementById('error-alert').classList.add('d-none');
    }
    
    // Function to hide loading spinner
    function hideLoading() {
        document.getElementById('loading').classList.add('d-none');
    }
    
    // Function to show error message with animation
    function showError(message) {
        const errorAlert = document.getElementById('error-alert');
        const errorMessage = document.getElementById('error-message');
        
        errorMessage.textContent = message;
        errorAlert.classList.remove('d-none');
        errorAlert.classList.add('animate-pulse');
        document.getElementById('results').classList.add('d-none');
        
        // Remove animation class after animation completes
        setTimeout(() => {
            errorAlert.classList.remove('animate-pulse');
        }, 500);
    }
    
    // Function to show results with animations 
    function showResults(data) { 
        const results = document.getElementById('results');
        if (!results) {
            console.error("Element with ID 'results' not found");
            return;
        }
        
        results.classList.remove('d-none');
        results.classList.add('animate-slide-up');
        
        const errorAlert = document.getElementById('error-alert');
        if (errorAlert) {
            errorAlert.classList.add('d-none');
        }
        
        // Update prediction badge and text
        const resultBadge = document.getElementById('result-badge');
        const predictionText = document.getElementById('prediction-text');
        const predictionBar = document.getElementById('prediction-bar');
        const confidenceText = document.getElementById('confidence-text');
        
        if (!resultBadge || !predictionText || !predictionBar || !confidenceText) {
            console.error("One or more required elements not found in the DOM");
            return;
        }
        
        resultBadge.textContent = data.prediction.toUpperCase();
        resultBadge.className = 'badge ' + (data.prediction === 'real' ? 'badge-real' : 'badge-fake');
        
        predictionText.textContent = data.prediction.toUpperCase();
        
        // Animate the progress bar
        const probability = Math.round(data.probability * 100);
        predictionBar.style.width = '0%';
        predictionBar.setAttribute('aria-valuenow', 0);
        predictionBar.className = 'progress-bar ' + (data.prediction === 'real' ? 'progress-bar-real' : 'progress-bar-fake');
        
        setTimeout(() => {
            predictionBar.style.transition = 'width 0.8s ease-in-out';
            predictionBar.style.width = probability + '%';
            predictionBar.setAttribute('aria-valuenow', probability);
        }, 200);
        
        confidenceText.textContent = data.confidence.toUpperCase();
        
        // Update explanation with typing effect
        if (data.explanation) {
            const explanationText = document.getElementById('explanation-text');
            if (!explanationText) {
                console.error("Element with ID 'explanation-text' not found");
            } else {
                const summary = data.explanation.summary || '';
                explanationText.textContent = '';
                
                let charIndex = 0;
                const typingInterval = setInterval(() => {
                    if (charIndex < summary.length) {
                        explanationText.textContent += summary.charAt(charIndex);
                        charIndex++;
                    } else {
                        clearInterval(typingInterval);
                    }
                }, 10);
            }
            
            // Update linguistic analysis with animation
            updateLinguisticMarkers('fake-markers-content', data.explanation.linguistic_analysis.fake_news_markers);
            updateLinguisticMarkers('real-markers-content', data.explanation.linguistic_analysis.credibility_markers);
            
            // Update verification tips with animation
            const tipsList = document.getElementById('verification-tips');
            if (tipsList && data.explanation.verification_tips) {
                tipsList.innerHTML = '';
                
                data.explanation.verification_tips.forEach((tip, index) => {
                    const li = document.createElement('li');
                    li.className = 'list-group-item verification-tip animate-slide-in';
                    li.style.animationDelay = (index * 100) + 'ms';
                    li.textContent = tip;
                    tipsList.appendChild(li);
                });
            }
        }
        
        // Update source verification if available
        if (data.source_verification) {
            updateSourceVerification(data);
        }
    }
    
    // Function to update source verification section
    function updateSourceVerification(data) {
        // Check if the element exists first
        let sourceVerificationDiv = document.getElementById('source-verification');
        if (!sourceVerificationDiv) {
            // Create the element if it doesn't exist
            const resultsDiv = document.getElementById('results');
            if (resultsDiv) {
                const newDiv = document.createElement('div');
                newDiv.id = 'source-verification';
                newDiv.className = 'mt-4';
                
                // Find the card in the results section and append to it
                const resultCard = resultsDiv.querySelector('.card-body');
                if (resultCard) {
                    resultCard.appendChild(newDiv);
                } else {
                    // Fallback - append to results div
                    resultsDiv.appendChild(newDiv);
                }
                
                // Now get the newly created element
                sourceVerificationDiv = document.getElementById('source-verification');
            } else {
                // If we can't find the results div, log an error and return
                console.error('Results div not found, cannot create source verification section');
                return;
            }
        }
        
        // Clear previous content
        sourceVerificationDiv.innerHTML = '';
        
        if (data.source_verification) {
            let html = '<h5>Source Verification</h5>';
            
            // Domain verification
            if (data.source_verification.domain) {
                const domain = data.source_verification.domain;
                html += `<div class="mb-2">
                    <strong>Domain:</strong> ${domain.name}
                    <span class="badge ${domain.verified ? 'bg-success' : 'bg-danger'} ms-2">
                        ${domain.verified ? 'Verified' : 'Unverified'}
                    </span>
                    ${domain.trusted_source ? '<span class="badge bg-info ms-2">Trusted Source</span>' : ''}
                </div>`;
            }
            
            // Google News verification
            if (data.source_verification.google_news) {
                const googleNews = data.source_verification.google_news;
                html += `<div class="mb-2">
                    <strong>Google News Verification:</strong>
                    <span class="badge ${googleNews.verified ? 'bg-success' : 'bg-danger'} ms-2">
                        ${googleNews.verified ? 'Verified' : 'Unverified'}
                    </span>
                    <div class="small mt-1">
                        Found ${googleNews.match_count} similar articles, including ${googleNews.trusted_match_count} from trusted sources.
                    </div>
                </div>`;
                
                // Show top matches if available
                if (googleNews.matches && googleNews.matches.length > 0) {
                    html += '<div class="mt-2"><strong>Google News Matches:</strong><ul class="list-group list-group-flush small">';
                    googleNews.matches.forEach(match => {
                        html += `<li class="list-group-item py-1">
                            ${match.title || 'Untitled'} 
                            <span class="badge ${match.trusted ? 'bg-info' : 'bg-secondary'} ms-1">
                                ${match.source || 'Unknown Source'}
                            </span>
                        </li>`;
                    });
                    html += '</ul></div>';
                }
            }
            
            // API verification
            if (data.source_verification.api) {
                const api = data.source_verification.api;
                html += `<div class="mb-2">
                    <strong>News API Verification:</strong>
                    <span class="badge ${api.verified ? 'bg-success' : 'bg-danger'} ms-2">
                        ${api.verified ? 'Verified' : 'Unverified'}
                    </span>
                    <div class="small mt-1">
                        Found ${api.match_count} similar articles, including ${api.trusted_match_count} from trusted sources.
                    </div>
                </div>`;
                
                // Show top matches if available
                if (api.top_matches && api.top_matches.length > 0) {
                    html += '<div class="mt-2"><strong>Top Matches:</strong><ul class="list-group list-group-flush small">';
                    api.top_matches.forEach(match => {
                        html += `<li class="list-group-item py-1">
                            ${match.title || 'Untitled'} 
                            <span class="badge ${match.trusted ? 'bg-info' : 'bg-secondary'} ms-1">
                                ${match.source || 'Unknown Source'}
                            </span>
                        </li>`;
                    });
                    html += '</ul></div>';
                }
            }
            
            sourceVerificationDiv.innerHTML = html;
        }
    }
    
    // Function to update linguistic markers
    function updateLinguisticMarkers(elementId, markers) {
        const container = document.getElementById(elementId);
        if (!container) {
            console.error(`Element with ID "${elementId}" not found`);
            return;
        }
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
}); // End of DOMContentLoaded event listener
    