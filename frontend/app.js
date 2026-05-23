/**
 * Resume Scorer - Frontend JavaScript
 * Handles form submission, API calls, and result rendering
 */

const API_BASE_URL = 'http://localhost:5000/api';

// DOM Elements
const resumeFileInput = document.getElementById('resumeFile');
const jdTextInput = document.getElementById('jdText');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingState = document.getElementById('loadingState');
const errorState = document.getElementById('errorState');
const errorMessage = document.getElementById('errorMessage');
const resultsSection = document.getElementById('resultsSection');

// Event Listeners
analyzeBtn.addEventListener('click', handleAnalyze);

/**
 * Handle analyze button click
 */
async function handleAnalyze() {
    const resumeFile = resumeFileInput.files[0];
    const jdText = jdTextInput.value.trim();

    // Validation
    if (!resumeFile) {
        showError('Please upload a resume file');
        return;
    }

    if (!jdText) {
        showError('Please provide a job description');
        return;
    }

    // Show loading state
    showLoading();
    hideError();
    hideResults();

    try {
        // Create form data
        const formData = new FormData();
        formData.append('resume', resumeFile);
        formData.append('jd_text', jdText);

        // Make API call
        const response = await fetch(`${API_BASE_URL}/score`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            displayResults(data);
        } else {
            showError(data.error || 'Failed to analyze resume');
        }
    } catch (error) {
        showError('Failed to connect to server. Please ensure the backend is running.');
        console.error('Error:', error);
    } finally {
        hideLoading();
    }
}

/**
 * Display scoring results
 */
function displayResults(data) {
    // Overall score
    const scoreCircle = document.getElementById('scoreCircle');
    const overallScore = document.getElementById('overallScore');
    
    overallScore.textContent = data.overall_score;
    
    // Color code the score circle based on score
    if (data.overall_score >= 80) {
        scoreCircle.style.background = 'linear-gradient(135deg, #198754 0%, #146c43 100%)';
        document.querySelector('.score-number').style.color = '#198754';
    } else if (data.overall_score >= 60) {
        scoreCircle.style.background = 'linear-gradient(135deg, #0d6efd 0%, #0a58ca 100%)';
        document.querySelector('.score-number').style.color = '#0d6efd';
    } else if (data.overall_score >= 40) {
        scoreCircle.style.background = 'linear-gradient(135deg, #ffc107 0%, #e0a800 100%)';
        document.querySelector('.score-number').style.color = '#e0a800';
    } else {
        scoreCircle.style.background = 'linear-gradient(135deg, #dc3545 0%, #b02a37 100%)';
        document.querySelector('.score-number').style.color = '#dc3545';
    }

    // Confidence
    document.getElementById('confidenceLevel').textContent = data.confidence.toUpperCase();
    document.getElementById('confidenceScore').textContent = (data.confidence_score * 100).toFixed(0) + '%';

    // Explanation
    document.getElementById('explanationText').textContent = data.explanation;

    // Category scores
    displayCategoryScores(data.category_scores);

    // Skills
    displaySkills(data.matched_skills, data.missing_skills, data.unmatched_skills);

    // Suggestions
    displaySuggestions(data.suggestions);

    // Strengths
    displayStrengths(data.strengths);

    // Gaps
    displayGaps(data.gaps);

    // Uncertainties
    displayUncertainties(data.uncertainties);

    // Show results with animation
    resultsSection.style.display = 'block';
    resultsSection.classList.add('fade-in');

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * Display category scores
 */
function displayCategoryScores(categoryScores) {
    const container = document.getElementById('categoryScores');
    container.innerHTML = '';

    for (const [name, scoreData] of Object.entries(categoryScores)) {
        const scorePercentage = Math.round(scoreData.score * 100);
        
        const html = `
            <div class="category-score-item">
                <div class="category-name">
                    <span>${formatCategoryName(name)}</span>
                    <span class="category-score-value">${scorePercentage}%</span>
                </div>
                <div class="progress">
                    <div class="progress-bar" role="progressbar" style="width: ${scorePercentage}%" 
                         aria-valuenow="${scorePercentage}" aria-valuemin="0" aria-valuemax="100">
                        ${scorePercentage}%
                    </div>
                </div>
                <div class="category-details">
                    ${scoreData.matched && scoreData.matched.length > 0 ? 
                        `<span class="matched"><i class="bi bi-check-circle"></i> Matched: ${scoreData.matched.join(', ')}</span>` : ''}
                    ${scoreData.missing && scoreData.missing.length > 0 ? 
                        `<span class="missing"><i class="bi bi-x-circle"></i> Missing: ${scoreData.missing.join(', ')}</span>` : ''}
                    ${scoreData.partial && scoreData.partial.length > 0 ? 
                        `<span class="partial"><i class="bi bi-dash-circle"></i> Partial: ${scoreData.partial.join(', ')}</span>` : ''}
                </div>
            </div>
        `;
        container.innerHTML += html;
    }
}

/**
 * Display skills (matched, missing, and unmatched)
 */
function displaySkills(matchedSkills, missingSkills, unmatchedSkills) {
    const matchedContainer = document.getElementById('matchedSkills');
    const missingContainer = document.getElementById('missingSkills');
    const unmatchedContainer = document.getElementById('unmatchedSkills');
    
    matchedContainer.innerHTML = '';
    missingContainer.innerHTML = '';
    unmatchedContainer.innerHTML = '';

    // Display matched skills
    if (matchedSkills && matchedSkills.length > 0) {
        matchedSkills.forEach(skill => {
            matchedContainer.innerHTML += `<span class="skill-tag matched">${skill}</span>`;
        });
    } else {
        matchedContainer.innerHTML = '<div class="empty-state"><p>No matched skills found</p></div>';
    }

    // Display missing skills
    if (missingSkills && missingSkills.length > 0) {
        missingSkills.forEach(skill => {
            missingContainer.innerHTML += `<span class="skill-tag missing">${skill}</span>`;
        });
    } else {
        missingContainer.innerHTML = '<div class="empty-state"><p>No missing skills - great job!</p></div>';
    }

    // Display unmatched skills (skills in resume but not in JD)
    if (unmatchedSkills && unmatchedSkills.length > 0) {
        unmatchedSkills.forEach(skill => {
            unmatchedContainer.innerHTML += `<span class="skill-tag unmatched">${skill}</span>`;
        });
    } else {
        unmatchedContainer.innerHTML = '<div class="empty-state"><p>No unmatched skills</p></div>';
    }
}

/**
 * Display suggestions
 */
function displaySuggestions(suggestions) {
    const container = document.getElementById('suggestionsList');
    container.innerHTML = '';

    if (suggestions && suggestions.length > 0) {
        suggestions.forEach(suggestion => {
            container.innerHTML += `
                <li class="list-group-item suggestion-item">
                    <i class="bi bi-lightbulb-fill"></i>
                    <span>${suggestion}</span>
                </li>
            `;
        });
    } else {
        container.innerHTML = '<div class="empty-state"><p>No suggestions - your resume looks great!</p></div>';
    }
}

/**
 * Display strengths
 */
function displayStrengths(strengths) {
    const container = document.getElementById('strengthsList');
    container.innerHTML = '';

    if (strengths && strengths.length > 0) {
        strengths.forEach(strength => {
            container.innerHTML += `
                <li class="list-group-item strength-item">
                    <i class="bi bi-star-fill"></i>
                    <div>
                        <strong>${strength.category}</strong>
                        <p class="mb-0 small text-muted">${strength.description}</p>
                    </div>
                </li>
            `;
        });
    } else {
        container.innerHTML = '<div class="empty-state"><p>No significant strengths identified</p></div>';
    }
}

/**
 * Display gaps
 */
function displayGaps(gaps) {
    const container = document.getElementById('gapsList');
    container.innerHTML = '';

    if (gaps && gaps.length > 0) {
        gaps.forEach(gap => {
            container.innerHTML += `
                <li class="list-group-item gap-item">
                    <i class="bi bi-exclamation-circle"></i>
                    <div>
                        <strong>${gap.category}</strong>
                        <p class="mb-0 small text-muted">${gap.description}</p>
                    </div>
                    <span class="gap-severity ${gap.severity}">${gap.severity}</span>
                </li>
            `;
        });
    } else {
        container.innerHTML = '<div class="empty-state"><p>No gaps identified - excellent match!</p></div>';
    }
}

/**
 * Display uncertainties
 */
function displayUncertainties(uncertainties) {
    const container = document.getElementById('uncertaintiesList');
    container.innerHTML = '';

    if (uncertainties && uncertainties.length > 0) {
        uncertainties.forEach(uncertainty => {
            container.innerHTML += `
                <div class="uncertainty-item ${uncertainty.severity}">
                    <div class="uncertainty-type">${uncertainty.type}</div>
                    <div class="uncertainty-description">${uncertainty.description}</div>
                </div>
            `;
        });
    } else {
        container.innerHTML = '<div class="empty-state"><p>No uncertainties detected - high confidence analysis</p></div>';
    }
}

/**
 * Format category name for display
 */
function formatCategoryName(name) {
    return name
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

/**
 * Show loading state
 */
function showLoading() {
    loadingState.style.display = 'block';
    analyzeBtn.disabled = true;
}

/**
 * Hide loading state
 */
function hideLoading() {
    loadingState.style.display = 'none';
    analyzeBtn.disabled = false;
}

/**
 * Show error message
 */
function showError(message) {
    errorMessage.textContent = message;
    errorState.style.display = 'block';
}

/**
 * Hide error message
 */
function hideError() {
    errorState.style.display = 'none';
}

/**
 * Hide results section
 */
function hideResults() {
    resultsSection.style.display = 'none';
    resultsSection.classList.remove('fade-in');
}

/**
 * Initialize on page load
 */
document.addEventListener('DOMContentLoaded', () => {
    console.log('Resume Scorer initialized');
});
