"""
Deterministic rule-based scorer for resume analysis.

This module implements rule-based scoring logic that evaluates how well a resume
matches a job description based on deterministic criteria. It provides transparent,
explainable scoring without using LLMs for the core scoring logic.

All functions in this module are pure functions with no side effects - they take
inputs and return outputs without modifying external state or making API calls.
This ensures reproducibility and testability.
"""

import re
from typing import List, Set

from models import ResumeData, JDData, CategoryScore, EducationLevel


# Education level ladder for comparison
EDUCATION_LADDER = {
    EducationLevel.UNKNOWN: 0,
    EducationLevel.HIGH_SCHOOL: 1,
    EducationLevel.BACHELOR: 2,
    EducationLevel.MASTER: 3,
    EducationLevel.PHD: 4,
}


def score_experience(resume: ResumeData, jd: JDData) -> CategoryScore:
    """
    Score experience match against job requirements using deterministic rules.
    
    Args:
        resume: Parsed resume data containing years_experience
        jd: Parsed job description data containing min_years_experience
        
    Returns:
        CategoryScore with name="experience", score between 0-1, weight=0.20,
        and matched/missing lists describing the experience match
        
    Scoring logic:
    - ratio = resume.years_experience / max(jd.min_years_experience, 1)
    - score = min(ratio, 1.0) if ratio >= 0.7 else ratio * 0.7
    - matched = [f"{resume.years_experience:.1f} years"] if ratio >= 1.0 else []
    - missing = [f"Need {jd.min_years_experience}y, have {resume.years_experience:.1f}y"] if ratio < 0.8 else []
    """
    required_years = max(jd.min_years_experience, 1)
    actual_years = resume.years_experience
    
    ratio = actual_years / required_years
    
    # Calculate score based on ratio
    if ratio >= 0.7:
        score = min(ratio, 1.0)
    else:
        score = ratio * 0.7
    
    # Determine matched and missing items
    matched = []
    missing = []
    
    if ratio >= 1.0:
        matched.append(f"{actual_years:.1f} years")
    elif ratio < 0.8:
        missing.append(f"Need {jd.min_years_experience}y, have {actual_years:.1f}y")
    
    return CategoryScore(
        name="experience",
        score=score,
        weight=0.20,
        matched=matched,
        missing=missing,
        partial=[]
    )


def score_education(resume: ResumeData, jd: JDData) -> CategoryScore:
    """
    Score education match against job requirements using a level ladder.
    
    Args:
        resume: Parsed resume data containing education_level
        jd: Parsed job description data containing required_education
        
    Returns:
        CategoryScore with name="education", score between 0-1, weight=0.15,
        and matched/missing lists describing the education match
        
    Scoring logic:
    - Education ladder: unknown=0, high_school=1, bachelor=2, master=3, phd=4
    - candidate_level = ladder[resume.education_level]
    - required_level = ladder[jd.required_education]
    - score = 1.0 if candidate >= required, 0.5 if one level below, 0.0 if two+ below
    """
    candidate_level = EDUCATION_LADDER.get(resume.education_level, 0)
    required_level = EDUCATION_LADDER.get(jd.required_education, 0)
    
    level_diff = candidate_level - required_level
    
    # Calculate score based on level difference
    if level_diff >= 0:
        score = 1.0
    elif level_diff == -1:
        score = 0.5
    else:
        score = 0.0
    
    # Determine matched and missing items
    matched = []
    missing = []
    
    if score >= 1.0:
        matched.append(f"{resume.education_level.value}")
    elif score == 0.0:
        missing.append(f"Need {jd.required_education.value}, have {resume.education_level.value}")
    
    return CategoryScore(
        name="education",
        score=score,
        weight=0.15,
        matched=matched,
        missing=missing,
        partial=[]
    )


def tokenize_text(text: str) -> Set[str]:
    """
    Tokenize text by splitting on non-alphabetic characters and lowercasing.
    
    Args:
        text: Input text to tokenize
        
    Returns:
        Set of lowercase tokens (alphabetic sequences)
    """
    # Split on non-alphabetic characters and filter empty strings
    tokens = re.split(r'[^a-zA-Z]+', text.lower())
    return {token for token in tokens if token}


def extract_keywords(skills: List[str], job_title: str) -> Set[str]:
    """
    Extract unigrams and bigrams from skills and job title.
    
    Args:
        skills: List of skill strings
        job_title: Job title string
        
    Returns:
        Set of unigrams and bigrams
    """
    keywords = set()
    
    # Add unigrams from skills
    for skill in skills:
        skill_lower = skill.lower()
        # Split skill into words and add as unigrams
        words = re.split(r'[^a-zA-Z]+', skill_lower)
        keywords.update(words)
        # Add the full skill as a keyword
        keywords.add(skill_lower)
    
    # Add unigrams from job title
    title_words = re.split(r'[^a-zA-Z]+', job_title.lower())
    keywords.update(title_words)
    
    # Add bigrams from job title
    for i in range(len(title_words) - 1):
        bigram = f"{title_words[i]} {title_words[i+1]}"
        keywords.add(bigram)
    
    return keywords


def score_keywords(resume: ResumeData, jd: JDData) -> CategoryScore:
    """
    Score keyword match between resume and job description.
    
    Args:
        resume: Parsed resume data containing raw_text
        jd: Parsed job description data containing preferred_skills and job_title
        
    Returns:
        CategoryScore with name="keywords", score between 0-1, weight=0.15,
        and matched/missing lists describing keyword matches
        
    Scoring logic:
    - Tokenize resume.raw_text and jd.raw_text (lowercase, split on non-alpha)
    - Extract jd bigrams and unigrams from preferred_skills + job_title
    - Count how many appear in resume tokens
    - score = matches / max(total_jd_keywords, 1), capped at 1.0
    """
    # Tokenize resume text
    resume_tokens = tokenize_text(resume.raw_text)
    
    # Extract keywords from JD (preferred skills + job title)
    jd_keywords = extract_keywords(jd.preferred_skills, jd.job_title)
    
    if not jd_keywords:
        # No keywords to match against
        return CategoryScore(
            name="keywords",
            score=0.5,
            weight=0.15,
            matched=[],
            missing=[],
            partial=[]
        )
    
    # Count matches
    matched_keywords = []
    missing_keywords = []
    
    for keyword in jd_keywords:
        # Check if keyword (or its parts) appear in resume tokens
        keyword_words = set(keyword.split())
        if keyword_words.issubset(resume_tokens):
            matched_keywords.append(keyword)
        else:
            missing_keywords.append(keyword)
    
    # Calculate score
    total_keywords = len(jd_keywords)
    matches = len(matched_keywords)
    score = matches / max(total_keywords, 1)
    score = min(score, 1.0)  # Cap at 1.0
    
    return CategoryScore(
        name="keywords",
        score=score,
        weight=0.15,
        matched=matched_keywords[:10],  # Limit to top 10 for display
        missing=missing_keywords[:10],  # Limit to top 10 for display
        partial=[]
    )


def score_responsibilities(resume: ResumeData, jd: JDData) -> CategoryScore:
    """
    Score responsibility alignment using TF-IDF cosine similarity.
    
    This function uses scikit-learn's TF-IDF vectorizer to compute semantic
    similarity between resume and job description responsibility sections.
    
    Args:
        resume: Parsed resume data containing raw_text
        jd: Parsed job description data containing raw_text
        
    Returns:
        CategoryScore with name="responsibilities", score between 0-1, weight=0.15
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    try:
        # Use full texts for responsibility matching
        texts = [resume.raw_text, jd.raw_text]
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
            min_df=1,
            max_features=1000
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Compute cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Ensure similarity is in [0, 1] range
        similarity = max(0.0, min(1.0, similarity))
        
        return CategoryScore(
            name="responsibilities",
            score=similarity,
            weight=0.15,
            matched=[],
            missing=[],
            partial=[]
        )
        
    except Exception as e:
        # Fallback to neutral score on error
        return CategoryScore(
            name="responsibilities",
            score=0.5,
            weight=0.15,
            matched=[],
            missing=[],
            partial=[]
        )
