"""
Weighted score combiner for aggregating individual component scores.

This module combines scores from different scoring components into a final overall
score using configurable weights. It also determines confidence levels and collects
uncertainty flags for transparency.

This function is pure Python with no API calls or randomness. Given the same inputs,
it always returns the same output, ensuring reproducibility.
"""

from typing import List, Dict

from models import CategoryScore, ResumeData, ConfidenceLevel


def combine_scores(
    category_scores: List[CategoryScore],
    uncertainty_notes: List[str],
    resume: ResumeData
) -> Dict:
    """
    Combine individual category scores into final weighted score with confidence assessment.
    
    Args:
        category_scores: List of CategoryScore objects from different scoring components
        uncertainty_notes: List of uncertainty notes from parsing (e.g., inferred values)
        resume: Parsed resume data for confidence assessment
        
    Returns:
        Dictionary containing:
        - overall_score: Overall weighted score (0-100, rounded to int)
        - category_scores: Original category scores list
        - confidence: Confidence level ("high", "medium", "low")
        - uncertainty_flags: List of uncertainty flags
        
    The function:
    1. Calculates weighted sum: overall = sum(c.score * c.weight for c in category_scores) * 100
    2. Determines confidence based on whether experience/education were inferred
    3. Collects uncertainty flags from notes and adds rule-based ones
    4. Returns structured result dictionary
    """
    # 1. Calculate weighted sum
    weighted_sum = sum(c.score * c.weight for c in category_scores)
    overall_score = int(round(weighted_sum * 100))
    overall_score = max(0, min(100, overall_score))  # Ensure within 0-100 range
    
    # 2. Determine confidence level
    confidence = _determine_confidence(uncertainty_notes, resume)
    
    # 3. Collect uncertainty flags
    uncertainty_flags = _collect_uncertainty_flags(uncertainty_notes, resume)
    
    return {
        "overall_score": overall_score,
        "category_scores": category_scores,
        "confidence": confidence,
        "uncertainty_flags": uncertainty_flags
    }


def _determine_confidence(uncertainty_notes: List[str], resume: ResumeData) -> str:
    """
    Determine confidence level based on inference indicators.
    
    Args:
        uncertainty_notes: List of uncertainty notes from parsing
        resume: Parsed resume data
        
    Returns:
        Confidence level: "high", "medium", or "low"
        
    Logic:
    - "high" if years_experience was explicit (not inferred) AND education was stated
    - "medium" if one of those was inferred
    - "low" if both were inferred or resume had very little text (< 200 chars)
    """
    # Check if experience was inferred
    experience_inferred = any(
        "infer" in note.lower() or "estimate" in note.lower() 
        for note in uncertainty_notes
    )
    
    # Check if education was stated (not unknown)
    education_stated = resume.education_level.value != "unknown"
    
    # Check if resume has very little text
    resume_too_short = len(resume.raw_text) < 200
    
    if resume_too_short:
        return ConfidenceLevel.LOW.value
    
    if not experience_inferred and education_stated:
        return ConfidenceLevel.HIGH.value
    elif experience_inferred or not education_stated:
        return ConfidenceLevel.MEDIUM.value
    else:
        return ConfidenceLevel.LOW.value


def _collect_uncertainty_flags(uncertainty_notes: List[str], resume: ResumeData) -> List[str]:
    """
    Collect uncertainty flags from notes and add rule-based ones.
    
    Args:
        uncertainty_notes: List of uncertainty notes from parsing
        resume: Parsed resume data
        
    Returns:
        List of uncertainty flags
        
    Rule-based flags:
    - If years_experience == 0: "Could not determine work experience duration"
    - If education == "unknown": "Education level not found in resume"
    - If len(resume.skills) < 3: "Very few skills detected — resume may be sparse"
    """
    flags = []
    
    # Add flags from uncertainty notes
    flags.extend(uncertainty_notes)
    
    # Add rule-based flags
    if resume.years_experience == 0:
        flags.append("Could not determine work experience duration")
    
    if resume.education_level.value == "unknown":
        flags.append("Education level not found in resume")
    
    if len(resume.skills) < 3:
        flags.append("Very few skills detected — resume may be sparse")
    
    return flags
