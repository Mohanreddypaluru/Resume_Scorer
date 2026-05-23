"""
Pydantic models for data validation and schema definition.

This module defines all Pydantic models used throughout the application for
data validation, serialization, and API request/response schemas.

Models include:
- ResumeData: Structured resume information
- JDData: Structured job description information
- CategoryScore: Individual category scoring details
- ScoreResult: Complete scoring analysis result
- AnalysisRequest: API request schema
- AnalysisResponse: API response schema
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional
from enum import Enum


class EducationLevel(str, Enum):
    """Education level enumeration."""
    HIGH_SCHOOL = "high_school"
    BACHELOR = "bachelor"
    MASTER = "master"
    PHD = "phd"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    """Confidence level enumeration."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ResumeData(BaseModel):
    """
    Structured data extracted from a resume.
    
    Attributes:
        candidate_name: Name of the candidate
        skills: List of detected skills and technologies
        job_titles: List of job titles held
        years_experience: Total inferred years of experience
        education_level: Highest education level achieved
        raw_text: Original resume text for LLM explainer
    """
    candidate_name: str = Field(default="", description="Name of the candidate")
    skills: List[str] = Field(default_factory=list, description="List of detected skills")
    job_titles: List[str] = Field(default_factory=list, description="List of job titles held")
    years_experience: float = Field(default=0.0, ge=0, description="Total years of experience")
    education_level: EducationLevel = Field(default=EducationLevel.UNKNOWN, description="Highest education level")
    raw_text: str = Field(..., description="Original resume text")


class JDData(BaseModel):
    """
    Structured data extracted from a job description.
    
    Attributes:
        job_title: Title of the position
        required_skills: List of required skills
        preferred_skills: List of preferred skills
        min_years_experience: Minimum years of experience required
        required_education: Required education level
        raw_text: Original job description text
    """
    job_title: str = Field(..., description="Title of the position")
    required_skills: List[str] = Field(default_factory=list, description="Required skills for the role")
    preferred_skills: List[str] = Field(default_factory=list, description="Preferred skills for the role")
    min_years_experience: float = Field(default=0.0, ge=0, description="Minimum years of experience required")
    required_education: EducationLevel = Field(default=EducationLevel.UNKNOWN, description="Required education level")
    raw_text: str = Field(..., description="Original job description text")


class CategoryScore(BaseModel):
    """
    Individual category scoring details.
    
    Attributes:
        name: Category name (e.g., "skills_match", "experience", "education", "keywords")
        score: Score between 0.0 and 1.0
        weight: Weight of this category in overall score
        matched: List of matched items
        missing: List of missing items
        partial: List of partially matched items
    """
    name: str = Field(..., description="Category name")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Score between 0.0 and 1.0")
    weight: float = Field(default=0.0, ge=0.0, le=1.0, description="Weight of this category")
    matched: List[str] = Field(default_factory=list, description="List of matched items")
    missing: List[str] = Field(default_factory=list, description="List of missing items")
    partial: List[str] = Field(default_factory=list, description="List of partially matched items")


class ScoreResult(BaseModel):
    """
    Complete scoring analysis result.
    
    Attributes:
        overall_score: Overall score from 0 to 100
        category_scores: List of individual category scores
        explanation: Narrative explanation from LLM explainer
        suggestions: List of specific improvement suggestions
        uncertainty_flags: List of assumptions the system made
        confidence: Confidence level in the analysis
    """
    overall_score: int = Field(default=0, ge=0, le=100, description="Overall score from 0 to 100")
    category_scores: List[CategoryScore] = Field(default_factory=list, description="Individual category scores")
    explanation: str = Field(default="", description="Narrative explanation from LLM")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    uncertainty_flags: List[str] = Field(default_factory=list, description="Assumptions made by the system")
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM, description="Confidence level")

    @model_validator(mode='after')
    def validate_weights_sum(self):
        """Validate that weights across categories sum to 1.0."""
        if self.category_scores:
            total_weight = sum(cat.weight for cat in self.category_scores)
            if not (0.99 <= total_weight <= 1.01):  # Allow small floating point errors
                raise ValueError(f"Category weights must sum to 1.0, got {total_weight}")
        return self


class AnalysisRequest(BaseModel):
    """
    API request schema for resume analysis.
    
    Attributes:
        resume_text: Text content of the resume
        jd_text: Text content of the job description
        resume_file: Optional file upload for resume
    """
    resume_text: Optional[str] = Field(default=None, description="Text content of the resume")
    jd_text: str = Field(..., description="Text content of the job description")
    resume_file: Optional[str] = Field(default=None, description="File upload for resume")


class AnalysisResponse(BaseModel):
    """
    API response schema for resume analysis.
    
    Attributes:
        success: Whether the analysis was successful
        result: ScoreResult if successful
        error: Error message if unsuccessful
    """
    success: bool = Field(..., description="Whether the analysis was successful")
    result: Optional[ScoreResult] = Field(default=None, description="Score result if successful")
    error: Optional[str] = Field(default=None, description="Error message if unsuccessful")
