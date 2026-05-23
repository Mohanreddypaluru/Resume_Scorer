"""
FastAPI application entry point for the Resume Scorer system.

This module sets up and runs the FastAPI web server that provides REST API endpoints
for resume analysis. It handles:
- Resume and job description text submission
- Score calculation using the multi-component pipeline
- Explanation generation using LLM
- CORS support for local development
"""

import asyncio
from typing import Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from parser import parse_resume, parse_jd, get_uncertainty_notes, clear_uncertainty_notes
from embedder import compute_skills_score, compute_semantic_score
from scorer import score_experience, score_education, score_keywords, score_responsibilities
from combiner import combine_scores
from explainer import generate_explanation
from models import ScoreResult, ConfidenceLevel


app = FastAPI(title="Resume Scorer API")

# Add CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScoreRequest(BaseModel):
    """Request body for /score endpoint."""
    resume_text: str
    jd_text: str


@app.get("/")
def read_root():
    """Root endpoint returning API information."""
    return {"message": "Resume Scorer API", "version": "1.0.0"}


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/score")
async def score_resume(request: ScoreRequest) -> ScoreResult:
    """
    Score a resume against a job description.
    
    Pipeline:
    1. Parse resume and JD concurrently using asyncio.gather
    2. Compute skills score using embeddings
    3. Compute experience, education, keywords, responsibilities, and semantic scores
    4. Combine scores with uncertainty notes
    5. Generate explanation using LLM
    6. Build and return ScoreResult
    
    Args:
        request: ScoreRequest with resume_text and jd_text
        
    Returns:
        ScoreResult with overall score, category scores, explanation, suggestions
        
    Raises:
        HTTPException: If any step in the pipeline fails
    """
    # Clear any previous uncertainty notes
    clear_uncertainty_notes()
    
    try:
        # Step 1: Parse resume and JD concurrently
        resume, jd = await asyncio.gather(
            parse_resume(request.resume_text),
            parse_jd(request.jd_text)
        )
        
        # Step 2: Compute skills score (embedding-based)
        skills_score = compute_skills_score(resume, jd)
        
        # Step 3: Compute other scores
        experience_score = score_experience(resume, jd)
        education_score = score_education(resume, jd)
        keywords_score = score_keywords(resume, jd)
        responsibilities_score = score_responsibilities(resume, jd)
        semantic_score = compute_semantic_score(resume, jd)
        
        # Step 4: Combine scores
        category_scores = [skills_score, experience_score, education_score, keywords_score, responsibilities_score, semantic_score]
        uncertainty_notes = get_uncertainty_notes()
        combined = combine_scores(category_scores, uncertainty_notes, resume)
        
        # Step 5: Generate explanation
        explanation, suggestions = await generate_explanation(resume, jd, combined)
        
        # Step 6: Build ScoreResult
        # Convert confidence string to enum
        try:
            confidence_enum = ConfidenceLevel(combined["confidence"])
        except ValueError:
            confidence_enum = ConfidenceLevel.MEDIUM
        
        result = ScoreResult(
            overall_score=combined["overall_score"],
            category_scores=category_scores,
            explanation=explanation,
            suggestions=suggestions,
            uncertainty_flags=combined["uncertainty_flags"],
            confidence=confidence_enum
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
