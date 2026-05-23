"""
LLM-based explanation generator for score analysis.

This module uses LLM to generate human-readable explanations of the scoring results.
It provides detailed insights about why a resume received a particular score and
offers actionable suggestions for improvement.

The explainer generates:
- Score explanations in natural language
- Highlights strengths and weaknesses
- Provides specific improvement suggestions
- Explains the scoring methodology

This is the only LLM component used for output generation, not for scoring logic.
"""

import json
from typing import Tuple, List
from openai import AsyncOpenAI

from config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE
from models import ResumeData, JDData


# Initialize OpenAI client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def generate_explanation(
    resume: ResumeData,
    jd: JDData,
    score_data: dict
) -> Tuple[str, List[str]]:
    """
    Generate a detailed explanation of the scoring results using LLM.
    
    Args:
        resume: Parsed resume data
        jd: Parsed job description data
        score_data: Dictionary containing overall_score, category_scores, confidence, uncertainty_flags
        
    Returns:
        Tuple of (full_explanation_string, suggestions_list)
        
    The function:
    1. Extracts category scores from score_data
    2. Builds compact prompt with score breakdown
    3. Sends to LLM with structured output request
    4. Parses JSON response
    5. Returns explanation and suggestions
    6. Handles JSON parse errors gracefully with fallback
    """
    # Extract category scores
    category_scores = score_data.get("category_scores", [])
    overall_score = score_data.get("overall_score", 0)
    
    # Find specific category scores
    skills_score = None
    exp_score = None
    edu_score = None
    kw_score = None
    resp_score = None
    semantic_score = None
    matched = []
    missing = []
    partial = []
    
    for cat in category_scores:
        if cat.name == "skills_match":
            skills_score = int(cat.score * 100)
            matched = cat.matched
            missing = cat.missing
            partial = cat.partial
        elif cat.name == "experience":
            exp_score = int(cat.score * 100)
        elif cat.name == "education":
            edu_score = int(cat.score * 100)
        elif cat.name == "keywords":
            kw_score = int(cat.score * 100)
        elif cat.name == "responsibilities":
            resp_score = int(cat.score * 100)
        elif cat.name == "semantic":
            semantic_score = int(cat.score * 100)
    
    # Default values if not found
    skills_score = skills_score or 0
    exp_score = exp_score or 0
    edu_score = edu_score or 0
    kw_score = kw_score or 0
    resp_score = resp_score or 0
    semantic_score = semantic_score or 0
    
    # Build user prompt
    user_prompt = f"""Resume fit score: {overall_score}/100

Category breakdown:
- Skills match: {skills_score}/100 — matched: {matched}, missing: {missing}, partial: {partial}
- Experience: {exp_score}/100 — candidate has {resume.years_experience}y, role requires {jd.min_years_experience}y
- Education: {edu_score}/100 — candidate: {resume.education_level.value}, required: {jd.required_education.value}
- Keywords: {kw_score}/100
- Responsibilities: {resp_score}/100
- Semantic similarity: {semantic_score}/100

Write:
1. A 2-3 sentence summary of the overall fit.
2. Top 3-5 specific, actionable suggestions (each one sentence, starting with a verb).
3. One sentence about what's strongest in this match.

Output as JSON: {{summary: str, suggestions: list[str], strength: str}}"""
    
    system_prompt = "You are a career advisor helping a job-seeker understand their resume fit. Be direct and constructive. Never be harsh. Focus on what they can actually improve."
    
    try:
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=OPENAI_TEMPERATURE,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "explanation",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "summary": {
                                "type": "string"
                            },
                            "suggestions": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "strength": {
                                "type": "string"
                            }
                        },
                        "required": ["summary", "suggestions", "strength"],
                        "additionalProperties": False
                    }
                }
            }
        )
        
        # Parse JSON response
        result = json.loads(response.choices[0].message.content)
        
        # Build full explanation string
        full_explanation = f"{result['summary']}\n\nStrength: {result['strength']}\n\nSuggestions:\n" + "\n".join(f"- {s}" for s in result['suggestions'])
        
        return full_explanation, result['suggestions']
        
    except json.JSONDecodeError as e:
        # Fallback to raw text if JSON parsing fails
        try:
            # Try again without structured output
            response = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=OPENAI_TEMPERATURE
            )
            raw_text = response.choices[0].message.content
            return raw_text, ["Could not parse suggestions from response"]
        except Exception as fallback_error:
            # Double fallback - return generic message
            generic_explanation = f"Your resume scored {overall_score}/100 against this job description. "
            if overall_score >= 70:
                generic_explanation += "This is a strong match. "
            elif overall_score >= 50:
                generic_explanation += "This is a moderate match. "
            else:
                generic_explanation += "This is a weak match. "
            
            generic_explanation += f"Focus on improving the missing skills: {', '.join(missing[:5])}"
            
            return generic_explanation, [f"Consider adding experience with: {skill}" for skill in missing[:3]]
    
    except Exception as e:
        # Handle other errors gracefully
        generic_explanation = f"Your resume scored {overall_score}/100 against this job description. "
        if overall_score >= 70:
            generic_explanation += "This is a strong match. "
        elif overall_score >= 50:
            generic_explanation += "This is a moderate match. "
        else:
            generic_explanation += "This is a weak match. "
        
        generic_explanation += f"Focus on improving the missing skills: {', '.join(missing[:5])}"
        
        return generic_explanation, [f"Consider adding experience with: {skill}" for skill in missing[:3]]
