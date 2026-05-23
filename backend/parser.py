"""
LLM-based resume and job description parser.

This module uses OpenAI's API to parse unstructured text from resumes and job
descriptions into structured data. It extracts:
- Skills and technologies
- Years of experience
- Education details
- Work responsibilities
- Company names and roles
- Other relevant information

The parser uses structured output with json_schema to guarantee valid JSON
matching the Pydantic schemas defined in models.py.
"""

import asyncio
from typing import List
from openai import AsyncOpenAI
from pydantic import ValidationError

from config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE
from models import ResumeData, JDData, EducationLevel


# Module-level list to store uncertainty notes for later collection
uncertainty_notes: List[str] = []


# Initialize OpenAI client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def parse_resume(text: str) -> ResumeData:
    """
    Parse resume text into structured data using LLM.
    
    Args:
        text: Raw text extracted from resume file
        
    Returns:
        ResumeData object containing parsed resume information
        
    The function will:
    1. Send resume text to LLM with parsing prompt
    2. Extract structured information using structured output
    3. Validate against ResumeData schema
    4. Store uncertainty notes in module-level list
    5. Return validated ResumeData model
    """
    system_prompt = """You are a resume parser. Extract structured data from the resume text. For years_experience, infer total professional years from dates. If dates are missing, estimate from job count and titles. For education_level use: high_school, bachelor, master, phd, or unknown. Output ONLY valid JSON with keys: candidate_name, skills, job_titles, years_experience, education_level. Add an uncertainty_notes list of strings noting any assumptions you made."""
    
    try:
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=OPENAI_TEMPERATURE,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "resume_data",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "candidate_name": {"type": "string"},
                            "skills": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "job_titles": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "years_experience": {"type": "number"},
                            "education_level": {
                                "type": "string",
                                "enum": ["high_school", "bachelor", "master", "phd", "unknown"]
                            },
                            "uncertainty_notes": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["candidate_name", "skills", "job_titles", "years_experience", "education_level", "uncertainty_notes"],
                        "additionalProperties": False
                    }
                }
            }
        )
        
        # Parse JSON response
        import json
        parsed_data = json.loads(response.choices[0].message.content)
        
        # Store uncertainty notes
        notes = parsed_data.pop("uncertainty_notes", [])
        uncertainty_notes.extend(notes)
        
        # Convert education_level string to enum
        education_level_str = parsed_data.get("education_level", "unknown")
        try:
            parsed_data["education_level"] = EducationLevel(education_level_str)
        except ValueError:
            parsed_data["education_level"] = EducationLevel.UNKNOWN
        
        # Add raw_text for LLM explainer
        parsed_data["raw_text"] = text
        
        # Validate and return ResumeData
        return ResumeData(**parsed_data)
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}")
    except ValidationError as e:
        raise ValueError(f"LLM response failed Pydantic validation: {e}")
    except Exception as e:
        raise ValueError(f"Error parsing resume: {e}")


async def parse_jd(text: str) -> JDData:
    """
    Parse job description text into structured data using LLM.
    
    Args:
        text: Raw text from job description
        
    Returns:
        JDData object containing parsed job description information
        
    The function will:
    1. Send JD text to LLM with parsing prompt
    2. Extract requirements, responsibilities, and preferences using structured output
    3. Validate against JDData schema
    4. Store uncertainty notes in module-level list
    5. Return validated JDData model
    """
    system_prompt = """You are a job description parser. Extract structured data. Distinguish required vs preferred skills carefully — treat 'must have' / 'required' as required, 'nice to have' / 'preferred' as preferred. Output ONLY valid JSON with keys: job_title, required_skills, preferred_skills, min_years_experience, required_education. Add uncertainty_notes."""
    
    try:
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=OPENAI_TEMPERATURE,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "jd_data",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "job_title": {"type": "string"},
                            "required_skills": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "preferred_skills": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "min_years_experience": {"type": "number"},
                            "required_education": {
                                "type": "string",
                                "enum": ["high_school", "bachelor", "master", "phd", "unknown"]
                            },
                            "uncertainty_notes": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["job_title", "required_skills", "preferred_skills", "min_years_experience", "required_education", "uncertainty_notes"],
                        "additionalProperties": False
                    }
                }
            }
        )
        
        # Parse JSON response
        import json
        parsed_data = json.loads(response.choices[0].message.content)
        
        # Store uncertainty notes
        notes = parsed_data.pop("uncertainty_notes", [])
        uncertainty_notes.extend(notes)
        
        # Convert required_education string to enum
        education_level_str = parsed_data.get("required_education", "unknown")
        try:
            parsed_data["required_education"] = EducationLevel(education_level_str)
        except ValueError:
            parsed_data["required_education"] = EducationLevel.UNKNOWN
        
        # Add raw_text for LLM explainer
        parsed_data["raw_text"] = text
        
        # Validate and return JDData
        return JDData(**parsed_data)
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}")
    except ValidationError as e:
        raise ValueError(f"LLM response failed Pydantic validation: {e}")
    except Exception as e:
        raise ValueError(f"Error parsing job description: {e}")


def get_uncertainty_notes() -> List[str]:
    """
    Get all uncertainty notes collected during parsing.
    
    Returns:
        List of uncertainty notes from resume and JD parsing
    """
    return uncertainty_notes.copy()


def clear_uncertainty_notes() -> None:
    """
    Clear all uncertainty notes from the module-level list.
    """
    uncertainty_notes.clear()
