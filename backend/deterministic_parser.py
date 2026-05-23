"""
Deterministic resume and job description parser using spaCy and regex.

This module provides rule-based extraction of structured information from
unstructured resume and job description text. It uses spaCy for NLP tasks
and regex for pattern matching.

Features:
- Skills extraction using predefined skill lists and pattern matching
- Education extraction with degree level and field detection
- Years of experience calculation from date ranges
- Certification extraction
- Project extraction
- Responsibility extraction
- Tools/technologies extraction
- Handles noisy text with robust pattern matching
"""

import re
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import spacy

# Load spaCy model (will download on first run)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    raise


# Predefined skill lists for matching
SKILL_DATABASE = {
    "programming_languages": [
        "python", "java", "javascript", "c++", "c#", "go", "rust", "swift",
        "kotlin", "php", "ruby", "typescript", "scala", "r", "matlab", "perl"
    ],
    "web_frameworks": [
        "react", "angular", "vue", "django", "flask", "fastapi", "spring",
        "express", "node.js", "next.js", "nuxt.js", "rails", "laravel"
    ],
    "data_science": [
        "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
        "matplotlib", "seaborn", "jupyter", "spark", "hadoop", "tableau"
    ],
    "databases": [
        "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "sqlite",
        "oracle", "sql server", "cassandra", "dynamodb", "firebase"
    ],
    "cloud": [
        "aws", "azure", "gcp", "google cloud", "heroku", "digitalocean",
        "docker", "kubernetes", "terraform", "ansible", "jenkins"
    ],
    "tools": [
        "git", "github", "gitlab", "jira", "confluence", "slack", "vs code",
        "intellij", "eclipse", "linux", "unix", "bash", "shell scripting"
    ]
}

# Degree level mapping
DEGREE_LEVELS = {
    "phd": "phd",
    "doctorate": "phd",
    "doctor": "phd",
    "master": "master",
    "m.s.": "master",
    "m.sc": "master",
    "mba": "master",
    "bachelor": "bachelor",
    "b.s.": "bachelor",
    "b.sc": "bachelor",
    "b.a.": "bachelor",
    "ba": "bachelor",
    "bs": "bachelor",
    "associate": "high_school",
    "high school": "high_school",
    "diploma": "high_school"
}

# Certification patterns
CERTIFICATION_PATTERNS = [
    r"(?:certified|certification|certificate)[\s]*(?:in|of|for|as)?[\s]*([a-z\s]+?)(?:\n|,|\.|from)",
    r"([a-z\s]+?)(?:certified|certification|certificate)",
    r"(aws|azure|gcp|pmp|cisa|cism|cissp|comptia|google|microsoft)[\s]*(?:certified|certification)",
]

# Date patterns for experience calculation
DATE_PATTERNS = [
    r"(\d{4})\s*-\s*(\d{4})",  # 2020-2023
    r"(\d{4})\s*-\s*(present|current|now)",  # 2020-present
    r"([a-z]{3})\s*(\d{4})\s*-\s*([a-z]{3})\s*(\d{4})",  # Jan 2020 - Dec 2023
    r"([a-z]{3})\s*(\d{4})\s*-\s*(present|current)",  # Jan 2020 - present
]


def extract_skills(text: str) -> List[str]:
    """
    Extract skills from text using predefined skill database and pattern matching.
    
    Args:
        text: Resume or job description text
        
    Returns:
        List of detected skills (lowercase, deduplicated)
    """
    text_lower = text.lower()
    detected_skills = set()
    
    # Match against skill database
    for category, skills in SKILL_DATABASE.items():
        for skill in skills:
            # Check for whole word match
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                detected_skills.add(skill)
    
    # Extract skills from "Skills:" sections
    skills_section = re.search(
        r'(?:skills|technologies|tech stack|proficient in|expertise)[\s*:]+([^]*?)(?:\n\n|\n[A-Z]|\Z)',
        text_lower,
        re.IGNORECASE
    )
    if skills_section:
        section_text = skills_section.group(1)
        # Extract comma-separated skills
        comma_skills = [s.strip() for s in section_text.split(',')]
        detected_skills.update([s for s in comma_skills if len(s) > 2])
    
    # Extract bullet-point skills
    bullet_skills = re.findall(r'[-•]\s*([a-z][a-z\s]{2,30})', text_lower)
    detected_skills.update([s.strip() for s in bullet_skills if len(s) > 2])
    
    return sorted(list(detected_skills))


def extract_education(text: str) -> List[Dict[str, str]]:
    """
    Extract education information including degree, institution, and field.
    
    Args:
        text: Resume or job description text
        
    Returns:
        List of education dictionaries with degree, institution, field, and level
    """
    education_list = []
    
    # Find education section
    education_section = re.search(
        r'(?:education|academic|qualifications)[\s*:]+([^]*?)(?:\n\n|\n[A-Z][a-z]+[a-z\s]*:|\Z)',
        text,
        re.IGNORECASE
    )
    
    if not education_section:
        return education_list
    
    section_text = education_section.group(1)
    
    # Split by common delimiters
    entries = re.split(r'\n\s*-|\n\s*•|\n\s*\d+\.|\n\n', section_text)
    
    for entry in entries:
        entry = entry.strip()
        if len(entry) < 10:
            continue
        
        edu_info = {
            "degree": "",
            "institution": "",
            "field": "",
            "level": "unknown"
        }
        
        # Extract degree level
        for degree_name, level in DEGREE_LEVELS.items():
            if degree_name in entry.lower():
                edu_info["level"] = level
                edu_info["degree"] = degree_name
                break
        
        # Extract institution (usually capitalized words at start)
        institution_match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', entry)
        if institution_match:
            edu_info["institution"] = institution_match.group(1)
        
        # Extract field of study (often after "in" or "of")
        field_match = re.search(r'(?:in|of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', entry)
        if field_match:
            edu_info["field"] = field_match.group(1)
        
        if edu_info["institution"] or edu_info["degree"]:
            education_list.append(edu_info)
    
    return education_list


def calculate_years_of_experience(text: str) -> float:
    """
    Calculate total years of experience from date ranges in text.
    
    Args:
        text: Resume text containing work experience dates
        
    Returns:
        Total years of experience (float)
    """
    total_years = 0.0
    
    # Find experience section
    experience_section = re.search(
        r'(?:experience|work history|employment|professional)[\s*:]+([^]*?)(?:\n\n|\n[A-Z][a-z]+[a-z\s]*:|\Z)',
        text,
        re.IGNORECASE
    )
    
    if not experience_section:
        # Try to find date patterns in entire text
        search_text = text
    else:
        search_text = experience_section.group(1)
    
    # Extract date ranges
    for pattern in DATE_PATTERNS:
        matches = re.findall(pattern, search_text, re.IGNORECASE)
        for match in matches:
            try:
                if len(match) == 2:
                    start, end = match
                    if end.lower() in ["present", "current", "now"]:
                        end = str(datetime.now().year)
                    
                    start_year = int(re.search(r'\d{4}', start).group())
                    end_year = int(re.search(r'\d{4}', end).group())
                    
                    years = end_year - start_year
                    if years > 0:
                        total_years += years
            except (AttributeError, ValueError):
                continue
    
    # Also look for explicit "X years of experience" mentions
    explicit_years = re.search(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)', text, re.IGNORECASE)
    if explicit_years:
        try:
            years = float(explicit_years.group(1))
            if years > total_years:
                total_years = years
        except ValueError:
            pass
    
    return round(total_years, 1)


def extract_certifications(text: str) -> List[str]:
    """
    Extract certifications from text using pattern matching.
    
    Args:
        text: Resume or job description text
        
    Returns:
        List of certification names
    """
    certifications = []
    text_lower = text.lower()
    
    for pattern in CERTIFICATION_PATTERNS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            cert_name = match.strip()
            if len(cert_name) > 3 and len(cert_name) < 100:
                certifications.append(cert_name.title())
    
    # Remove duplicates while preserving order
    seen = set()
    unique_certs = []
    for cert in certifications:
        if cert.lower() not in seen:
            seen.add(cert.lower())
            unique_certs.append(cert)
    
    return unique_certs


def extract_projects(text: str) -> List[Dict[str, str]]:
    """
    Extract project information from text.
    
    Args:
        text: Resume or job description text
        
    Returns:
        List of project dictionaries with name, description, and technologies
    """
    projects = []
    
    # Find projects section
    projects_section = re.search(
        r'(?:projects|portfolio)[\s*:]+([^]*?)(?:\n\n|\n[A-Z][a-z]+[a-z\s]*:|\Z)',
        text,
        re.IGNORECASE
    )
    
    if not projects_section:
        return projects
    
    section_text = projects_section.group(1)
    
    # Split by common delimiters
    entries = re.split(r'\n\s*-|\n\s*•|\n\s*\d+\.|\n\n', section_text)
    
    for entry in entries:
        entry = entry.strip()
        if len(entry) < 15:
            continue
        
        project = {
            "name": "",
            "description": "",
            "technologies": []
        }
        
        # Extract project name (usually first sentence or before colon)
        name_match = re.search(r'^([^:]+):', entry)
        if name_match:
            project["name"] = name_match.group(1).strip()
            project["description"] = entry[len(name_match.group(0)):].strip()
        else:
            # Use first 50 chars as name
            words = entry.split()
            if len(words) > 3:
                project["name"] = " ".join(words[:5])
                project["description"] = " ".join(words[5:])
            else:
                project["description"] = entry
        
        # Extract technologies mentioned
        project["technologies"] = extract_skills(entry)
        
        if project["name"] or project["description"]:
            projects.append(project)
    
    return projects


def extract_responsibilities(text: str) -> List[str]:
    """
    Extract responsibilities from text, typically from experience sections.
    
    Args:
        text: Resume or job description text
        
    Returns:
        List of responsibility strings
    """
    responsibilities = []
    
    # Find experience section
    experience_section = re.search(
        r'(?:experience|work history|employment)[\s*:]+([^]*?)(?:\n\n|\n[A-Z][a-z]+[a-z\s]*:|\Z)',
        text,
        re.IGNORECASE
    )
    
    if not experience_section:
        return responsibilities
    
    section_text = experience_section.group(1)
    
    # Extract bullet points and numbered lists
    bullets = re.findall(r'[-•]\s*([^•\n]+)', section_text)
    numbered = re.findall(r'\d+\.\s*([^\n]+)', section_text)
    
    all_items = bullets + numbered
    
    for item in all_items:
        item = item.strip()
        if len(item) > 10 and len(item) < 200:
            responsibilities.append(item)
    
    return responsibilities


def extract_tools(text: str) -> List[str]:
    """
    Extract tools and technologies from text.
    
    Args:
        text: Resume or job description text
        
    Returns:
        List of tool/technology names
    """
    # This is similar to skills extraction but focuses on tools
    return extract_skills(text)


def parse_resume(text: str) -> Dict:
    """
    Parse resume text into structured JSON.
    
    Args:
        text: Raw resume text
        
    Returns:
        Dictionary containing structured resume information
    """
    # Clean text
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    
    result = {
        "skills": extract_skills(cleaned_text),
        "education": extract_education(cleaned_text),
        "years_of_experience": calculate_years_of_experience(cleaned_text),
        "certifications": extract_certifications(cleaned_text),
        "projects": extract_projects(cleaned_text),
        "responsibilities": extract_responsibilities(cleaned_text),
        "tools": extract_tools(cleaned_text),
        "raw_text_length": len(cleaned_text)
    }
    
    return result


def parse_job_description(text: str) -> Dict:
    """
    Parse job description text into structured JSON.
    
    Args:
        text: Raw job description text
        
    Returns:
        Dictionary containing structured job description information
    """
    # Clean text
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    
    result = {
        "required_skills": extract_skills(cleaned_text),
        "preferred_skills": [],  # Would need more sophisticated parsing to distinguish
        "education_requirements": extract_education(cleaned_text),
        "min_years_experience": calculate_years_of_experience(cleaned_text),
        "certifications": extract_certifications(cleaned_text),
        "responsibilities": extract_responsibilities(cleaned_text),
        "tools": extract_tools(cleaned_text),
        "raw_text_length": len(cleaned_text)
    }
    
    return result


# Sample input and output
if __name__ == "__main__":
    # Sample resume input
    sample_resume = """
    John Doe
    Software Engineer
    
    EXPERIENCE
    Senior Software Engineer - Tech Corp (2020 - Present)
    - Developed RESTful APIs using Python and Django
    - Led a team of 5 developers
    - Implemented CI/CD pipelines with Jenkins and Docker
    
    Software Developer - Startup Inc (2018 - 2020)
    - Built web applications using React and Node.js
    - Optimized database queries reducing response time by 40%
    
    EDUCATION
    Bachelor of Science in Computer Science - State University (2014 - 2018)
    
    SKILLS
    Python, JavaScript, React, Django, Flask, PostgreSQL, MongoDB, Docker, Kubernetes, AWS, Git
    
    CERTIFICATIONS
    AWS Certified Solutions Architect
    Certified Kubernetes Administrator
    
    PROJECTS
    E-commerce Platform: Built a full-stack e-commerce platform using Django and React
    Data Pipeline: Created a data processing pipeline using Apache Spark
    """
    
    # Sample job description input
    sample_jd = """
    Senior Backend Engineer
    
    REQUIREMENTS
    - 5+ years of experience in software development
    - Proficient in Python and Django
    - Experience with cloud services (AWS preferred)
    - Knowledge of containerization (Docker, Kubernetes)
    - Bachelor's degree in Computer Science or related field
    
    RESPONSIBILITIES
    - Design and implement scalable backend systems
    - Mentor junior developers
    - Collaborate with cross-functional teams
    
    PREFERRED
    - Experience with microservices architecture
    - Knowledge of GraphQL
    """
    
    # Parse resume
    print("=" * 50)
    print("RESUME PARSING RESULT")
    print("=" * 50)
    resume_result = parse_resume(sample_resume)
    print(json.dumps(resume_result, indent=2))
    
    # Parse job description
    print("\n" + "=" * 50)
    print("JOB DESCRIPTION PARSING RESULT")
    print("=" * 50)
    jd_result = parse_job_description(sample_jd)
    print(json.dumps(jd_result, indent=2))
