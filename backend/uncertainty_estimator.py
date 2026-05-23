"""
Uncertainty Estimation Module for Resume Scoring System.

This module identifies and quantifies uncertainty in resume scoring results.
It detects ambiguity, missing information, unclear experience levels, inferred skills,
and flags assumptions made during the scoring process.

Key Features:
- Ambiguity detection in resume text
- Missing information identification
- Experience level clarity assessment
- Inferred skill detection
- Assumption flagging
- Confidence score calculation
- Structured uncertainty explanations

Design Principles:
------------------
1. **Transparent**: Clearly identify what is uncertain and why
2. **Quantifiable**: Provide confidence scores with clear thresholds
3. **Actionable**: Suggest how to reduce uncertainty
4. **Comprehensive**: Cover all aspects of resume analysis
5. **Consistent**: Use standardized uncertainty categories
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class UncertaintyType(str, Enum):
    """Types of uncertainty that can be detected."""
    AMBIGUITY = "ambiguity"
    MISSING_INFO = "missing_info"
    UNCLEAR_EXPERIENCE = "unclear_experience"
    INFERRED_SKILL = "inferred_skill"
    ASSUMPTION = "assumption"
    LOW_CONFIDENCE = "low_confidence"


class ConfidenceLevel(str, Enum):
    """Confidence levels for overall assessment."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Uncertainty:
    """
    Individual uncertainty detected in the analysis.
    """
    type: UncertaintyType
    aspect: str  # What aspect is uncertain (e.g., "experience", "skills")
    description: str  # Human-readable description
    severity: str  # "low", "medium", "high"
    evidence: str  # Evidence from resume text (if applicable)
    suggestion: str  # How to reduce this uncertainty


@dataclass
class Assumption:
    """
    Assumption made during the analysis.
    """
    aspect: str
    assumption: str
    justification: str
    impact: str  # How this assumption affects the score


@dataclass
class UncertaintyEstimation:
    """
    Complete uncertainty estimation result.
    """
    confidence_score: float  # 0.0 to 1.0
    confidence_level: ConfidenceLevel
    uncertainties: List[Uncertainty]
    assumptions: List[Assumption]
    uncertainty_explanations: List[str]
    total_uncertainties: int
    high_severity_count: int


# ============================================================================
# UNCERTAINTY ESTIMATION ENGINE
# ============================================================================

class UncertaintyEstimator:
    """
    Engine for estimating uncertainty in resume scoring results.
    
    This class analyzes resume data and scoring results to identify
    sources of uncertainty and calculate an overall confidence score.
    """
    
    def __init__(self):
        """Initialize the uncertainty estimator."""
        # Patterns for detecting uncertainty indicators
        self.ambiguity_patterns = [
            r'\b(?:approximately|about|around|roughly|estimated|~)\s*\d+\b',
            r'\b(?:several|multiple|various|numerous)\s+(?:years?|months?)\b',
            r'\b\d+\+?\s*(?:years?|months?)\b',  # "5+ years"
            r'\b(?:recently|lately|recent)\b',
        ]
        
        self.missing_info_patterns = [
            r'\b(?:n/a|not specified|not mentioned|tbd|to be determined)\b',
            r'\b(?:will add|coming soon|pending)\b',
        ]
        
        # Cloud-related terms that might indicate inferred cloud experience
        self.cloud_indicators = [
            "docker", "container", "kubernetes", "deployment", "ci/cd",
            "devops", "infrastructure", "scaling"
        ]
        
        # Experience ambiguity indicators
        self.experience_ambiguity = [
            "intern", "junior", "senior", "lead", "manager",
            "contract", "freelance", "consultant"
        ]
    
    def estimate(
        self,
        parsed_resume: Dict,
        parsed_jd: Dict,
        scoring_breakdown: Dict,
        skill_similarities: Dict = None
    ) -> UncertaintyEstimation:
        """
        Estimate uncertainty in resume scoring results.
        
        Args:
            parsed_resume: Parsed resume data
            parsed_jd: Parsed job description
            scoring_breakdown: Scoring results
            skill_similarities: Optional skill similarity scores
            
        Returns:
            UncertaintyEstimation with complete uncertainty analysis
        """
        resume_text = parsed_resume.get("raw_text", "")
        
        # Detect all uncertainties
        uncertainties = []
        
        # 1. Detect ambiguity
        uncertainties.extend(self._detect_ambiguity(resume_text))
        
        # 2. Detect missing information
        uncertainties.extend(self._detect_missing_info(resume_text, parsed_resume))
        
        # 3. Detect unclear experience levels
        uncertainties.extend(self._detect_unclear_experience(resume_text, parsed_resume))
        
        # 4. Detect inferred skills
        uncertainties.extend(self._detect_inferred_skills(
            resume_text,
            parsed_resume,
            skill_similarities
        ))
        
        # 5. Detect assumptions from scoring
        uncertainties.extend(self._detect_assumptions(parsed_resume, scoring_breakdown))
        
        # Extract assumptions separately
        assumptions = self._extract_assumptions(parsed_resume, scoring_breakdown)
        
        # Generate uncertainty explanations
        uncertainty_explanations = self._generate_uncertainty_explanations(uncertainties)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(uncertainties)
        
        # Determine confidence level
        confidence_level = self._determine_confidence_level(confidence_score)
        
        # Count high severity uncertainties
        high_severity_count = sum(1 for u in uncertainties if u.severity == "high")
        
        return UncertaintyEstimation(
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            uncertainties=uncertainties,
            assumptions=assumptions,
            uncertainty_explanations=uncertainty_explanations,
            total_uncertainties=len(uncertainties),
            high_severity_count=high_severity_count
        )
    
    def _detect_ambiguity(self, resume_text: str) -> List[Uncertainty]:
        """
        Detect ambiguous statements in resume text.
        
        Args:
            resume_text: Raw resume text
            
        Returns:
            List of Uncertainty objects for ambiguity
        """
        uncertainties = []
        
        for pattern in self.ambiguity_patterns:
            matches = re.finditer(pattern, resume_text, re.IGNORECASE)
            for match in matches:
                context_start = max(0, match.start() - 30)
                context_end = min(len(resume_text), match.end() + 30)
                context = resume_text[context_start:context_end].strip()
                
                uncertainties.append(Uncertainty(
                    type=UncertaintyType.AMBIGUITY,
                    aspect="experience",
                    description=f"Ambiguous experience duration: '{match.group()}'",
                    severity="medium",
                    evidence=context,
                    suggestion="Use specific date ranges instead of approximate terms"
                ))
        
        return uncertainties
    
    def _detect_missing_info(self, resume_text: str, parsed_resume: Dict) -> List[Uncertainty]:
        """
        Detect missing information in resume.
        
        Args:
            resume_text: Raw resume text
            parsed_resume: Parsed resume data
            
        Returns:
            List of Uncertainty objects for missing information
        """
        uncertainties = []
        
        # Check for explicit missing info markers
        for pattern in self.missing_info_patterns:
            matches = re.finditer(pattern, resume_text, re.IGNORECASE)
            for match in matches:
                context_start = max(0, match.start() - 30)
                context_end = min(len(resume_text), match.end() + 30)
                context = resume_text[context_start:context_end].strip()
                
                uncertainties.append(Uncertainty(
                    type=UncertaintyType.MISSING_INFO,
                    aspect="general",
                    description=f"Missing information marker: '{match.group()}'",
                    severity="high",
                    evidence=context,
                    suggestion="Complete all sections with specific information"
                ))
        
        # Check for missing critical fields
        if parsed_resume.get("years_experience", 0) == 0:
            uncertainties.append(Uncertainty(
                type=UncertaintyType.MISSING_INFO,
                aspect="experience",
                description="Could not determine years of experience",
                severity="high",
                evidence="No date ranges found in experience section",
                suggestion="Add explicit date ranges for each position"
            ))
        
        if parsed_resume.get("education_level") == "unknown":
            uncertainties.append(Uncertainty(
                type=UncertaintyType.MISSING_INFO,
                aspect="education",
                description="Education level not detected",
                severity="medium",
                evidence="No education section or unclear degree",
                suggestion="Clearly state your highest degree and field of study"
            ))
        
        if not parsed_resume.get("skills", []):
            uncertainties.append(Uncertainty(
                type=UncertaintyType.MISSING_INFO,
                aspect="skills",
                description="No skills detected in resume",
                severity="high",
                evidence="Skills section missing or empty",
                suggestion="Add a dedicated skills section with technical skills"
            ))
        
        return uncertainties
    
    def _detect_unclear_experience(self, resume_text: str, parsed_resume: Dict) -> List[Uncertainty]:
        """
        Detect unclear experience levels.
        
        Args:
            resume_text: Raw resume text
            parsed_resume: Parsed resume data
            
        Returns:
            List of Uncertainty objects for unclear experience
        """
        uncertainties = []
        
        # Check for role ambiguity
        for term in self.experience_ambiguity:
            pattern = r'\b' + re.escape(term) + r'\b'
            matches = re.finditer(pattern, resume_text, re.IGNORECASE)
            for match in matches:
                context_start = max(0, match.start() - 40)
                context_end = min(len(resume_text), match.end() + 40)
                context = resume_text[context_start:context_end].strip()
                
                uncertainties.append(Uncertainty(
                    type=UncertaintyType.UNCLEAR_EXPERIENCE,
                    aspect="role_level",
                    description=f"Unclear role level: '{match.group()}'",
                    severity="low",
                    evidence=context,
                    suggestion="Specify responsibilities and team size to clarify role level"
                ))
        
        # Check for missing duration in experience entries
        exp_section = re.search(
            r'(?:experience|work history|employment)[\s*:]+([^]*?)(?:\n\n|\n[A-Z][a-z]+[a-z\s]*:|\Z)',
            resume_text,
            re.IGNORECASE
        )
        
        if exp_section:
            section_text = exp_section.group(1)
            # Look for entries without dates
            entries = re.split(r'\n\s*-|\n\s*•|\n\s*\d+\.|\n\n', section_text)
            for entry in entries:
                entry = entry.strip()
                if len(entry) > 20 and not re.search(r'\d{4}', entry):
                    uncertainties.append(Uncertainty(
                        type=UncertaintyType.UNCLEAR_EXPERIENCE,
                        aspect="duration",
                        description="Experience entry without date range",
                        severity="medium",
                        evidence=entry[:100],
                        suggestion="Add specific start and end dates for each position"
                    ))
        
        return uncertainties
    
    def _detect_inferred_skills(
        self,
        resume_text: str,
        parsed_resume: Dict,
        skill_similarities: Dict = None
    ) -> List[Uncertainty]:
        """
        Detect inferred skills (skills not explicitly mentioned but inferred).
        
        Args:
            resume_text: Raw resume text
            parsed_resume: Parsed resume data
            skill_similarities: Optional skill similarity scores
            
        Returns:
            List of Uncertainty objects for inferred skills
        """
        uncertainties = []
        
        # Check for inferred cloud experience
        has_cloud_indicators = any(
            indicator in resume_text.lower() 
            for indicator in self.cloud_indicators
        )
        
        resume_skills = [s.lower() for s in parsed_resume.get("skills", [])]
        has_explicit_cloud = any(
            cloud in resume_skills 
            for cloud in ["aws", "azure", "gcp", "cloud"]
        )
        
        if has_cloud_indicators and not has_explicit_cloud:
            uncertainties.append(Uncertainty(
                type=UncertaintyType.INFERRED_SKILL,
                aspect="cloud_experience",
                description="Cloud experience inferred from related technologies",
                severity="medium",
                evidence="Found: " + ", ".join([
                    ind for ind in self.cloud_indicators 
                    if ind in resume_text.lower()
                ]),
                suggestion="Explicitly state cloud platform experience (AWS, Azure, GCP)"
            ))
        
        # Check for partial skill matches from similarity scores
        if skill_similarities:
            for skill, similarity in skill_similarities.items():
                if 0.65 <= similarity < 0.85:  # Partial match range
                    uncertainties.append(Uncertainty(
                        type=UncertaintyType.INFERRED_SKILL,
                        aspect="skill_match",
                        description=f"Partial skill match inferred: {skill}",
                        severity="low",
                        evidence=f"Similarity score: {similarity:.2f}",
                        suggestion=f"Add explicit examples of {skill} experience"
                    ))
        
        return uncertainties
    
    def _detect_assumptions(
        self,
        parsed_resume: Dict,
        scoring_breakdown: Dict
    ) -> List[Uncertainty]:
        """
        Detect assumptions made during scoring.
        
        Args:
            parsed_resume: Parsed resume data
            scoring_breakdown: Scoring results
            
        Returns:
            List of Uncertainty objects for assumptions
        """
        uncertainties = []
        
        # Check if experience was estimated
        years_exp = parsed_resume.get("years_experience", 0)
        if years_exp > 0 and years_exp < 1:
            uncertainties.append(Uncertainty(
                type=UncertaintyType.ASSUMPTION,
                aspect="experience_estimation",
                description=f"Experience estimated as {years_exp:.1f} years (likely from job count)",
                severity="medium",
                evidence="No explicit date ranges found",
                suggestion="Provide explicit date ranges for accurate experience calculation"
            ))
        
        # Check for education assumptions
        if parsed_resume.get("education_level") == "unknown":
            uncertainties.append(Uncertainty(
                type=UncertaintyType.ASSUMPTION,
                aspect="education_assumption",
                description="Assumed no education requirement met",
                severity="medium",
                evidence="Education not detected",
                suggestion="Clearly state education level"
            ))
        
        # Check for semantic similarity assumptions
        semantic_score = scoring_breakdown.get("semantic_similarity", {}).get("score", 0)
        if semantic_score < 0.5:
            uncertainties.append(Uncertainty(
                type=UncertaintyType.ASSUMPTION,
                aspect="semantic_interpretation",
                description="Low semantic similarity may indicate terminology mismatch",
                severity="low",
                evidence=f"Semantic score: {semantic_score:.2f}",
                suggestion="Use terminology from job description in resume"
            ))
        
        return uncertainties
    
    def _extract_assumptions(
        self,
        parsed_resume: Dict,
        scoring_breakdown: Dict
    ) -> List[Assumption]:
        """
        Extract explicit assumptions made during analysis.
        
        Args:
            parsed_resume: Parsed resume data
            scoring_breakdown: Scoring results
            
        Returns:
            List of Assumption objects
        """
        assumptions = []
        
        # Experience assumption
        years_exp = parsed_resume.get("years_experience", 0)
        if years_exp > 0:
            assumptions.append(Assumption(
                aspect="experience",
                assumption=f"Years of experience calculated as {years_exp:.1f}",
                justification="Based on date ranges or job count estimation",
                impact="Affects experience score (15% weight)"
            ))
        
        # Skills assumption
        skills = parsed_resume.get("skills", [])
        if skills:
            assumptions.append(Assumption(
                aspect="skills",
                assumption=f"Detected {len(skills)} skills from resume",
                justification="Using keyword and pattern matching",
                impact="Affects skills score (45% weight)"
            ))
        
        # Education assumption
        edu_level = parsed_resume.get("education_level", "unknown")
        assumptions.append(Assumption(
            aspect="education",
            assumption=f"Education level identified as {edu_level}",
            justification="Based on degree keywords in education section",
            impact="Affects education score (10% weight)"
        ))
        
        # Semantic assumption
        semantic_score = scoring_breakdown.get("semantic_similarity", {}).get("score", 0)
        assumptions.append(Assumption(
            aspect="semantic",
            assumption=f"Semantic similarity score of {semantic_score:.2f}",
            justification="Based on sentence-transformer embeddings",
            impact="Affects semantic score (10% weight)"
        ))
        
        return assumptions
    
    def _generate_uncertainty_explanations(self, uncertainties: List[Uncertainty]) -> List[str]:
        """
        Generate human-readable explanations for uncertainties.
        
        Args:
            uncertainties: List of detected uncertainties
            
        Returns:
            List of explanation strings
        """
        explanations = []
        
        # Group by type
        by_type = {}
        for u in uncertainties:
            if u.type not in by_type:
                by_type[u.type] = []
            by_type[u.type].append(u)
        
        # Generate explanations for each type
        for uncertainty_type, items in by_type.items():
            if uncertainty_type == UncertaintyType.AMBIGUITY:
                explanations.append(
                    f"Found {len(items)} ambiguous statement(s) that may affect accuracy. "
                    "Use specific dates and quantities instead of approximate terms."
                )
            elif uncertainty_type == UncertaintyType.MISSING_INFO:
                explanations.append(
                    f"Detected {len(items)} missing information item(s). "
                    "Complete all sections with specific details to improve scoring accuracy."
                )
            elif uncertainty_type == UncertaintyType.UNCLEAR_EXPERIENCE:
                explanations.append(
                    f"Identified {len(items)} unclear experience indicator(s). "
                    "Clarify role levels and provide date ranges for each position."
                )
            elif uncertainty_type == UncertaintyType.INFERRED_SKILL:
                explanations.append(
                    f"Inferred {len(items)} skill(s) from related technologies. "
                    "Explicitly state these skills to strengthen your profile."
                )
            elif uncertainty_type == UncertaintyType.ASSUMPTION:
                explanations.append(
                    f"Made {len(items)} assumption(s) during analysis. "
                    "Provide explicit information to reduce reliance on assumptions."
                )
        
        if not explanations:
            explanations.append("No significant uncertainties detected. Resume appears complete and clear.")
        
        return explanations
    
    def _calculate_confidence_score(self, uncertainties: List[Uncertainty]) -> float:
        """
        Calculate overall confidence score based on uncertainties.
        
        Formula:
        confidence = 1.0 - (weighted_uncertainty_penalty)
        
        Weights:
        - High severity: 0.15
        - Medium severity: 0.08
        - Low severity: 0.03
        
        Args:
            uncertainties: List of detected uncertainties
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not uncertainties:
            return 1.0
        
        penalty = 0.0
        
        for u in uncertainties:
            if u.severity == "high":
                penalty += 0.15
            elif u.severity == "medium":
                penalty += 0.08
            elif u.severity == "low":
                penalty += 0.03
        
        # Cap penalty at 0.8 (minimum confidence of 0.2)
        penalty = min(penalty, 0.8)
        
        confidence = max(0.2, 1.0 - penalty)
        
        return round(confidence, 2)
    
    def _determine_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """
        Determine confidence level from confidence score.
        
        Args:
            confidence_score: Confidence score (0.0 to 1.0)
            
        Returns:
            ConfidenceLevel enum value
        """
        if confidence_score >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def to_json(self, result: UncertaintyEstimation) -> str:
        """
        Convert uncertainty estimation result to JSON string.
        
        Args:
            result: UncertaintyEstimation object
            
        Returns:
            JSON string
        """
        # Convert dataclass to dict
        result_dict = asdict(result)
        
        # Convert enums to strings
        result_dict["confidence_level"] = result_dict["confidence_level"].value
        
        # Convert uncertainty types to strings
        for u in result_dict["uncertainties"]:
            u["type"] = u["type"].value
        
        return json.dumps(result_dict, indent=2)


# ============================================================================
# SAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("UNCERTAINTY ESTIMATION MODULE - SAMPLE USAGE")
    print("=" * 70)
    
    # Initialize estimator
    estimator = UncertaintyEstimator()
    
    # Sample parsed resume with uncertainties
    parsed_resume = {
        "raw_text": """
        John Doe - Software Engineer
        
        EXPERIENCE
        Senior Software Engineer at Tech Corp (approximately 5 years)
        - Developed RESTful APIs using Python
        - Led several developers
        - Implemented CI/CD with Docker
        
        Software Developer at Startup Inc (recently)
        - Built web applications
        - Collaborated with teams
        
        EDUCATION
        TBD - Will add details later
        
        SKILLS
        Python, Docker, Kubernetes, CI/CD
        """,
        "years_experience": 0.5,  # Estimated/inferred
        "education_level": "unknown",
        "skills": ["python", "docker", "kubernetes", "ci/cd"]
    }
    
    # Sample parsed JD
    parsed_jd = {
        "required_skills": ["python", "aws", "kubernetes", "docker"]
    }
    
    # Sample scoring breakdown
    scoring_breakdown = {
        "overall_score": 65,
        "required_skills": {"score": 0.7},
        "experience": {"score": 0.5},
        "education": {"score": 0.0},
        "semantic_similarity": {"score": 0.45}
    }
    
    # Sample skill similarities (partial matches)
    skill_similarities = {
        "aws": 0.72  # Partial match (has cloud experience but not AWS specifically)
    }
    
    # Estimate uncertainty
    print("\nEstimating uncertainty...\n")
    result = estimator.estimate(
        parsed_resume=parsed_resume,
        parsed_jd=parsed_jd,
        scoring_breakdown=scoring_breakdown,
        skill_similarities=skill_similarities
    )
    
    # Print results
    print("-" * 70)
    print("CONFIDENCE SCORE")
    print("-" * 70)
    print(f"Score: {result.confidence_score:.2f}/1.0")
    print(f"Level: {result.confidence_level.value.upper()}")
    print(f"Total Uncertainties: {result.total_uncertainties}")
    print(f"High Severity: {result.high_severity_count}")
    
    print("\n" + "-" * 70)
    print("UNCERTAINTIES")
    print("-" * 70)
    for i, uncertainty in enumerate(result.uncertainties, 1):
        print(f"\n{i}. {uncertainty.type.value.upper()} - {uncertainty.aspect}")
        print(f"   Description: {uncertainty.description}")
        print(f"   Severity: {uncertainty.severity}")
        if uncertainty.evidence:
            print(f"   Evidence: \"{uncertainty.evidence}\"")
        print(f"   Suggestion: {uncertainty.suggestion}")
    
    print("\n" + "-" * 70)
    print("ASSUMPTIONS")
    print("-" * 70)
    for i, assumption in enumerate(result.assumptions, 1):
        print(f"\n{i}. {assumption.aspect.upper()}")
        print(f"   Assumption: {assumption.assumption}")
        print(f"   Justification: {assumption.justification}")
        print(f"   Impact: {assumption.impact}")
    
    print("\n" + "-" * 70)
    print("UNCERTAINTY EXPLANATIONS")
    print("-" * 70)
    for i, explanation in enumerate(result.uncertainty_explanations, 1):
        print(f"{i}. {explanation}")
    
    # JSON output
    print("\n" + "=" * 70)
    print("JSON OUTPUT")
    print("=" * 70)
    print(estimator.to_json(result))
