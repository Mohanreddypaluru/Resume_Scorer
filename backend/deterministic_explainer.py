"""
Deterministic Resume Scoring Explanation Module.

This module provides structured, evidence-based explanations for resume scoring results.
Unlike LLM-based explainers, this module uses deterministic rules to extract evidence
from resume text and provide specific, actionable feedback.

Key Features:
- Evidence extraction from resume text
- Structured explanation generation
- Avoids generic feedback
- Cites specific resume text as evidence
- Explains assumptions made during scoring
- Identifies where scoring may be inaccurate
- Returns JSON format

Design Principles:
------------------
1. **Evidence-Based**: Every claim must be backed by evidence from the resume text
2. **Specific**: Avoid generic statements like "improve your resume"
3. **Transparent**: Explain assumptions and potential scoring errors
4. **Actionable**: Provide concrete, implementable suggestions
5. **Structured**: Consistent JSON output format
"""

import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Evidence:
    """
    Evidence extracted from resume text to support claims.
    """
    claim: str
    evidence_text: str
    context: str
    confidence: str  # "high", "medium", "low"


@dataclass
class Strength:
    """
    Identified strength in the resume.
    """
    category: str
    description: str
    evidence: List[Evidence]
    impact: str  # "high", "medium", "low"


@dataclass
class Gap:
    """
    Identified gap in the resume relative to job requirements.
    """
    category: str
    description: str
    severity: str  # "critical", "moderate", "minor"
    suggestions: List[str]


@dataclass
class PartialMatch:
    """
    Partial match between resume and job requirements.
    """
    skill: str
    resume_evidence: str
    jd_requirement: str
    similarity_score: float
    improvement_suggestion: str


@dataclass
class Uncertainty:
    """
    Uncertainty or potential scoring inaccuracy.
    """
    aspect: str
    reason: str
    potential_impact: str
    mitigation: str


@dataclass
class ExplanationResult:
    """
    Complete explanation result in structured format.
    """
    strengths: List[Strength]
    gaps: List[Gap]
    partial_matches: List[PartialMatch]
    actionable_suggestions: List[str]
    uncertainties: List[Uncertainty]
    overall_assessment: str
    confidence_level: str


# ============================================================================
# EXPLANATION ENGINE
# ============================================================================

class DeterministicExplainer:
    """
    Deterministic explanation engine for resume scoring results.
    
    This class provides structured explanations based on evidence extracted
    from the resume text and scoring results.
    """
    
    def __init__(self):
        """Initialize the explanation engine."""
        self.skill_keywords = {
            "programming": ["develop", "code", "program", "build", "implement", "create"],
            "leadership": ["lead", "manage", "supervise", "mentor", "guide", "direct"],
            "collaboration": ["collaborate", "team", "partner", "work with", "coordinate"],
            "optimization": ["optimize", "improve", "enhance", "reduce", "increase", "accelerate"],
            "architecture": ["design", "architecture", "system", "structure", "framework"]
        }
    
    def explain(
        self,
        parsed_resume: Dict,
        parsed_jd: Dict,
        scoring_breakdown: Dict,
        matched_skills: List[str],
        missing_skills: List[str],
        semantic_scores: Dict = None
    ) -> ExplanationResult:
        """
        Generate comprehensive explanation for resume scoring results.
        
        Args:
            parsed_resume: Parsed resume data with raw_text
            parsed_jd: Parsed job description data
            scoring_breakdown: Category scores and overall score
            matched_skills: List of matched skills
            missing_skills: List of missing skills
            semantic_scores: Optional semantic similarity scores
            
        Returns:
            ExplanationResult with structured explanation
        """
        resume_text = parsed_resume.get("raw_text", "")
        
        # Extract strengths
        strengths = self._extract_strengths(
            resume_text,
            matched_skills,
            scoring_breakdown
        )
        
        # Extract gaps
        gaps = self._extract_gaps(
            missing_skills,
            scoring_breakdown,
            parsed_jd
        )
        
        # Extract partial matches
        partial_matches = self._extract_partial_matches(
            resume_text,
            scoring_breakdown
        )
        
        # Generate actionable suggestions
        actionable_suggestions = self._generate_suggestions(
            gaps,
            partial_matches,
            scoring_breakdown
        )
        
        # Identify uncertainties
        uncertainties = self._identify_uncertainties(
            parsed_resume,
            scoring_breakdown,
            semantic_scores
        )
        
        # Generate overall assessment
        overall_assessment = self._generate_overall_assessment(
            scoring_breakdown,
            strengths,
            gaps
        )
        
        # Determine confidence level
        confidence_level = self._determine_confidence(
            parsed_resume,
            uncertainties
        )
        
        return ExplanationResult(
            strengths=strengths,
            gaps=gaps,
            partial_matches=partial_matches,
            actionable_suggestions=actionable_suggestions,
            uncertainties=uncertainties,
            overall_assessment=overall_assessment,
            confidence_level=confidence_level
        )
    
    def _extract_strengths(
        self,
        resume_text: str,
        matched_skills: List[str],
        scoring_breakdown: Dict
    ) -> List[Strength]:
        """
        Extract strengths from resume with evidence.
        
        Args:
            resume_text: Raw resume text
            matched_skills: List of matched skills
            scoring_breakdown: Scoring results
            
        Returns:
            List of Strength objects with evidence
        """
        strengths = []
        
        # Skill-based strengths
        if matched_skills:
            for skill in matched_skills[:5]:  # Top 5 matched skills
                evidence = self._find_skill_evidence(resume_text, skill)
                if evidence:
                    strengths.append(Strength(
                        category="skills",
                        description=f"Demonstrated proficiency in {skill}",
                        evidence=[evidence],
                        impact="high"
                    ))
        
        # Experience strength
        exp_score = scoring_breakdown.get("experience", {})
        if exp_score.get("score", 0) >= 0.8:
            years = exp_score.get("matched", [""])[0] if exp_score.get("matched") else ""
            evidence = self._find_experience_evidence(resume_text)
            strengths.append(Strength(
                category="experience",
                description=f"Strong experience level ({years})",
                evidence=[evidence] if evidence else [],
                impact="high"
            ))
        
        # Responsibility strength
        resp_score = scoring_breakdown.get("responsibilities", {})
        if resp_score.get("score", 0) >= 0.7:
            evidence = self._find_responsibility_evidence(resume_text)
            strengths.append(Strength(
                category="responsibilities",
                description="Strong alignment with job responsibilities",
                evidence=[evidence] if evidence else [],
                impact="medium"
            ))
        
        # Education strength
        edu_score = scoring_breakdown.get("education", {})
        if edu_score.get("score", 0) >= 0.8:
            evidence = self._find_education_evidence(resume_text)
            strengths.append(Strength(
                category="education",
                description="Meets or exceeds education requirements",
                evidence=[evidence] if evidence else [],
                impact="medium"
            ))
        
        # Semantic strength
        semantic_score = scoring_breakdown.get("semantic_similarity", {})
        if semantic_score.get("score", 0) >= 0.7:
            strengths.append(Strength(
                category="semantic",
                description="Strong overall semantic fit with job description",
                evidence=[],
                impact="low"
            ))
        
        return strengths
    
    def _extract_gaps(
        self,
        missing_skills: List[str],
        scoring_breakdown: Dict,
        parsed_jd: Dict
    ) -> List[Gap]:
        """
        Extract gaps between resume and job requirements.
        
        Args:
            missing_skills: List of missing skills
            scoring_breakdown: Scoring results
            parsed_jd: Parsed job description
            
        Returns:
            List of Gap objects
        """
        gaps = []
        
        # Skill gaps
        if missing_skills:
            # Categorize by severity
            critical_skills = parsed_jd.get("required_skills", [])[:3]
            critical_missing = [s for s in missing_skills if s in critical_skills]
            
            for skill in missing_skills[:5]:
                severity = "critical" if skill in critical_missing else "moderate"
                suggestions = self._generate_skill_suggestions(skill)
                gaps.append(Gap(
                    category="skills",
                    description=f"Missing required skill: {skill}",
                    severity=severity,
                    suggestions=suggestions
                ))
        
        # Experience gap
        exp_score = scoring_breakdown.get("experience", {})
        if exp_score.get("score", 0) < 0.6:
            gaps.append(Gap(
                category="experience",
                description="Insufficient years of experience",
                severity="moderate",
                suggestions=[
                    "Highlight relevant projects to demonstrate capability",
                    "Emphasize quality of experience over quantity",
                    "Consider certifications to bridge experience gap"
                ]
            ))
        
        # Education gap
        edu_score = scoring_breakdown.get("education", {})
        if edu_score.get("score", 0) < 0.5:
            gaps.append(Gap(
                category="education",
                description="Education level below requirement",
                severity="minor",
                suggestions=[
                    "Highlight equivalent experience",
                    "Consider relevant certifications",
                    "Emphasize practical skills and projects"
                ]
            ))
        
        return gaps
    
    def _extract_partial_matches(
        self,
        resume_text: str,
        scoring_breakdown: Dict
    ) -> List[PartialMatch]:
        """
        Extract partial matches between resume and job requirements.
        
        Args:
            resume_text: Raw resume text
            scoring_breakdown: Scoring results
            
        Returns:
            List of PartialMatch objects
        """
        partial_matches = []
        
        # Get partial skills from scoring breakdown
        skills_score = scoring_breakdown.get("required_skills", {})
        partial_skills = skills_score.get("partial", [])
        
        for skill in partial_skills:
            # Find evidence in resume
            evidence = self._find_skill_evidence(resume_text, skill)
            resume_evidence = evidence.evidence_text if evidence else "No direct evidence found"
            
            partial_matches.append(PartialMatch(
                skill=skill,
                resume_evidence=resume_evidence,
                jd_requirement=f"Experience with {skill}",
                similarity_score=0.75,  # Default for partial matches
                improvement_suggestion=f"Add specific examples of {skill} experience to strengthen match"
            ))
        
        return partial_matches
    
    def _generate_suggestions(
        self,
        gaps: List[Gap],
        partial_matches: List[PartialMatch],
        scoring_breakdown: Dict
    ) -> List[str]:
        """
        Generate actionable suggestions based on gaps and partial matches.
        
        Args:
            gaps: List of identified gaps
            partial_matches: List of partial matches
            scoring_breakdown: Scoring results
            
        Returns:
            List of actionable suggestion strings
        """
        suggestions = []
        
        # Suggestions from gaps
        for gap in gaps:
            if gap.severity in ["critical", "moderate"]:
                suggestions.extend(gap.suggestions[:2])
        
        # Suggestions from partial matches
        for partial in partial_matches[:3]:
            suggestions.append(partial.improvement_suggestion)
        
        # Category-specific suggestions
        skills_score = scoring_breakdown.get("required_skills", {})
        if skills_score.get("score", 0) < 0.7:
            suggestions.append("Strengthen your skills section by including specific projects, tools, and technologies you've worked with. Quantify your experience where possible.")
        
        resp_score = scoring_breakdown.get("responsibilities", {})
        if resp_score.get("score", 0) < 0.6:
            suggestions.append("Enhance your responsibilities section by using action verbs and including measurable achievements. Focus on outcomes and impact rather than just listing duties.")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in seen:
                seen.add(suggestion)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions[:10]  # Limit to top 10
    
    def _identify_uncertainties(
        self,
        parsed_resume: Dict,
        scoring_breakdown: Dict,
        semantic_scores: Dict = None
    ) -> List[Uncertainty]:
        """
        Identify uncertainties and potential scoring inaccuracies.
        
        Args:
            parsed_resume: Parsed resume data
            scoring_breakdown: Scoring results
            semantic_scores: Optional semantic similarity scores
            
        Returns:
            List of Uncertainty objects
        """
        uncertainties = []
        
        # Resume text length uncertainty
        resume_text = parsed_resume.get("raw_text", "")
        if len(resume_text) < 500:
            uncertainties.append(Uncertainty(
                aspect="resume_length",
                reason="Resume text is very short, may not contain full experience",
                potential_impact="Skills and experience may be under-detected",
                mitigation="Provide more detailed descriptions of experience and projects"
            ))
        
        # Experience inference uncertainty
        if parsed_resume.get("years_experience", 0) == 0:
            uncertainties.append(Uncertainty(
                aspect="experience_detection",
                reason="Could not determine years of experience from resume",
                potential_impact="Experience score may be inaccurate",
                mitigation="Add explicit date ranges for each position"
            ))
        
        # Education uncertainty
        if parsed_resume.get("education_level") == "unknown":
            uncertainties.append(Uncertainty(
                aspect="education_detection",
                reason="Education level not detected in resume",
                potential_impact="Education score may be underestimated",
                mitigation="Clearly state education level and field of study"
            ))
        
        # Semantic score uncertainty
        if semantic_scores:
            if semantic_scores.get("overall_semantic_score", 0) < 0.5:
                uncertainties.append(Uncertainty(
                    aspect="semantic_similarity",
                    reason="Low semantic similarity may indicate different terminology",
                    potential_impact="May miss relevant experience due to wording differences",
                    mitigation="Use terminology from job description in your resume"
                ))
        
        # Skill detection uncertainty
        skills_score = scoring_breakdown.get("required_skills", {})
        if skills_score.get("score", 0) < 0.5 and skills_score.get("missing"):
            uncertainties.append(Uncertainty(
                aspect="skill_detection",
                reason="Many required skills not detected in resume",
                potential_impact="Skills score may be lower than actual capability",
                mitigation="Ensure skills are listed explicitly in skills section"
            ))
        
        return uncertainties
    
    def _generate_overall_assessment(
        self,
        scoring_breakdown: Dict,
        strengths: List[Strength],
        gaps: List[Gap]
    ) -> str:
        """
        Generate overall assessment of resume fit.
        
        Args:
            scoring_breakdown: Scoring results
            strengths: List of identified strengths
            gaps: List of identified gaps
            
        Returns:
            Overall assessment string
        """
        overall_score = scoring_breakdown.get("overall_score", 0)
        
        if overall_score >= 80:
            assessment = f"Excellent match (score: {overall_score}/100). "
            assessment += f"Strong alignment with {len(strengths)} key strengths. "
            if gaps:
                assessment += f"Minor gaps in {len(gaps)} areas could be addressed to strengthen application."
            else:
                assessment += "No significant gaps detected."
        elif overall_score >= 65:
            assessment = f"Strong match (score: {overall_score}/100). "
            assessment += f"Good alignment with {len(strengths)} strengths. "
            critical_gaps = [g for g in gaps if g.severity == "critical"]
            if critical_gaps:
                assessment += f"Address {len(critical_gaps)} critical gaps to improve competitiveness."
            else:
                assessment += "Focus on strengthening partial matches for better fit."
        elif overall_score >= 50:
            assessment = f"Moderate match (score: {overall_score}/100). "
            assessment += f"Some alignment with {len(strengths)} strengths. "
            assessment += f"Significant gaps in {len(gaps)} areas require attention. "
            assessment += "Consider gaining experience with missing skills or highlighting transferable skills."
        else:
            assessment = f"Weak match (score: {overall_score}/100). "
            assessment += f"Limited alignment with job requirements. "
            assessment += f"Major gaps in {len(gaps)} areas. "
            assessment += "Consider gaining relevant experience or targeting roles that better match current skill set."
        
        return assessment
    
    def _determine_confidence(
        self,
        parsed_resume: Dict,
        uncertainties: List[Uncertainty]
    ) -> str:
        """
        Determine confidence level in the scoring results.
        
        Args:
            parsed_resume: Parsed resume data
            uncertainties: List of identified uncertainties
            
        Returns:
            Confidence level: "high", "medium", or "low"
        """
        # Check for high-impact uncertainties
        high_impact_uncertainties = [u for u in uncertainties if "under" in u.potential_impact.lower()]
        
        if len(high_impact_uncertainties) >= 2:
            return "low"
        elif len(uncertainties) >= 1:
            return "medium"
        else:
            return "high"
    
    def _find_skill_evidence(self, resume_text: str, skill: str) -> Optional[Evidence]:
        """
        Find evidence of a skill in resume text.
        
        Args:
            resume_text: Raw resume text
            skill: Skill to search for
            
        Returns:
            Evidence object if found, None otherwise
        """
        # Search for skill mentions
        pattern = r'\b' + re.escape(skill) + r'\b'
        matches = list(re.finditer(pattern, resume_text, re.IGNORECASE))
        
        if matches:
            # Get context around the match
            match = matches[0]
            start = max(0, match.start() - 50)
            end = min(len(resume_text), match.end() + 50)
            context = resume_text[start:end].strip()
            
            return Evidence(
                claim=f"Proficiency in {skill}",
                evidence_text=match.group(),
                context=context,
                confidence="high"
            )
        
        return None
    
    def _find_experience_evidence(self, resume_text: str) -> Optional[Evidence]:
        """
        Find evidence of work experience in resume text.
        
        Args:
            resume_text: Raw resume text
            
        Returns:
            Evidence object if found, None otherwise
        """
        # Look for experience section
        exp_section = re.search(
            r'(?:experience|work history|employment)[\s*:]+([^]*?)(?:\n\n|\n[A-Z][a-z]+[a-z\s]*:|\Z)',
            resume_text,
            re.IGNORECASE
        )
        
        if exp_section:
            text = exp_section.group(1)
            # Extract first position
            position_match = re.search(r'([A-Z][a-z\s]+(?:Engineer|Developer|Manager|Analyst))', text)
            if position_match:
                return Evidence(
                    claim="Professional work experience",
                    evidence_text=position_match.group(),
                    context=text[:100],
                    confidence="high"
                )
        
        return None
    
    def _find_responsibility_evidence(self, resume_text: str) -> Optional[Evidence]:
        """
        Find evidence of responsibilities in resume text.
        
        Args:
            resume_text: Raw resume text
            
        Returns:
            Evidence object if found, None otherwise
        """
        # Look for bullet points with action verbs
        bullet_matches = re.findall(r'[-•]\s*([A-Z][a-z]+(?:\s+[a-z]+){5,30})', resume_text)
        
        if bullet_matches:
            return Evidence(
                claim="Relevant responsibilities",
                evidence_text=bullet_matches[0],
                context=bullet_matches[0],
                confidence="high"
            )
        
        return None
    
    def _find_education_evidence(self, resume_text: str) -> Optional[Evidence]:
        """
        Find evidence of education in resume text.
        
        Args:
            resume_text: Raw resume text
            
        Returns:
            Evidence object if found, None otherwise
        """
        # Look for education section
        edu_section = re.search(
            r'(?:education|academic)[\s*:]+([^]*?)(?:\n\n|\n[A-Z][a-z]+[a-z\s]*:|\Z)',
            resume_text,
            re.IGNORECASE
        )
        
        if edu_section:
            text = edu_section.group(1)
            # Look for degree
            degree_match = re.search(r'(Bachelor|Master|PhD|Doctorate|B\.S\.|M\.S\.|MBA)', text)
            if degree_match:
                return Evidence(
                    claim="Formal education",
                    evidence_text=degree_match.group(),
                    context=text[:100],
                    confidence="high"
                )
        
        return None
    
    def _generate_skill_suggestions(self, skill: str) -> List[str]:
        """
        Generate suggestions for improving a skill gap.
        
        Args:
            skill: Missing skill
            
        Returns:
            List of suggestion strings
        """
        suggestions = [
            f"Add {skill} to skills section if you have experience",
            f"Highlight any projects using {skill}",
            f"Consider online courses or certifications in {skill}"
        ]
        return suggestions
    
    def to_json(self, result: ExplanationResult) -> str:
        """
        Convert explanation result to JSON string.
        
        Args:
            result: ExplanationResult object
            
        Returns:
            JSON string
        """
        # Convert dataclass to dict
        result_dict = asdict(result)
        
        return json.dumps(result_dict, indent=2)


# ============================================================================
# SAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DETERMINISTIC EXPLANATION ENGINE - SAMPLE USAGE")
    print("=" * 70)
    
    # Initialize explainer
    explainer = DeterministicExplainer()
    
    # Sample parsed resume
    parsed_resume = {
        "raw_text": """
        John Doe - Senior Software Engineer
        
        EXPERIENCE
        Senior Software Engineer at Tech Corp (2020 - Present)
        - Developed RESTful APIs using Python and Django
        - Led a team of 5 developers
        - Implemented CI/CD pipelines with Jenkins and Docker
        - Optimized database queries reducing response time by 40%
        
        Software Developer at Startup Inc (2018 - 2020)
        - Built web applications using React and Node.js
        - Collaborated with cross-functional teams
        
        EDUCATION
        Bachelor of Science in Computer Science - State University (2014 - 2018)
        
        SKILLS
        Python, Django, React, SQL, Docker, Kubernetes, AWS, Git
        """,
        "years_experience": 5.0,
        "education_level": "bachelor"
    }
    
    # Sample parsed JD
    parsed_jd = {
        "raw_text": "Senior Backend Engineer position requiring Python, Django, SQL, AWS, Kubernetes",
        "required_skills": ["python", "django", "sql", "aws", "kubernetes"]
    }
    
    # Sample scoring breakdown
    scoring_breakdown = {
        "overall_score": 75,
        "required_skills": {
            "score": 0.8,
            "matched": ["python", "django", "sql"],
            "partial": ["aws"],
            "missing": ["kubernetes"]
        },
        "experience": {
            "score": 0.9,
            "matched": ["5.0 years"]
        },
        "responsibilities": {
            "score": 0.75
        },
        "education": {
            "score": 1.0
        },
        "semantic_similarity": {
            "score": 0.7
        }
    }
    
    # Generate explanation
    print("\nGenerating explanation...\n")
    result = explainer.explain(
        parsed_resume=parsed_resume,
        parsed_jd=parsed_jd,
        scoring_breakdown=scoring_breakdown,
        matched_skills=["python", "django", "sql"],
        missing_skills=["kubernetes"],
        semantic_scores={"overall_semantic_score": 0.7}
    )
    
    # Print results
    print("-" * 70)
    print("STRENGTHS")
    print("-" * 70)
    for strength in result.strengths:
        print(f"\n{strength.category.upper()} (Impact: {strength.impact})")
        print(f"  {strength.description}")
        for evidence in strength.evidence:
            print(f"  Evidence: \"{evidence.evidence_text}\"")
            print(f"  Context: \"{evidence.context}\"")
    
    print("\n" + "-" * 70)
    print("GAPS")
    print("-" * 70)
    for gap in result.gaps:
        print(f"\n{gap.category.upper()} (Severity: {gap.severity})")
        print(f"  {gap.description}")
        print(f"  Suggestions:")
        for suggestion in gap.suggestions:
            print(f"    - {suggestion}")
    
    print("\n" + "-" * 70)
    print("PARTIAL MATCHES")
    print("-" * 70)
    for partial in result.partial_matches:
        print(f"\nSkill: {partial.skill}")
        print(f"  Resume Evidence: \"{partial.resume_evidence}\"")
        print(f"  JD Requirement: {partial.jd_requirement}")
        print(f"  Similarity: {partial.similarity_score:.2f}")
        print(f"  Suggestion: {partial.improvement_suggestion}")
    
    print("\n" + "-" * 70)
    print("ACTIONABLE SUGGESTIONS")
    print("-" * 70)
    for i, suggestion in enumerate(result.actionable_suggestions, 1):
        print(f"{i}. {suggestion}")
    
    print("\n" + "-" * 70)
    print("UNCERTAINTIES")
    print("-" * 70)
    for uncertainty in result.uncertainties:
        print(f"\n{uncertainty.aspect.upper()}")
        print(f"  Reason: {uncertainty.reason}")
        print(f"  Potential Impact: {uncertainty.potential_impact}")
        print(f"  Mitigation: {uncertainty.mitigation}")
    
    print("\n" + "-" * 70)
    print("OVERALL ASSESSMENT")
    print("-" * 70)
    print(result.overall_assessment)
    print(f"\nConfidence Level: {result.confidence_level}")
    
    # JSON output
    print("\n" + "=" * 70)
    print("JSON OUTPUT")
    print("=" * 70)
    print(explainer.to_json(result))
