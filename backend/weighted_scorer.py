"""
Weighted Resume Scoring Engine.

This module implements a deterministic scoring engine that evaluates how well a resume
matches a job description using weighted categories. The scoring is transparent,
explainable, and reproducible.

Weight Distribution:
- required_skills: 0.45 (45%) - Most critical for technical roles
- responsibilities: 0.20 (20%) - Contextual fit and relevant experience
- experience: 0.15 (15%) - Seniority match
- education: 0.10 (10%) - Baseline qualification
- semantic_similarity: 0.10 (10%) - Overall semantic fit

Why These Weights Are Reasonable:
---------------------------------
1. **Skills (45%)**: Skills are the most direct indicator of capability for technical roles.
   A candidate with the right skills can often learn on the job, but missing core skills
   is a significant barrier. This high weight reflects the importance of technical capability.

2. **Responsibilities (20%)**: This measures contextual fit - whether the candidate has
   done similar work. It's important but secondary to actual skills since skills can be
   applied in different contexts.

3. **Experience (15%)**: Years of experience matter but are less critical than skills.
   A brilliant 3-year developer can outperform an average 10-year developer. This weight
   acknowledges seniority without overemphasizing it.

4. **Education (10%)**: Education is often a baseline requirement but frequently flexible
   with equivalent experience. The low weight reflects that many successful developers
   are self-taught or have non-traditional backgrounds.

5. **Semantic Similarity (10%)**: This catches overall fit beyond explicit requirements.
   It's a tiebreaker that can identify candidates who might be a good fit even if they
   don't match all explicit criteria.

Tradeoffs:
----------
- **Skills vs Experience**: Heavy emphasis on skills may overlook candidates with strong
  experience but different tech stacks. However, skills are more transferable than
  domain-specific experience.

- **Education weight**: Low education weight may disadvantage candidates from traditional
  backgrounds, but it also advantages self-taught developers and bootcamp graduates.

- **Semantic similarity**: This is a "catch-all" that can compensate for weaknesses in
  other categories, potentially masking skill gaps. However, it also helps identify
  hidden talent.

- **Deterministic vs Contextual**: The formulas are deterministic for reproducibility,
  but they may miss nuanced context that a human reviewer would catch.

Limitations:
-----------
1. **Skill detection accuracy**: Depends on the quality of skill extraction. Missed skills
   or false positives can significantly impact the score.

2. **Experience quality**: Years of experience doesn't capture quality or relevance.
   10 years of maintenance work ≠ 10 years of feature development.

3. **Education field**: Only captures degree level, not field of study. A PhD in
   literature ≠ PhD in Computer Science for a technical role.

4. **Semantic similarity limitations**: Embeddings capture semantic meaning but not
   practical capability. "Built APIs" and "API development" are similar semantically,
   but the former could mean using a tool while the latter means designing them.

5. **No seniority distinction**: Can't distinguish between junior and senior roles
   of the same skill set.

6. **Industry-specific knowledge**: Doesn't account for domain expertise (e.g., finance,
   healthcare) which may be critical for some roles.

7. **Soft skills**: Doesn't evaluate communication, leadership, or teamwork skills,
   which are often critical for senior roles.

8. **Cultural fit**: Doesn't assess alignment with company culture or values.
"""

import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict


# ============================================================================
# SCORING WEIGHTS CONFIGURATION
# ============================================================================

SCORING_WEIGHTS = {
    "required_skills": 0.45,
    "responsibilities": 0.20,
    "experience": 0.15,
    "education": 0.10,
    "semantic_similarity": 0.10
}

# Verify weights sum to 1.0
assert abs(sum(SCORING_WEIGHTS.values()) - 1.0) < 0.001, "Weights must sum to 1.0"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class CategoryScore:
    """
    Individual category score with detailed breakdown.
    """
    name: str
    score: float  # 0.0 to 1.0
    weight: float
    explanation: str
    matched: List[str]
    missing: List[str]
    partial: List[str]
    penalty: float = 0.0
    bonus: float = 0.0


@dataclass
class ScoringResult:
    """
    Complete scoring result with all category scores and overall score.
    """
    overall_score: int  # 0 to 100
    category_scores: Dict[str, CategoryScore]
    explanation: str
    recommendations: List[str]


# ============================================================================
# SCORING FUNCTIONS
# ============================================================================

class WeightedScoringEngine:
    """
    Deterministic weighted scoring engine for resume-job fit analysis.
    """
    
    def __init__(self, critical_skills: List[str] = None):
        """
        Initialize the scoring engine.
        
        Args:
            critical_skills: List of skills that are considered critical (penalize heavily if missing)
        """
        self.critical_skills = critical_skills or []
    
    def score_required_skills(
        self,
        resume_skills: List[str],
        jd_required_skills: List[str],
        skill_similarities: Dict[str, float] = None
    ) -> CategoryScore:
        """
        Score required skills match.
        
        Formula:
        score = (matched_count + 0.5 * partial_count) / total_required
        penalty = 0.2 * (missing_critical_skills / total_critical_skills)
        
        Args:
            resume_skills: List of skills from resume
            jd_required_skills: List of required skills from JD
            skill_similarities: Optional dict of skill similarity scores (for partial matches)
            
        Returns:
            CategoryScore with detailed breakdown
        """
        if not jd_required_skills:
            return CategoryScore(
                name="required_skills",
                score=0.5,  # Neutral score if no requirements specified
                weight=SCORING_WEIGHTS["required_skills"],
                explanation="No required skills specified in job description",
                matched=[],
                missing=[],
                partial=[]
            )
        
        if not resume_skills:
            return CategoryScore(
                name="required_skills",
                score=0.0,
                weight=SCORING_WEIGHTS["required_skills"],
                explanation="No skills detected in resume",
                matched=[],
                missing=jd_required_skills.copy(),
                partial=[],
                penalty=0.0
            )
        
        # Normalize to lowercase for comparison
        resume_skills_lower = [s.lower() for s in resume_skills]
        jd_required_lower = [s.lower() for s in jd_required_skills]
        
        # Classify matches
        matched = []
        partial = []
        missing = []
        
        for jd_skill in jd_required_lower:
            if jd_skill in resume_skills_lower:
                matched.append(jd_skill)
            elif skill_similarities and jd_skill in skill_similarities:
                # Use similarity threshold for partial matches
                if skill_similarities[jd_skill] >= 0.65:
                    partial.append(jd_skill)
                else:
                    missing.append(jd_skill)
            else:
                missing.append(jd_skill)
        
        # Calculate base score
        matched_count = len(matched)
        partial_count = len(partial)
        total_required = len(jd_required_skills)
        
        base_score = (matched_count + 0.5 * partial_count) / max(total_required, 1)
        base_score = min(base_score, 1.0)
        
        # Apply penalty for missing critical skills
        penalty = 0.0
        missing_critical = [s for s in missing if s in [c.lower() for c in self.critical_skills]]
        if missing_critical:
            penalty = 0.2 * (len(missing_critical) / max(len(self.critical_skills), 1))
        
        final_score = max(0.0, base_score - penalty)
        
        # Generate explanation
        explanation_parts = []
        explanation_parts.append(f"Matched {matched_count}/{total_required} required skills")
        if partial_count > 0:
            explanation_parts.append(f"({partial_count} partial matches)")
        if penalty > 0:
            explanation_parts.append(f"Penalty: {penalty:.2f} for missing critical skills: {missing_critical}")
        
        explanation = ". ".join(explanation_parts)
        
        return CategoryScore(
            name="required_skills",
            score=final_score,
            weight=SCORING_WEIGHTS["required_skills"],
            explanation=explanation,
            matched=matched,
            missing=missing,
            partial=partial,
            penalty=penalty
        )
    
    def score_responsibilities(
        self,
        resume_resp: List[str],
        jd_resp: List[str],
        similarity_scores: List[float] = None
    ) -> CategoryScore:
        """
        Score responsibilities match.
        
        Formula:
        score = average_similarity_score
        bonus = 0.1 if top_similarity > 0.85 (exceptional match)
        
        Args:
            resume_resp: List of responsibilities from resume
            jd_resp: List of responsibilities from JD
            similarity_scores: Optional list of similarity scores for each JD responsibility
            
        Returns:
            CategoryScore with detailed breakdown
        """
        if not jd_resp:
            return CategoryScore(
                name="responsibilities",
                score=0.5,
                weight=SCORING_WEIGHTS["responsibilities"],
                explanation="No responsibilities specified in job description",
                matched=[],
                missing=[],
                partial=[]
            )
        
        if not resume_resp:
            return CategoryScore(
                name="responsibilities",
                score=0.0,
                weight=SCORING_WEIGHTS["responsibilities"],
                explanation="No responsibilities detected in resume",
                matched=[],
                missing=jd_resp.copy(),
                partial=[]
            )
        
        # If similarity scores provided, use them
        if similarity_scores:
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
            max_similarity = max(similarity_scores)
        else:
            # Fallback to simple keyword overlap
            avg_similarity = self._calculate_keyword_overlap(resume_resp, jd_resp)
            max_similarity = avg_similarity
        
        # Apply bonus for exceptional match
        bonus = 0.1 if max_similarity > 0.85 else 0.0
        final_score = min(1.0, avg_similarity + bonus)
        
        # Generate explanation
        explanation = f"Average responsibility similarity: {avg_similarity:.2f}"
        if bonus > 0:
            explanation += f". Bonus: {bonus:.2f} for exceptional match"
        
        return CategoryScore(
            name="responsibilities",
            score=final_score,
            weight=SCORING_WEIGHTS["responsibilities"],
            explanation=explanation,
            matched=[],
            missing=[],
            partial=[],
            bonus=bonus
        )
    
    def score_experience(
        self,
        resume_years: float,
        jd_min_years: float,
        adjacent_experience: bool = False
    ) -> CategoryScore:
        """
        Score experience match.
        
        Formula:
        ratio = resume_years / max(jd_min_years, 1)
        score = min(ratio, 1.0) if ratio >= 0.7 else ratio * 0.7
        bonus = 0.1 if adjacent_experience and ratio >= 0.8
        
        Args:
            resume_years: Years of experience from resume
            jd_min_years: Minimum years required from JD
            adjacent_experience: Whether candidate has adjacent/related experience
            
        Returns:
            CategoryScore with detailed breakdown
        """
        required_years = max(jd_min_years, 1)
        actual_years = max(resume_years, 0)
        
        ratio = actual_years / required_years
        
        # Calculate base score
        if ratio >= 0.7:
            base_score = min(ratio, 1.0)
        else:
            base_score = ratio * 0.7
        
        # Apply bonus for adjacent experience
        bonus = 0.1 if adjacent_experience and ratio >= 0.8 else 0.0
        final_score = min(1.0, base_score + bonus)
        
        # Generate explanation
        explanation = f"Candidate has {actual_years:.1f} years, requires {required_years:.1f} years"
        if bonus > 0:
            explanation += f". Bonus: {bonus:.2f} for adjacent experience"
        elif ratio < 0.7:
            explanation += f". Penalty applied for insufficient experience"
        
        return CategoryScore(
            name="experience",
            score=final_score,
            weight=SCORING_WEIGHTS["experience"],
            explanation=explanation,
            matched=[f"{actual_years:.1f} years"] if ratio >= 1.0 else [],
            missing=[f"Need {required_years:.1f}y, have {actual_years:.1f}y"] if ratio < 0.8 else [],
            partial=[],
            bonus=bonus
        )
    
    def score_education(
        self,
        resume_education: str,
        jd_required_education: str
    ) -> CategoryScore:
        """
        Score education match.
        
        Formula:
        score = 1.0 if candidate >= required
        score = 0.5 if candidate is one level below
        score = 0.0 if candidate is two+ levels below
        
        Education ladder: unknown=0, high_school=1, bachelor=2, master=3, phd=4
        
        Args:
            resume_education: Education level from resume (e.g., "bachelor", "master", "phd")
            jd_required_education: Required education level from JD
            
        Returns:
            CategoryScore with detailed breakdown
        """
        education_ladder = {
            "unknown": 0,
            "high_school": 1,
            "bachelor": 2,
            "master": 3,
            "phd": 4
        }
        
        candidate_level = education_ladder.get(resume_education.lower(), 0)
        required_level = education_ladder.get(jd_required_education.lower(), 0)
        
        level_diff = candidate_level - required_level
        
        # Calculate score
        if level_diff >= 0:
            score = 1.0
        elif level_diff == -1:
            score = 0.5
        else:
            score = 0.0
        
        # Generate explanation
        explanation = f"Candidate education: {resume_education}, Required: {jd_required_education}"
        if score == 1.0:
            explanation += ". Meets or exceeds requirement"
        elif score == 0.5:
            explanation += ". One level below requirement"
        else:
            explanation += ". Below requirement"
        
        return CategoryScore(
            name="education",
            score=score,
            weight=SCORING_WEIGHTS["education"],
            explanation=explanation,
            matched=[resume_education] if score >= 1.0 else [],
            missing=[jd_required_education] if score < 1.0 else [],
            partial=[]
        )
    
    def score_semantic_similarity(
        self,
        semantic_score: float
    ) -> CategoryScore:
        """
        Score semantic similarity between resume and JD.
        
        Formula:
        score = semantic_score (already normalized 0-1)
        
        Args:
            semantic_score: Semantic similarity score from embedding model (0-1)
            
        Returns:
            CategoryScore with detailed breakdown
        """
        # Semantic score is already normalized 0-1
        score = max(0.0, min(1.0, semantic_score))
        
        # Generate explanation
        if score >= 0.85:
            interpretation = "Excellent semantic match"
        elif score >= 0.70:
            interpretation = "Strong semantic match"
        elif score >= 0.55:
            interpretation = "Moderate semantic match"
        elif score >= 0.40:
            interpretation = "Weak semantic match"
        else:
            interpretation = "Poor semantic match"
        
        explanation = f"Semantic similarity: {score:.2f} - {interpretation}"
        
        return CategoryScore(
            name="semantic_similarity",
            score=score,
            weight=SCORING_WEIGHTS["semantic_similarity"],
            explanation=explanation,
            matched=[],
            missing=[],
            partial=[]
        )
    
    def calculate_overall_score(
        self,
        category_scores: Dict[str, CategoryScore]
    ) -> Tuple[int, str]:
        """
        Calculate overall weighted score.
        
        Formula:
        overall = sum(category_score * category_weight) * 100
        
        Args:
            category_scores: Dictionary of category scores
            
        Returns:
            Tuple of (overall_score_0_to_100, explanation)
        """
        weighted_sum = 0.0
        
        for category_name, category_score in category_scores.items():
            weighted_sum += category_score.score * category_score.weight
        
        overall_score = int(round(weighted_sum * 100))
        overall_score = max(0, min(100, overall_score))
        
        # Generate explanation
        explanation_parts = []
        for name, score in category_scores.items():
            contribution = score.score * score.weight
            explanation_parts.append(
                f"{name}: {score.score:.2f} × {score.weight:.2f} = {contribution:.2f}"
            )
        
        explanation = "Overall score: " + " + ".join(explanation_parts) + f" = {overall_score}/100"
        
        return overall_score, explanation
    
    def generate_recommendations(
        self,
        category_scores: Dict[str, CategoryScore]
    ) -> List[str]:
        """
        Generate improvement recommendations based on category scores.
        
        Args:
            category_scores: Dictionary of category scores
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Skills recommendations
        skills_score = category_scores.get("required_skills")
        if skills_score and skills_score.score < 0.8:
            if skills_score.missing:
                missing_list = ', '.join(skills_score.missing[:5])
                recommendations.append(f"To strengthen your application, consider gaining experience with {missing_list}. Online courses, certifications, or personal projects can help build these skills.")
            if skills_score.partial:
                partial_list = ', '.join(skills_score.partial[:3])
                recommendations.append(f"You have some familiarity with {partial_list}. Highlight specific projects or experiences where you've used these technologies to demonstrate your proficiency.")
        
        # Experience recommendations
        exp_score = category_scores.get("experience")
        if exp_score and exp_score.score < 0.7:
            recommendations.append("Your experience level is below the requirement. Consider taking on more challenging projects or seeking opportunities that align with the seniority level of this role.")
        
        # Responsibilities recommendations
        resp_score = category_scores.get("responsibilities")
        if resp_score and resp_score.score < 0.6:
            recommendations.append("Enhance your resume by quantifying your achievements with specific metrics and outcomes. Use action verbs to describe your responsibilities and highlight results that demonstrate your impact.")
        
        # Education recommendations
        edu_score = category_scores.get("education")
        if edu_score and edu_score.score < 0.5:
            recommendations.append("Your education level is below the requirement. Consider pursuing relevant certifications or highlighting equivalent experience that demonstrates your qualifications for this role.")
        
        # Semantic recommendations
        semantic_score = category_scores.get("semantic_similarity")
        if semantic_score and semantic_score.score < 0.5:
            recommendations.append("Improve your resume's alignment with the job description by using similar terminology and keywords. Review the job requirements and incorporate relevant language into your resume.")
        
        if not recommendations:
            recommendations.append("Your profile shows strong alignment with this position. Continue to highlight your key strengths and achievements in your application materials.")
        
        return recommendations
    
    def score_resume(
        self,
        resume_data: Dict,
        jd_data: Dict,
        semantic_score: float = None,
        skill_similarities: Dict[str, float] = None,
        resp_similarities: List[float] = None
    ) -> ScoringResult:
        """
        Score a resume against a job description using all categories.
        
        Args:
            resume_data: Dictionary containing resume data (skills, responsibilities, experience, education)
            jd_data: Dictionary containing JD data (required_skills, responsibilities, min_experience, required_education)
            semantic_score: Optional semantic similarity score (0-1)
            skill_similarities: Optional dict of skill similarity scores
            resp_similarities: Optional list of responsibility similarity scores
            
        Returns:
            ScoringResult with overall score and category breakdown
        """
        # Score each category
        category_scores = {}
        
        # Required skills
        category_scores["required_skills"] = self.score_required_skills(
            resume_data.get("skills", []),
            jd_data.get("required_skills", []),
            skill_similarities
        )
        
        # Responsibilities
        category_scores["responsibilities"] = self.score_responsibilities(
            resume_data.get("responsibilities", []),
            jd_data.get("responsibilities", []),
            resp_similarities
        )
        
        # Experience
        category_scores["experience"] = self.score_experience(
            resume_data.get("years_experience", 0),
            jd_data.get("min_years_experience", 0),
            resume_data.get("adjacent_experience", False)
        )
        
        # Education
        category_scores["education"] = self.score_education(
            resume_data.get("education_level", "unknown"),
            jd_data.get("required_education", "unknown")
        )
        
        # Semantic similarity
        if semantic_score is not None:
            category_scores["semantic_similarity"] = self.score_semantic_similarity(semantic_score)
        else:
            # Default to neutral if not provided
            category_scores["semantic_similarity"] = CategoryScore(
                name="semantic_similarity",
                score=0.5,
                weight=SCORING_WEIGHTS["semantic_similarity"],
                explanation="Semantic similarity not provided",
                matched=[],
                missing=[],
                partial=[]
            )
        
        # Calculate overall score
        overall_score, overall_explanation = self.calculate_overall_score(category_scores)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(category_scores)
        
        # Combine explanations
        category_explanations = "\n".join([
            f"- {cat.name}: {cat.explanation}" for cat in category_scores.values()
        ])
        
        full_explanation = f"{overall_explanation}\n\nCategory Breakdown:\n{category_explanations}"
        
        return ScoringResult(
            overall_score=overall_score,
            category_scores=category_scores,
            explanation=full_explanation,
            recommendations=recommendations
        )
    
    def _calculate_keyword_overlap(self, list1: List[str], list2: List[str]) -> float:
        """
        Calculate simple keyword overlap between two lists of strings.
        
        Args:
            list1: First list of strings
            list2: Second list of strings
            
        Returns:
            Overlap ratio (0-1)
        """
        if not list2:
            return 0.5  # Neutral if no requirements
        
        if not list1:
            return 0.0
        
        # Tokenize and normalize
        tokens1 = set(" ".join(list1).lower().split())
        tokens2 = set(" ".join(list2).lower().split())
        
        if not tokens2:
            return 0.5
        
        # Calculate overlap
        intersection = tokens1.intersection(tokens2)
        overlap_ratio = len(intersection) / len(tokens2)
        
        return overlap_ratio
    
    def to_json(self, result: ScoringResult) -> str:
        """
        Convert scoring result to JSON string.
        
        Args:
            result: ScoringResult object
            
        Returns:
            JSON string
        """
        # Convert dataclass to dict
        result_dict = asdict(result)
        
        # Convert CategoryScore objects to dicts
        result_dict["category_scores"] = {
            name: asdict(score) for name, score in result.category_scores.items()
        }
        
        return json.dumps(result_dict, indent=2)


# ============================================================================
# SAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("WEIGHTED RESUME SCORING ENGINE - SAMPLE USAGE")
    print("=" * 70)
    
    # Initialize engine with critical skills
    engine = WeightedScoringEngine(critical_skills=["python", "sql"])
    
    # Sample resume data
    resume_data = {
        "skills": ["python", "django", "react", "sql", "docker"],
        "responsibilities": [
            "Developed RESTful APIs using Python and Django",
            "Led a team of 5 developers",
            "Implemented CI/CD pipelines"
        ],
        "years_experience": 5.0,
        "education_level": "bachelor",
        "adjacent_experience": True
    }
    
    # Sample job description data
    jd_data = {
        "required_skills": ["python", "django", "sql", "aws", "kubernetes"],
        "responsibilities": [
            "Design scalable backend systems",
            "Develop RESTful APIs",
            "Collaborate with cross-functional teams"
        ],
        "min_years_experience": 3.0,
        "required_education": "bachelor"
    }
    
    # Sample similarity scores (would come from semantic similarity module)
    skill_similarities = {
        "aws": 0.72,  # Partial match (has cloud experience but not AWS specifically)
        "kubernetes": 0.45  # Weak match (has Docker but not Kubernetes)
    }
    
    resp_similarities = [0.85, 0.78, 0.65]  # Similarity scores for each JD responsibility
    
    semantic_score = 0.75  # Overall semantic similarity
    
    # Score the resume
    print("\nScoring resume against job description...\n")
    result = engine.score_resume(
        resume_data,
        jd_data,
        semantic_score=semantic_score,
        skill_similarities=skill_similarities,
        resp_similarities=resp_similarities
    )
    
    # Print results
    print("-" * 70)
    print("OVERALL SCORE")
    print("-" * 70)
    print(f"Score: {result.overall_score}/100")
    
    print("\n" + "-" * 70)
    print("CATEGORY BREAKDOWN")
    print("-" * 70)
    for name, score in result.category_scores.items():
        print(f"\n{name.upper()} (Weight: {score.weight:.2f})")
        print(f"  Score: {score.score:.2f}")
        print(f"  Explanation: {score.explanation}")
        if score.matched:
            print(f"  Matched: {score.matched}")
        if score.missing:
            print(f"  Missing: {score.missing}")
        if score.partial:
            print(f"  Partial: {score.partial}")
        if score.penalty > 0:
            print(f"  Penalty: {score.penalty:.2f}")
        if score.bonus > 0:
            print(f"  Bonus: {score.bonus:.2f}")
    
    print("\n" + "-" * 70)
    print("RECOMMENDATIONS")
    print("-" * 70)
    for i, rec in enumerate(result.recommendations, 1):
        print(f"{i}. {rec}")
    
    print("\n" + "-" * 70)
    print("FULL EXPLANATION")
    print("-" * 70)
    print(result.explanation)
    
    # JSON output
    print("\n" + "=" * 70)
    print("JSON OUTPUT")
    print("=" * 70)
    print(engine.to_json(result))
