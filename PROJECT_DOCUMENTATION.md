# Resume Scoring System - Project Documentation

**Version**: 1.0  
**Date**: May 2026  
**Author**: Development Team

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Objectives](#2-objectives)
3. [System Architecture](#3-system-architecture)
4. [Technologies Used](#4-technologies-used)
5. [Deterministic Components](#5-deterministic-components)
6. [Embedding-Based Similarity](#6-embedding-based-similarity)
7. [Scoring Methodology](#7-scoring-methodology)
8. [Explainability](#8-explainability)
9. [Uncertainty Handling](#9-uncertainty-handling)
10. [Limitations](#10-limitations)
11. [Future Improvements](#11-future-improvements)
12. [Sample Outputs](#12-sample-outputs)
13. [Design Decisions](#13-design-decisions)
14. [Why Hybrid AI Approach](#14-why-hybrid-ai-approach)

---

## 1. Problem Statement

### 1.1 Background

In today's competitive job market, recruiters and hiring managers receive hundreds of resumes for each open position. Manually screening and evaluating each resume against job requirements is time-consuming, subjective, and prone to inconsistencies. Traditional keyword-based resume screening systems often fail to capture semantic meaning, leading to both false positives (candidates who appear qualified but are not) and false negatives (qualified candidates who are filtered out due to terminology differences).

### 1.2 Challenges

1. **Semantic Understanding**: Keyword matching cannot distinguish between "Python developer" and "software engineer with Python experience"
2. **Context Awareness**: Systems struggle to understand context (e.g., "bank" in finance vs. "bank" in geography)
3. **Explainability**: Black-box AI models provide scores without justification, making it difficult for recruiters to trust the results
4. **Bias and Fairness**: Unchecked AI systems may perpetuate existing biases in hiring
5. **Cost**: Pure LLM-based solutions are expensive and slow for high-volume screening
6. **Transparency**: Recruiters need to understand why a candidate received a particular score

### 1.3 Problem Definition

Design and implement a resume scoring system that:
- Accurately evaluates candidate-job fit using multiple scoring dimensions
- Provides transparent, explainable results with evidence
- Balances accuracy with cost-effectiveness
- Identifies uncertainty and assumptions in the analysis
- Supports recruiters in making informed hiring decisions

---

## 2. Objectives

### 2.1 Primary Objectives

1. **Multi-Dimensional Scoring**: Evaluate resumes across multiple dimensions (skills, experience, education, responsibilities, semantic fit)
2. **Deterministic Core**: Use deterministic, rule-based logic for core scoring to ensure reproducibility and transparency
3. **Semantic Understanding**: Incorporate semantic similarity to capture meaning beyond keywords
4. **Explainability**: Provide detailed explanations with evidence from resume text
5. **Uncertainty Quantification**: Identify and flag uncertainties and assumptions
6. **Cost-Effectiveness**: Minimize API costs while maintaining accuracy

### 2.2 Secondary Objectives

1. **Modular Architecture**: Design a system with clear separation of concerns
2. **Extensibility**: Allow easy addition of new scoring dimensions and models
3. **User-Friendly Interface**: Provide an intuitive web interface for users
4. **Performance**: Ensure fast response times for practical use
5. **Documentation**: Maintain comprehensive documentation for maintenance and extension

---

## 3. System Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                           │
│                    (Modern Bootstrap Dashboard)                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ HTTP Request (multipart/form-data or JSON)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Layer (Flask/FastAPI)                     │
│                    - Request Validation                            │
│                    - File Handling                                  │
│                    - Response Formatting                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ Orchestration
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Scoring Pipeline (Orchestrator)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   Parser     │  │   Scorer     │  │  Explainer   │           │
│  │  (LLM/Det.)  │  │  (Det./Emb.) │  │  (LLM/Det.)  │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Result Aggregation                               │
│              - Score Combination                                   │
│              - Confidence Assessment                               │
│              - Uncertainty Flagging                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Response to User                              │
│         {overall_score, breakdown, explanation, suggestions}      │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Breakdown

#### 3.2.1 Parser Module
- **LLM-Based Parser**: Uses OpenAI GPT-4o-mini for structured extraction
- **Deterministic Parser**: Uses spaCy and regex for rule-based extraction
- **Output**: Structured data (skills, experience, education, responsibilities)

#### 3.2.2 Scoring Module
- **Skill Scorer**: Sentence-transformers embeddings for semantic matching
- **Experience Scorer**: Deterministic ratio-based calculation
- **Education Scorer**: Level ladder comparison
- **Responsibility Scorer**: TF-IDF cosine similarity
- **Semantic Scorer**: Overall text similarity via embeddings

#### 3.2.3 Explanation Module
- **LLM Explainer**: Natural language explanations and suggestions
- **Deterministic Explainer**: Evidence-based explanations with citations

#### 3.2.4 Uncertainty Module
- **Ambiguity Detection**: Identifies approximate terms and unclear statements
- **Missing Information**: Flags incomplete resume sections
- **Assumption Tracking**: Records assumptions made during analysis
- **Confidence Calculation**: Weighted penalty-based confidence scoring

### 3.3 Data Flow

1. **Input**: Resume file/text + Job description text
2. **Parsing**: Extract structured information from both inputs
3. **Scoring**: Compute scores for each dimension
4. **Combination**: Weighted aggregation of category scores
5. **Explanation**: Generate human-readable explanations
6. **Uncertainty**: Identify and quantify uncertainties
7. **Output**: Structured JSON with all results

---

## 4. Technologies Used

### 4.1 Backend Technologies

| Technology | Purpose | Version |
|------------|---------|---------|
| Python | Primary language | 3.11+ |
| FastAPI | REST API framework | 0.104.1 |
| Flask | Alternative REST API framework | 3.0.0 |
| Pydantic | Data validation | 2.5.0 |
| OpenAI API | LLM for parsing and explanation | GPT-4o-mini |
| sentence-transformers | Semantic embeddings | all-MiniLM-L6-v2 |
| scikit-learn | TF-IDF and similarity calculations | 1.3.2 |
| spaCy | NLP for deterministic parsing | en_core_web_sm |
| numpy | Numerical computations | 1.26.2 |
| pdfplumber | PDF text extraction | 0.10.3 |
| python-docx | DOCX text extraction | 1.1.0 |

### 4.2 Frontend Technologies

| Technology | Purpose | Version |
|------------|---------|---------|
| HTML5 | Markup | 5 |
| Bootstrap 5 | UI framework | 5.3.0 |
| Bootstrap Icons | Icon library | 1.11.0 |
| CSS3 | Styling | 3 |
| JavaScript | Client-side logic | ES6+ |

### 4.3 Technology Rationale

#### 4.3.1 Python
- Extensive ML/AI ecosystem
- Strong typing support with Pydantic
- Excellent async support (FastAPI)
- Widely used in industry

#### 4.3.2 sentence-transformers
- Deterministic (same input → same output)
- No API costs (runs locally)
- Fast inference
- Good quality for semantic similarity
- Easy to integrate

#### 4.3.3 OpenAI GPT-4o-mini
- Excellent for parsing unstructured text
- Structured output support
- Cost-effective for explanation generation
- High-quality natural language generation

#### 4.3.4 FastAPI vs Flask
- **FastAPI**: Modern, fast, automatic API documentation, async support
- **Flask**: Simpler, more mature, widely used
- Both implemented to provide flexibility

---

## 5. Deterministic Components

### 5.1 Overview

Deterministic components are rule-based, reproducible functions that always produce the same output for the same input. These form the core of the scoring system to ensure transparency and auditability.

### 5.2 Deterministic Parser

**File**: `deterministic_parser.py`

**Features**:
- Regex-based extraction of skills, education, experience
- spaCy NLP for entity recognition
- Pattern matching for certifications and projects
- Date range extraction for experience calculation

**Advantages**:
- No API costs
- Fast execution
- Transparent logic
- Reproducible results

**Limitations**:
- May miss uncommon phrasings
- Requires pattern maintenance
- Less flexible than LLM parsing

### 5.3 Skill Extraction Engine

**File**: `skill_extractor.py`

**Features**:
- Predefined skill database with categories
- Synonym and related technology detection
- Fuzzy matching for typos
- Normalized skill output

**Algorithm**:
```
1. Exact match against skill database
2. Synonym matching
3. Related technology matching
4. Fuzzy matching (if enabled)
5. Deduplication and normalization
```

### 5.4 Weighted Scoring Engine

**File**: `weighted_scorer.py`

**Scoring Formulas**:

**Skills**:
```
score = (matched + 0.5 × partial) / total_required
penalty = 0.2 × (missing_critical / total_critical)
final_score = max(0, base_score - penalty)
```

**Experience**:
```
ratio = actual_years / required_years
if ratio >= 0.7:
    base_score = min(ratio, 1.0)
else:
    base_score = ratio × 0.7
bonus = 0.1 if adjacent_experience and ratio >= 0.8
final_score = min(1.0, base_score + bonus)
```

**Education**:
```
level_diff = candidate_level - required_level
if level_diff >= 0: score = 1.0
elif level_diff == -1: score = 0.5
else: score = 0.0
```

### 5.5 Deterministic Explainer

**File**: `deterministic_explainer.py`

**Features**:
- Evidence extraction from resume text
- Strength identification with citations
- Gap detection with severity levels
- Actionable suggestions generation
- Uncertainty identification

**Advantages**:
- Evidence-based explanations
- No generic feedback
- Citations from resume text
- Transparent reasoning

---

## 6. Embedding-Based Similarity

### 6.1 Overview

Embedding-based similarity uses sentence-transformers to capture semantic meaning beyond exact keyword matching. This enables the system to recognize related concepts and synonyms.

### 6.2 Model Selection

**Model**: `sentence-transformers/all-MiniLM-L6-v2`

**Rationale**:
- Lightweight (384 dimensions)
- Fast inference
- Good quality for semantic similarity
- No API costs
- Deterministic output

### 6.3 Applications

#### 6.3.1 Skill Matching
- Matches skills with semantic similarity (threshold: 0.85 for exact, 0.65-0.84 for partial)
- Captures related technologies (e.g., "pandas" ≈ "numpy")
- Handles synonyms (e.g., "ML" ≈ "Machine Learning")

#### 6.3.2 Responsibility Matching
- Computes cosine similarity between resume and JD responsibilities
- Identifies best matches for each JD requirement
- Provides similarity scores for transparency

#### 6.3.3 Overall Semantic Fit
- Computes similarity between full resume and JD texts
- Acts as a tiebreaker for edge cases
- Catches overall fit beyond explicit requirements

### 6.4 Why Embeddings Help Beyond Keywords

1. **Semantic Understanding**: Captures meaning, not just exact words
   - Example: "Python developer" and "Software engineer with Python experience" are semantically similar

2. **Synonym Recognition**: Related terms map to similar vector spaces
   - Example: "ML", "Machine Learning", and "AI" have similar embeddings

3. **Context Awareness**: Embeddings understand words in context
   - Example: "bank" (financial) vs "bank" (river) have different embeddings

4. **Phrase-Level Matching**: Captures multi-word concepts
   - Example: "machine learning" is treated as a single concept

5. **Language Agnostic**: Works across different phrasings
   - Example: "built APIs" and "API development" are semantically similar

---

## 7. Scoring Methodology

### 7.1 Weight Distribution

| Category | Weight | Rationale |
|----------|--------|-----------|
| Skills | 45% | Most critical for technical roles; direct capability indicator |
| Responsibilities | 20% | Contextual fit; relevant experience |
| Experience | 15% | Seniority match; less critical than skills |
| Education | 10% | Baseline requirement; flexible with experience |
| Semantic Similarity | 10% | Overall fit; tiebreaker |

**Total**: 100%

### 7.2 Scoring Pipeline

```
1. Parse Resume and JD (concurrent)
   ├─ LLM-based parsing (primary)
   └─ Deterministic parsing (fallback)

2. Extract Skills
   ├─ Skill database matching
   ├─ Synonym detection
   ├─ Related technology matching
   └─ Fuzzy matching

3. Compute Category Scores
   ├─ Skills: Embedding similarity
   ├─ Experience: Ratio-based calculation
   ├─ Education: Level ladder comparison
   ├─ Responsibilities: TF-IDF similarity
   └─ Semantic: Overall text similarity

4. Combine Scores
   ├─ Weighted sum: Σ(score × weight)
   ├─ Confidence assessment
   └─ Uncertainty flagging

5. Generate Explanation
   ├─ Evidence extraction
   ├─ Strength identification
   ├─ Gap detection
   └─ Suggestion generation

6. Return Results
   └─ Structured JSON response
```

### 7.3 Score Interpretation

| Score Range | Interpretation | Action |
|-------------|----------------|--------|
| 80-100 | Excellent match | Strong candidate, proceed with interview |
| 65-79 | Strong match | Good candidate, consider for interview |
| 50-64 | Moderate match | Potential candidate, review manually |
| 0-49 | Weak match | Unlikely to be a good fit |

---

## 8. Explainability

### 8.1 Importance of Explainability

Explainability is critical for:
- **Recruiter Trust**: Understanding why a score was assigned
- **Candidate Feedback**: Providing actionable improvement suggestions
- **Compliance**: Meeting regulatory requirements for fair hiring
- **Debugging**: Identifying and fixing scoring errors
- **Transparency**: Avoiding black-box AI concerns

### 8.2 Explanation Components

#### 8.2.1 Evidence-Based Explanations
- Cites specific resume text as evidence
- Shows matched skills with context
- Provides examples of strengths

#### 8.2.2 Category Breakdown
- Shows individual category scores
- Displays matched/missing/partial items
- Explains why each score was assigned

#### 8.2.3 Suggestions
- Actionable improvement recommendations
- Specific to identified gaps
- Prioritized by severity

#### 8.2.4 Uncertainty Flags
- Identifies assumptions made
- Highlights areas of low confidence
- Suggests how to reduce uncertainty

### 8.3 Dual Explanation Approach

**LLM Explainer**:
- Natural language generation
- Human-readable summaries
- Contextual suggestions
- Used for final output

**Deterministic Explainer**:
- Evidence extraction with citations
- Rule-based strength/gap identification
- Transparent reasoning
- Used for detailed analysis

---

## 9. Uncertainty Handling

### 9.1 Sources of Uncertainty

1. **Ambiguity**: Approximate terms ("approximately 5 years")
2. **Missing Information**: Incomplete resume sections
3. **Inference**: Skills inferred from related technologies
4. **Parsing Errors**: Incorrect extraction from unstructured text
5. **Semantic Mismatch**: Different terminology for same concept

### 9.2 Uncertainty Detection

**File**: `uncertainty_estimator.py`

**Detection Methods**:
- Pattern matching for ambiguous terms
- Validation of required fields
- Experience clarity assessment
- Cloud experience inference detection
- Partial match identification

### 9.3 Confidence Calculation

**Formula**:
```
confidence = 1.0 - weighted_penalty

Penalty Weights:
- High severity: 0.15
- Medium severity: 0.08
- Low severity: 0.03

Minimum confidence: 0.2 (to avoid zero confidence)
```

### 9.4 Confidence Levels

| Confidence Score | Level | Interpretation |
|------------------|-------|----------------|
| ≥0.8 | High | Reliable scores, minimal uncertainty |
| 0.5-0.79 | Medium | Generally reliable, some uncertainty |
| <0.5 | Low | Significant uncertainty, manual review recommended |

### 9.5 Uncertainty Mitigation

- **Flag to User**: Display uncertainties in results
- **Suggest Improvements**: Provide guidance on reducing uncertainty
- **Adjust Score**: Apply penalty for high uncertainty
- **Recommend Manual Review**: For low confidence cases

---

## 10. Limitations

### 10.1 Technical Limitations

1. **Skill Detection Accuracy**: Depends on extraction quality; may miss rare tools
2. **Experience Quality**: Years of experience doesn't capture quality or relevance
3. **Education Field**: Only detects degree level, not field of study
4. **Semantic Limitations**: Embeddings don't reason about seniority or capability
5. **File Format Support**: Limited to PDF, DOCX, TXT

### 10.2 Design Limitations

1. **No Seniority Distinction**: Can't distinguish junior vs senior roles of same skill set
2. **Industry-Specific Knowledge**: Doesn't account for domain expertise (finance, healthcare)
3. **Soft Skills**: Doesn't evaluate communication, leadership, teamwork
4. **Cultural Fit**: Doesn't assess alignment with company culture
5. **Portfolio/Projects**: Limited project analysis

### 10.3 Data Limitations

1. **Resume Quality**: Assumes well-structured resumes; struggles with poor formatting
2. **JD Quality**: Assumes clear job descriptions; ambiguous JDs reduce accuracy
3. **Language**: Primarily optimized for English; other languages may have lower accuracy
4. **Geographic**: No consideration of location or timezone

### 10.4 Ethical Limitations

1. **Bias Risk**: Training data may contain biases
2. **Accessibility**: May disadvantage candidates with non-traditional backgrounds
3. **Privacy**: Resume data processed on external servers (if using cloud APIs)
4. **Fairness**: May perpetuate existing hiring biases

---

## 11. Future Improvements

### 11.1 Technical Enhancements

1. **Multi-Language Support**: Add support for non-English resumes and JDs
2. **Advanced File Formats**: Support for more file types (RTF, ODT, etc.)
3. **Image OCR**: Extract text from image-based PDFs
4. **Real-Time Analysis**: WebSocket support for real-time scoring updates
5. **Batch Processing**: Analyze multiple resumes against one JD

### 11.2 Scoring Enhancements

1. **Seniority Detection**: Distinguish between junior, mid, senior roles
2. **Industry Expertise**: Add domain-specific scoring (finance, healthcare, etc.)
3. **Soft Skills Evaluation**: Assess communication, leadership, teamwork
4. **Portfolio Analysis**: Evaluate GitHub, LinkedIn, or portfolio links
5. **Culture Fit**: Assess alignment with company values

### 11.3 UX Enhancements

1. **Resume Editor**: Built-in resume editor with real-time scoring
2. **Job Search Integration**: Import JDs from job boards
3. **Score History**: Track scoring history over time
4. **Comparison Mode**: Compare multiple resumes side-by-side
5. **Mobile App**: Native mobile applications

### 11.4 Infrastructure Enhancements

1. **Caching**: Cache embeddings for repeated analyses
2. **Queue System**: Handle high-volume requests with job queues
3. **Monitoring**: Add logging, metrics, and alerting
4. **Scalability**: Horizontal scaling with load balancing
5. **Security**: Add authentication, authorization, and audit logging

---

## 12. Sample Outputs

### 12.1 Sample API Response

```json
{
  "success": true,
  "overall_score": 75,
  "category_scores": {
    "required_skills": {
      "score": 0.8,
      "weight": 0.45,
      "matched": ["python", "django", "sql"],
      "missing": ["kubernetes"],
      "partial": ["aws"]
    },
    "experience": {
      "score": 0.9,
      "weight": 0.15,
      "matched": ["5.0 years"],
      "missing": [],
      "partial": []
    },
    "education": {
      "score": 1.0,
      "weight": 0.1,
      "matched": ["bachelor"],
      "missing": [],
      "partial": []
    },
    "responsibilities": {
      "score": 0.75,
      "weight": 0.2,
      "matched": [],
      "missing": [],
      "partial": []
    },
    "semantic_similarity": {
      "score": 0.7,
      "weight": 0.1,
      "matched": [],
      "missing": [],
      "partial": []
    }
  },
  "explanation": "Strong match (score: 75/100). Good alignment with 3 key strengths. Address 1 critical gap to improve competitiveness.",
  "suggestions": [
    "Add Kubernetes experience to strengthen cloud skills",
    "Highlight AWS projects in experience section"
  ],
  "strengths": [
    {"category": "skills", "description": "Strong Python and Django proficiency"},
    {"category": "experience", "description": "5 years of relevant experience"}
  ],
  "gaps": [
    {"category": "skills", "description": "Missing Kubernetes", "severity": "critical"}
  ],
  "confidence": "medium",
  "confidence_score": 0.75,
  "uncertainties": [
    {"type": "inferred_skill", "description": "AWS experience inferred from related technologies", "severity": "medium"}
  ]
}
```

### 12.2 Sample UI Display

**Overall Score**: 75/100 (Blue - Strong Match)

**Category Breakdown**:
- Skills: 80% (Matched: python, django, sql | Missing: kubernetes | Partial: aws)
- Experience: 90% (5.0 years)
- Education: 100% (Bachelor)
- Responsibilities: 75%
- Semantic: 70%

**Strengths**:
- Strong Python and Django proficiency
- 5 years of relevant experience

**Gaps**:
- Missing Kubernetes (Critical)

**Suggestions**:
- Add Kubernetes experience to strengthen cloud skills
- Highlight AWS projects in experience section

**Confidence**: Medium (75%)

---

## 13. Design Decisions

### 13.1 Hybrid AI Approach

**Decision**: Use both LLM and deterministic components

**Rationale**:
- LLM excels at parsing unstructured text and generating explanations
- Deterministic components ensure transparency and reproducibility
- Balances accuracy with cost-effectiveness
- Provides fallback options if one component fails

### 13.2 sentence-transformers vs OpenAI Embeddings

**Decision**: Use sentence-transformers for embeddings

**Rationale**:
- Deterministic output (same input → same output)
- No API costs
- Faster inference
- Sufficient quality for this use case
- Runs locally (no data privacy concerns)

### 13.3 Weight Distribution

**Decision**: Skills (45%), Responsibilities (20%), Experience (15%), Education (10%), Semantic (10%)

**Rationale**:
- Skills are most critical for technical roles
- Responsibilities provide context but are secondary
- Experience matters but is less critical than capability
- Education is often flexible with experience
- Semantic similarity acts as a tiebreaker

### 13.4 Confidence Thresholds

**Decision**: High (≥0.8), Medium (0.5-0.79), Low (<0.5)

**Rationale**:
- Provides clear guidance to users
- Balances sensitivity and specificity
- Aligns with common industry practices
- Easy to communicate to non-technical users

### 13.5 Dual Backend Implementation

**Decision**: Implement both FastAPI and Flask backends

**Rationale**:
- FastAPI: Modern, fast, async support
- Flask: Simpler, more mature, widely used
- Provides flexibility for different use cases
- Demonstrates architectural versatility

---

## 14. Why Hybrid AI Approach

### 14.1 Pure LLM Limitations

1. **Cost**: Every API call incurs cost; high-volume screening becomes expensive
2. **Speed**: LLM calls are slower than deterministic calculations
3. **Reproducibility**: LLMs may produce different outputs for the same input
4. **Transparency**: Black-box nature makes it difficult to explain decisions
5. **Privacy**: Sending resume data to external APIs raises privacy concerns

### 14.2 Pure Deterministic Limitations

1. **Flexibility**: Rule-based systems struggle with unstructured text
2. **Context**: Cannot understand context or nuance
3. **Natural Language**: Cannot generate human-readable explanations
4. **Adaptability**: Requires manual updates for new patterns
5. **Semantic Understanding**: Cannot capture meaning beyond keywords

### 14.3 Hybrid Approach Benefits

1. **Cost-Effectiveness**: Use LLM only where needed (parsing, explanation)
2. **Speed**: Deterministic components for fast scoring
3. **Transparency**: Core scoring is rule-based and explainable
4. **Flexibility**: LLM handles unstructured text and natural language
5. **Reproducibility**: Deterministic core ensures consistent results
6. **Best of Both Worlds**: Leverages strengths of both approaches

### 14.4 Component Allocation

| Component | Approach | Rationale |
|------------|----------|-----------|
| Parsing | LLM (primary), Deterministic (fallback) | LLM excels at understanding unstructured text |
| Skill Matching | Embeddings (deterministic) | Fast, semantic understanding, no API cost |
| Experience Scoring | Deterministic | Simple ratio calculation, transparent |
| Education Scoring | Deterministic | Level ladder comparison, transparent |
| Responsibility Scoring | TF-IDF (deterministic) | Fast, effective for text similarity |
| Semantic Similarity | Embeddings (deterministic) | Captures overall fit, no API cost |
| Explanation | LLM (primary), Deterministic (fallback) | LLM excels at natural language generation |
| Uncertainty | Deterministic | Rule-based detection, transparent |

### 14.5 Fallback Strategy

The system includes fallback mechanisms:
- If LLM parsing fails, use deterministic parser
- If LLM explanation fails, use deterministic explainer
- If embeddings fail, use keyword matching
- This ensures system resilience and reliability

---

## Conclusion

The Resume Scoring System represents a balanced approach to AI-assisted hiring that prioritizes transparency, explainability, and cost-effectiveness while maintaining high accuracy. By combining LLM capabilities with deterministic components, the system achieves the best of both worlds: the flexibility and natural language understanding of AI with the transparency and reproducibility of rule-based systems.

The modular architecture allows for easy extension and modification, making the system adaptable to different use cases and requirements. The comprehensive uncertainty handling ensures that users are aware of the limitations of the analysis, enabling informed decision-making.

This system is suitable for both academic study and practical deployment in real-world hiring scenarios, providing recruiters with a powerful tool to streamline the resume screening process while maintaining fairness and transparency.

---

## Appendix

### A. Project Structure

```
new_resume_scorer/
├── backend/                          # Core scoring modules
│   ├── models.py                     # Pydantic schemas
│   ├── config.py                     # Configuration
│   ├── parser.py                     # LLM-based parser
│   ├── embedder.py                   # Sentence-transformers embeddings
│   ├── scorer.py                     # Deterministic scoring
│   ├── combiner.py                   # Score combination
│   ├── explainer.py                  # LLM explainer
│   ├── main.py                       # FastAPI app
│   ├── deterministic_parser.py        # Rule-based parser
│   ├── skill_extractor.py            # Skill extraction engine
│   ├── semantic_similarity.py        # Semantic similarity module
│   ├── weighted_scorer.py            # Weighted scoring engine
│   ├── deterministic_explainer.py     # Evidence-based explainer
│   └── uncertainty_estimator.py      # Uncertainty estimation
├── flask_backend/                    # Flask API implementation
│   ├── app.py                        # Flask app factory
│   ├── routes/                       # API endpoints
│   ├── services/                     # Business logic
│   ├── utils/                        # Utility functions
│   ├── models/                       # Data schemas
│   ├── static/                       # Static files
│   └── templates/                    # HTML templates
├── modern_frontend/                   # Bootstrap UI
│   ├── index.html                    # Main HTML
│   ├── custom.css                    # Custom styling
│   ├── app.js                        # JavaScript
│   └── README.md                     # Frontend documentation
├── frontend/                         # Original frontend
│   ├── index.html
│   ├── style.css
│   └── app.js
├── requirements.txt                  # Python dependencies
├── ARCHITECTURE.md                   # Detailed architecture
├── README.md                         # Setup instructions
└── PROJECT_DOCUMENTATION.md          # This file
```

### B. Key Metrics

- **Lines of Code**: ~8,000+
- **Modules**: 15+
- **API Endpoints**: 6
- **Scoring Dimensions**: 5
- **Supported File Formats**: 3 (PDF, DOCX, TXT)
- **Average Response Time**: <2 seconds

### C. References

1. sentence-transformers Documentation: https://www.sbert.net/
2. OpenAI API Documentation: https://platform.openai.com/docs
3. FastAPI Documentation: https://fastapi.tiangolo.com/
4. Bootstrap Documentation: https://getbootstrap.com/docs/
5. Pydantic Documentation: https://docs.pydantic.dev/

---

**End of Documentation**
