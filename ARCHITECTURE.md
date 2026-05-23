# Architecture Documentation

## System Overview

The Resume Scoring System is a multi-component pipeline that analyzes how well a resume matches a job description. It outputs a 0-100 fit score with structured, explainable insights. The system is designed to be auditable, fast, and cost-effective while providing actionable feedback to job seekers.

## Architecture Diagram (Text)

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                           │
│                    (Frontend: HTML/CSS/JS)                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ HTTP POST /score
                         │ {resume_text, jd_text}
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Application                          │
│                          (main.py)                                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ 1. Parse Resume & JD (concurrent)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Parser (parser.py)                        │
│              - OpenAI GPT-4o-mini for structured extraction     │
│              - Extracts: skills, experience, education, etc.     │
│              - Collects uncertainty notes                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ 2. Compute Scores
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Embedding Similarity (embedder.py)                 │
│         - sentence-transformers all-MiniLM-L6-v2                 │
│         - Skills matching via semantic similarity                │
│         - Overall semantic similarity between texts              │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────────────────────┐
│           Deterministic Scorer (scorer.py)                       │
│         - Experience: ratio-based scoring                        │
│         - Education: level ladder comparison                     │
│         - Keywords: token matching                               │
│         - Responsibilities: TF-IDF cosine similarity              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ 3. Combine Scores
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Score Combiner (combiner.py)                        │
│         - Weighted sum of category scores                        │
│         - Confidence level assessment                            │
│         - Uncertainty flag collection                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ 4. Generate Explanation
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              LLM Explainer (explainer.py)                        │
│         - OpenAI GPT-4o-mini for natural language output         │
│         - Generates summary, suggestions, strength analysis     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ 5. Return ScoreResult
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Response to Frontend                        │
│         {overall_score, category_scores, explanation, ...}      │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Pydantic Models (models.py)

**Purpose**: Define data validation schemas and ensure type safety throughout the application.

**Models**:
- `EducationLevel`: Enum for education levels (high_school, bachelor, master, phd, unknown)
- `ConfidenceLevel`: Enum for confidence levels (high, medium, low)
- `ResumeData`: Structured resume information (skills, experience, education, etc.)
- `JDData`: Structured job description information (required skills, preferred skills, etc.)
- `CategoryScore`: Individual category scoring details with matched/missing/partial items
- `ScoreResult`: Complete scoring analysis result with overall score, explanation, suggestions
- `AnalysisRequest`: API request schema
- `AnalysisResponse`: API response schema

**Why this component exists**: Ensures data integrity, provides automatic validation, and serves as documentation for the data structures used throughout the system.

### 2. Configuration (config.py)

**Purpose**: Centralize all configuration settings including API keys, model names, scoring weights, and environment-specific settings.

**Settings**:
- OpenAI API configuration (model, temperature)
- Sentence-transformers model configuration (model name, device)
- Scoring weights (skills: 40%, experience: 20%, education: 15%, responsibilities: 15%, semantic: 10%)
- File upload settings
- API host and port
- Scoring thresholds

**Why this component exists**: Makes the system configurable without code changes, allows easy experimentation with different models and weights, and separates configuration from business logic.

### 3. LLM Parser (parser.py)

**Purpose**: Parse unstructured resume and job description text into structured data using OpenAI's API.

**Functions**:
- `parse_resume(text)`: Extracts candidate name, skills, job titles, years of experience, education level
- `parse_jd(text)`: Extracts job title, required skills, preferred skills, minimum experience, required education
- `get_uncertainty_notes()`: Returns collected uncertainty notes from parsing
- `clear_uncertainty_notes()`: Clears uncertainty notes for fresh analysis

**Why this component exists**: LLMs excel at understanding unstructured text and extracting structured information. Using structured output with JSON schema ensures the output matches our Pydantic models exactly.

**Why LLM is used here**: Parsing requires understanding context, inferring missing information (like years of experience from dates), and distinguishing between required vs preferred skills - tasks where LLMs perform well.

### 4. Embedding Similarity (embedder.py)

**Purpose**: Compute semantic similarity between resume and job description using sentence-transformers.

**Functions**:
- `embed_texts(texts)`: Generate embeddings for a list of texts using sentence-transformers
- `cosine_similarity_numpy(embedding1, embedding2)`: Compute cosine similarity between embeddings
- `compute_skills_score(resume, jd)`: Match skills using semantic similarity (matched >=0.85, partial 0.65-0.84, missing <0.65)
- `compute_semantic_score(resume, jd)`: Compute overall semantic similarity between full texts

**Why this component exists**: Pure keyword matching misses semantic relationships (e.g., "Python" vs "pandas", "machine learning" vs "ML"). Embeddings capture semantic meaning, allowing the system to recognize related concepts.

**Why sentence-transformers**: 
- Deterministic: Same input always produces same output (unlike some LLM embeddings)
- No API costs: Runs locally
- Fast: Efficient inference
- Explainable: Can inspect similarity scores
- Sufficient quality: all-MiniLM-L6-v2 provides good semantic understanding for this use case

**Why embeddings are used**: To capture semantic overlap beyond exact keyword matching, enabling the system to recognize related skills and concepts that a pure keyword matcher would miss.

### 5. Deterministic Scorer (scorer.py)

**Purpose**: Compute category-specific scores using deterministic rules for transparency and reproducibility.

**Functions**:
- `score_experience(resume, jd)`: Ratio-based scoring (actual_years / required_years)
- `score_education(resume, jd)`: Level ladder comparison (unknown=0, high_school=1, bachelor=2, master=3, phd=4)
- `score_keywords(resume, jd)`: Token matching with unigrams and bigrams
- `score_responsibilities(resume, jd)`: TF-IDF cosine similarity between full texts

**Why this component exists**: Deterministic rules are transparent, auditable, and fast. Users can understand exactly how points were awarded. No black-box LLM calls for core scoring logic.

**Why deterministic logic is used**: 
- Auditable: Users can see exactly what was detected and why
- Fast: No API calls, no model loading
- Consistent: Same input always produces same output
- Debuggable: Easy to fix issues with skill detection
- Cost-effective: No API costs for scoring

### 6. Score Combiner (combiner.py)

**Purpose**: Combine individual category scores into a final weighted score with confidence assessment.

**Functions**:
- `combine_scores(category_scores, uncertainty_notes, resume)`: Calculates weighted sum, determines confidence, collects uncertainty flags
- `_determine_confidence(uncertainty_notes, resume)`: Determines confidence level based on inference indicators
- `_collect_uncertainty_flags(uncertainty_notes, resume)`: Collects uncertainty flags from notes and adds rule-based ones

**Why this component exists**: Provides a single overall score while maintaining transparency about component performance. Confidence assessment helps users understand when to trust the score.

**How uncertainty is estimated**: 
- Parser uncertainty notes (e.g., "inferred years of experience from job count")
- Rule-based flags (e.g., "education level not found", "very few skills detected")
- Resume quality indicators (e.g., text too short)
- Confidence level determined by whether key fields were inferred vs explicit

### 7. LLM Explainer (explainer.py)

**Purpose**: Generate human-readable explanations of scoring results using LLM.

**Functions**:
- `generate_explanation(resume, jd, score_data)`: Generates summary, suggestions, and strength analysis

**Why this component exists**: While scoring is deterministic, the explanation benefits from LLM's natural language generation capabilities to provide clear, actionable feedback.

**Why LLM is used here**: Explanation generation requires natural language fluency, the ability to synthesize information from multiple sources, and the capacity to provide constructive feedback - tasks where LLMs excel.

**How explainability is achieved**:
- Category breakdown shows individual component scores with matched/missing/partial items
- Explanation provides narrative summary of overall fit
- Suggestions offer specific, actionable improvements
- Uncertainty flags highlight assumptions made by the system
- Confidence level indicates reliability of the analysis

### 8. FastAPI Application (main.py)

**Purpose**: Provide REST API endpoints for resume analysis.

**Endpoints**:
- `GET /`: Root endpoint returning API information
- `GET /health`: Health check endpoint
- `POST /score`: Score a resume against a job description

**Pipeline**:
1. Parse resume and JD concurrently using asyncio.gather
2. Compute skills score using embeddings
3. Compute experience, education, keywords, responsibilities, and semantic scores
4. Combine scores with uncertainty notes
5. Generate explanation using LLM
6. Build and return ScoreResult

**Why this component exists**: Provides a clean API interface for the frontend, handles concurrent operations efficiently, and manages error handling gracefully.

### 9. Frontend (frontend/)

**Purpose**: User-facing interface for submitting resumes and job descriptions, viewing results.

**Files**:
- `index.html`: Single-page UI structure
- `style.css`: Modern, responsive styling
- `app.js`: API integration and result rendering

**Why this component exists**: Provides an intuitive interface for users to interact with the system without needing to make API calls directly.

## Data Flow

1. **User Input**: User pastes resume text and job description into the frontend
2. **API Request**: Frontend sends POST request to `/score` endpoint with resume_text and jd_text
3. **Parsing**: Backend concurrently parses resume and JD using LLM, collecting uncertainty notes
4. **Scoring**: Backend computes scores from multiple components:
   - Skills matching via sentence-transformers embeddings
   - Experience via deterministic ratio calculation
   - Education via level ladder comparison
   - Keywords via token matching
   - Responsibilities via TF-IDF similarity
   - Semantic similarity via sentence-transformers
5. **Combination**: Backend combines weighted scores, determines confidence, collects uncertainty flags
6. **Explanation**: Backend generates natural language explanation using LLM
7. **Response**: Backend returns ScoreResult with overall score, category breakdown, explanation, suggestions
8. **Display**: Frontend renders results with visual score display, category breakdown, and suggestions

## Technology Choices

### Python
- **Why**: Excellent ecosystem for ML/AI, strong typing support with Pydantic, async support with FastAPI
- **Alternatives considered**: JavaScript/TypeScript (less mature ML ecosystem), Java (more verbose)

### FastAPI
- **Why**: Modern, fast, automatic API documentation, async support, Pydantic integration
- **Alternatives considered**: Flask (simpler but less features), Django REST Framework (heavier)

### sentence-transformers
- **Why**: Deterministic, no API costs, fast inference, good quality for semantic similarity
- **Alternatives considered**: OpenAI embeddings (API costs, less deterministic), spaCy (less semantic understanding)

### OpenAI GPT-4o-mini
- **Why**: Excellent for parsing and explanation, cost-effective, structured output support
- **Alternatives considered**: Claude (similar but different API), local LLMs (slower, less capable)

### scikit-learn
- **Why**: Mature, reliable TF-IDF implementation, widely used
- **Alternatives considered**: Custom TF-IDF (more work, less tested)

### Pydantic
- **Why**: Automatic validation, type safety, excellent documentation
- **Alternatives considered**: Plain Python dictionaries (no validation), dataclasses (less validation)

## Design Decisions

### Why Multi-Component Architecture?
- **Separation of concerns**: Each component has a single responsibility
- **Testability**: Components can be tested independently
- **Maintainability**: Changes to one component don't affect others
- **Explainability**: Users can trace how the score was derived
- **Flexibility**: Easy to swap out individual components (e.g., different embedding model)

### Why Not a Single LLM Call?
- **Transparency**: Multi-component approach shows exactly how scores were calculated
- **Cost**: LLM calls are expensive; deterministic components are free
- **Speed**: Embeddings and deterministic scoring are faster than LLM
- **Reliability**: Deterministic components always produce the same output
- **Auditability**: Users can inspect each component's contribution

### Why Embeddings Instead of Pure Keyword Matching?
- **Semantic understanding**: Captures related concepts (e.g., "Python" vs "pandas")
- **Phrase matching**: N-grams capture multi-word terms (e.g., "machine learning")
- **Flexibility**: Handles synonyms and related terms
- **Quality**: Better matches than simple string matching

### Why Deterministic Scoring for Core Logic?
- **Auditable**: Users can see exactly what was detected
- **Fast**: No API calls, no model loading
- **Consistent**: Same input always produces same output
- **Debuggable**: Easy to fix issues with skill detection
- **Cost-effective**: No API costs for scoring

### Why LLM for Parsing and Explanation?
- **Parsing**: Requires understanding context, inferring missing information, distinguishing requirements
- **Explanation**: Requires natural language fluency, synthesis of information, constructive feedback
- **Quality**: LLMs excel at these tasks compared to rule-based approaches

### How Uncertainty is Estimated?
- **Parser uncertainty**: LLM notes when it infers values (e.g., "estimated years from job count")
- **Rule-based flags**: Detect missing data (e.g., "education not found", "very few skills")
- **Resume quality**: Flag short or sparse resumes
- **Confidence levels**: High (explicit data), Medium (some inference), Low (heavy inference or poor quality)

### How Explainability is Achieved?
- **Category breakdown**: Shows individual component scores with matched/missing/partial items
- **Evidence snippets**: Shows which skills were matched and why
- **Natural language explanation**: LLM provides clear summary and suggestions
- **Uncertainty flags**: Highlights assumptions and limitations
- **Confidence indicator**: Shows reliability of the analysis

## Score Weights Justification

- **Skills (40%)**: Most important for technical roles - direct capability indicator
- **Experience (20%)**: Seniority match - important but less critical than actual skills
- **Education (15%)**: Baseline requirement - often flexible with experience
- **Responsibilities (15%)**: Contextual fit - shows relevant experience
- **Semantic (10%)**: Overall fit - catches things skill lists miss

## Limitations

1. **Skill detection is LLM-based**: May miss rare tools or unusual wording
2. **Embeddings don't reason about seniority**: Can't distinguish junior vs senior
3. **Education detection is basic**: Only detects degree level, not field of study
4. **Experience extraction is LLM-based**: May miss non-standard phrasing
5. **TF-IDF doesn't capture deep semantics**: Limited compared to more advanced methods

## Future Enhancements

1. **Add file upload support**: Parse PDF/DOCX resumes directly
2. **Improve parsing**: Add named entity recognition for companies/roles
3. **Better education detection**: Detect field of study and institution
4. **More sophisticated experience extraction**: Handle non-standard formats
5. **Add more test cases**: Different industries, seniority levels, international resumes
6. **Improve UI**: Real-time scoring preview, drag-and-drop upload, score history
7. **Add caching**: Cache embeddings for repeated analyses
8. **Batch analysis**: Support analyzing multiple resumes against one JD
