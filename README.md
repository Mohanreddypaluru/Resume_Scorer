# Resume Scoring System

A modern resume scoring system that analyzes how well a resume matches a job description using a multi-component approach combining LLM-based parsing, deterministic rule-based scoring, and non-LLM semantic similarity with sentence-transformers.

## Features

- **Multi-component pipeline**: Not a single LLM call - combines multiple specialized components
- **Deterministic scoring**: Rule-based scoring for transparency and reproducibility
- **Semantic similarity**: Uses sentence-transformers for skill matching beyond keywords
- **Explainable results**: Shows matched skills, missing skills, partial matches, and uncertainty flags
- **LLM-powered explanations**: Natural language summaries and actionable suggestions
- **Confidence assessment**: Indicates reliability of the analysis
- **Modern UI**: Clean, responsive web interface

## Architecture

The system is designed with a clear separation of concerns:

- **LLM Components**: Parser and Explainer use OpenAI API for intelligent text processing and explanation generation
- **Non-LLM Components**: Embedder, Scorer, and Combiner use deterministic methods for fast, transparent scoring
- **Frontend**: Clean, minimal single-page application for user interaction

For detailed architecture documentation, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Project Structure

```
new_resume_scorer/
├── backend/
│   ├── main.py           # FastAPI app entry point
│   ├── parser.py         # LLM-based resume + JD parser
│   ├── embedder.py       # Embedding similarity (sentence-transformers)
│   ├── scorer.py         # Deterministic rule-based scorer
│   ├── combiner.py       # Weighted score combiner
│   ├── explainer.py      # LLM explanation generator
│   ├── models.py         # Pydantic schemas
│   └── config.py         # API keys, model names, weights config
├── frontend/
│   ├── index.html        # Single-page UI
│   ├── app.js            # Fetch + render logic
│   └── style.css         # Clean minimal styles
├── requirements.txt
├── ARCHITECTURE.md       # Detailed architecture documentation
└── README.md
```

## Technology Stack

- **Backend**: Python 3.11+, FastAPI, Pydantic v2
- **LLM**: OpenAI GPT-4o-mini for parsing and explanation
- **Embeddings**: sentence-transformers all-MiniLM-L6-v2 for semantic similarity
- **Scoring**: scikit-learn, numpy for deterministic calculations
- **Frontend**: Vanilla HTML, CSS, JavaScript (no frameworks)

## Installation

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- OpenAI API key

### Setup Steps

1. **Clone or navigate to the project directory**:
```bash
cd c:\Users\Admin\Desktop\new_resume_scorer
```

2. **Create a virtual environment**:
```bash
python -m venv venv
```

3. **Activate the virtual environment**:

On Windows:
```bash
venv\Scripts\activate
```

On macOS/Linux:
```bash
source venv/bin/activate
```

4. **Install dependencies**:
```bash
pip install -r requirements.txt
```

5. **Configure environment variables**:

Create a `.env` file in the project root:
```bash
# .env file
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.3
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
API_HOST=0.0.0.0
API_PORT=8000
```

Replace `your_openai_api_key_here` with your actual OpenAI API key.

## Usage

### Start the Backend Server

```bash
cd backend
python main.py
```

Or using uvicorn directly:
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

API documentation (Swagger UI) will be available at `http://localhost:8000/docs`

### Open the Frontend

Open `frontend/index.html` in your web browser, or serve it with a simple HTTP server:

```bash
cd frontend
python -m http.server 3000
```

Then navigate to `http://localhost:3000`

## API Endpoints

### POST /score

Analyze a resume against a job description.

**Request Body**:
```json
{
  "resume_text": "Your resume text here...",
  "jd_text": "Job description text here..."
}
```

**Response**:
```json
{
  "overall_score": 75,
  "category_scores": [
    {
      "name": "skills_match",
      "score": 0.8,
      "weight": 0.4,
      "matched": ["Python", "SQL"],
      "missing": ["Docker"],
      "partial": ["AWS"]
    },
    ...
  ],
  "explanation": "Your resume scored 75/100...",
  "suggestions": ["Add Docker experience", "Highlight AWS projects"],
  "uncertainty_flags": ["Inferred years of experience"],
  "confidence": "medium"
}
```

### GET /

Root endpoint returning API information.

### GET /health

Health check endpoint.

## Scoring Components

The system uses weighted scoring across multiple dimensions:

1. **Skills (40%)**: Skill overlap and matching using sentence-transformer embeddings
2. **Experience (20%)**: Years of experience comparison
3. **Education (15%)**: Education level matching
4. **Keywords (15%)**: Keyword alignment using token matching
5. **Semantic (10%)**: Overall semantic similarity using sentence-transformers

## How It Works

1. **Parsing**: LLM parses resume and job description into structured data
2. **Scoring**: Multiple components compute scores deterministically
3. **Combination**: Scores are combined with weighted averaging
4. **Explanation**: LLM generates natural language explanation and suggestions
5. **Display**: Frontend shows results with visual breakdown

## Key Design Principles

- **Not a single LLM call**: Multi-component approach for transparency
- **Deterministic scoring**: Core scoring logic is rule-based and reproducible
- **Semantic understanding**: Embeddings capture meaning beyond keywords
- **Explainable**: Users can see exactly how scores were calculated
- **Cost-effective**: Embeddings and deterministic scoring have no API costs

## Configuration

### Scoring Weights

Adjust scoring weights in `backend/config.py`:

```python
SCORE_WEIGHTS = {
    "skills": 0.40,
    "experience": 0.20,
    "education": 0.15,
    "responsibilities": 0.15,
    "semantic": 0.10,
}
```

### Model Selection

Change models in `backend/config.py`:

```python
OPENAI_MODEL = "gpt-4o-mini"  # For parsing and explanation
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # For semantic similarity
```

## Development

### Running Tests

The project structure supports testing. Add test files in a `tests/` directory and run:

```bash
pytest
```

### Adding New Scoring Components

1. Create a new function in `backend/scorer.py` that returns a `CategoryScore`
2. Add the score to the pipeline in `backend/main.py`
3. Update the weight configuration in `backend/config.py`
4. Ensure weights sum to 1.0

## Troubleshooting

### Common Issues

**Issue**: Module not found errors
- **Solution**: Ensure virtual environment is activated and dependencies are installed

**Issue**: OpenAI API errors
- **Solution**: Check that OPENAI_API_KEY is set correctly in `.env` file

**Issue**: Embedding model download fails
- **Solution**: Ensure internet connection for first download; model is cached locally after first use

**Issue**: CORS errors in frontend
- **Solution**: Backend allows all origins for local development; ensure backend is running on port 8000

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For detailed architecture information, see [ARCHITECTURE.md](ARCHITECTURE.md).
