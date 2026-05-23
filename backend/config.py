"""
Configuration settings for the Resume Scorer application.

This module contains all configuration settings including:
- API keys for external services (OpenAI, etc.)
- Model names and versions
- Scoring weights for different components
- File paths and directories
- Environment-specific settings

Settings are loaded from environment variables using python-dotenv.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))

# Embedding Model Configuration (sentence-transformers)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# Scoring Weights (should sum to 1.0)
SCORE_WEIGHTS = {
    "skills": 0.40,
    "experience": 0.20,
    "education": 0.15,
    "responsibilities": 0.15,
    "semantic": 0.10,
}

# File Upload Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_FILE_TYPES = [".pdf", ".txt", ".docx"]
UPLOAD_DIR = "uploads"

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_TITLE = "Resume Scorer API"
API_VERSION = "1.0.0"

# Parsing Configuration
MAX_RESUME_LENGTH = 10000  # Maximum characters to parse
MAX_JD_LENGTH = 5000  # Maximum characters to parse

# Scoring Thresholds
EXCELLENT_THRESHOLD = 80
STRONG_THRESHOLD = 65
MODERATE_THRESHOLD = 50
WEAK_THRESHOLD = 35

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
