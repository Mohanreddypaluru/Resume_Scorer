"""
Embedding similarity calculator using sentence-transformers.

This module computes semantic similarity between resume and job description
using sentence-transformers models and cosine similarity. It provides
a non-LLM approach to measuring how well content matches beyond simple keyword
matching.

IMPORTANT: Embeddings are deterministic given the same model version. For the
same input text and model version, this component will always produce identical
embeddings and therefore identical similarity scores. This determinism ensures
reproducible results at inference time.

The module uses numpy for all mathematical operations to maintain speed and
determinism.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, EMBEDDING_DEVICE
from models import ResumeData, JDData, CategoryScore


# Initialize sentence-transformers model (cached globally)
_model = None


def get_model():
    """
    Lazy-load the sentence-transformers model.
    
    Returns:
        SentenceTransformer model instance
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Generate embeddings for a list of texts using sentence-transformers.
    
    This function makes a single batch call to embed all texts efficiently.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        numpy array of shape (N, embedding_dim) where N is the number of texts
        
    The function:
    1. Loads the sentence-transformers model (cached)
    2. Encodes all texts in a single batch
    3. Returns as numpy array for efficient mathematical operations
    """
    if not texts:
        model = get_model()
        return np.array([]).reshape(0, model.get_sentence_embedding_dimension())
    
    try:
        model = get_model()
        embeddings = model.encode(texts, show_progress_bar=False)
        return np.array(embeddings)
        
    except Exception as e:
        raise ValueError(f"Failed to generate embeddings: {e}")


def cosine_similarity_numpy(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embedding vectors using numpy.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine similarity score between -1 and 1 (typically 0 to 1 for text embeddings)
    """
    # Compute dot product
    dot_product = np.dot(embedding1, embedding2)
    
    # Compute norms
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    # Avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def compute_skills_score(resume: ResumeData, jd: JDData) -> CategoryScore:
    """
    Compute skills matching score using embedding-based semantic similarity.
    
    This function:
    1. Embeds all resume skills and JD required skills
    2. For each required skill, finds the best matching resume skill
    3. Classifies matches as matched (>=0.85), partial (0.65-0.84), or missing (<0.65)
    4. Computes a weighted score based on match quality
    
    Args:
        resume: Parsed resume data containing skills
        jd: Parsed job description data containing required skills
        
    Returns:
        CategoryScore with name="skills_match", score between 0-1, weight=0.40,
        and lists of matched, missing, and partial skills
    """
    if not jd.required_skills:
        # No required skills specified - return neutral score
        return CategoryScore(
            name="skills_match",
            score=0.5,
            weight=0.40,
            matched=[],
            missing=[],
            partial=[]
        )
    
    if not resume.skills:
        # Resume has no skills - all required skills are missing
        return CategoryScore(
            name="skills_match",
            score=0.0,
            weight=0.40,
            matched=[],
            missing=jd.required_skills.copy(),
            partial=[]
        )
    
    # Embed all skills
    all_texts = resume.skills + jd.required_skills
    embeddings = embed_texts(all_texts)
    
    # Split embeddings back into resume and JD embeddings
    resume_embeddings = embeddings[:len(resume.skills)]
    jd_embeddings = embeddings[len(resume.skills):]
    
    # For each required skill, find best match in resume skills
    matched = []
    missing = []
    partial = []
    
    for jd_idx, required_skill in enumerate(jd.required_skills):
        jd_embedding = jd_embeddings[jd_idx]
        
        # Find best matching resume skill
        best_similarity = 0.0
        best_match_idx = -1
        
        for resume_idx, resume_embedding in enumerate(resume_embeddings):
            similarity = cosine_similarity_numpy(jd_embedding, resume_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = resume_idx
        
        # Classify based on similarity thresholds
        if best_similarity >= 0.85:
            matched.append(required_skill)
        elif best_similarity >= 0.65:
            partial.append(required_skill)
        else:
            missing.append(required_skill)
    
    # Compute score: (matched_count + 0.5 * partial_count) / total_required
    matched_count = len(matched)
    partial_count = len(partial)
    total_required = len(jd.required_skills)
    
    score = (matched_count + 0.5 * partial_count) / max(total_required, 1)
    score = min(score, 1.0)  # Ensure score doesn't exceed 1.0
    
    return CategoryScore(
        name="skills_match",
        score=score,
        weight=0.40,
        matched=matched,
        missing=missing,
        partial=partial
    )


def compute_semantic_score(resume: ResumeData, jd: JDData) -> CategoryScore:
    """
    Compute overall semantic similarity between resume and job description.
    
    This function:
    1. Embeds the full resume text and full JD text
    2. Computes cosine similarity between the two embeddings
    3. Returns a CategoryScore with the similarity as the score
    
    Args:
        resume: Parsed resume data containing raw_text
        jd: Parsed job description data containing raw_text
        
    Returns:
        CategoryScore with name="semantic", score between 0-1, weight=0.10
    """
    try:
        # Embed full texts
        embeddings = embed_texts([resume.raw_text, jd.raw_text])
        
        if embeddings.shape[0] < 2:
            return CategoryScore(
                name="semantic",
                score=0.5,
                weight=0.10,
                matched=[],
                missing=[],
                partial=[]
            )
        
        # Compute similarity
        similarity = cosine_similarity_numpy(embeddings[0], embeddings[1])
        
        # Ensure similarity is in [0, 1] range
        similarity = max(0.0, min(1.0, similarity))
        
        return CategoryScore(
            name="semantic",
            score=similarity,
            weight=0.10,
            matched=[],
            missing=[],
            partial=[]
        )
        
    except Exception as e:
        # Fallback to neutral score on error
        return CategoryScore(
            name="semantic",
            score=0.5,
            weight=0.10,
            matched=[],
            missing=[],
            partial=[]
        )
