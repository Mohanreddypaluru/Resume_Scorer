"""
Semantic Similarity Module using Sentence-Transformers.

This module provides semantic similarity computation between resumes and job descriptions
using sentence-transformers embeddings. It goes beyond simple keyword matching by
capturing semantic meaning and context.

Why Embeddings Help Beyond Keyword Matching:
-------------------------------------------
1. **Semantic Understanding**: Embeddings capture meaning, not just exact words.
   - "Python developer" and "Software engineer with Python experience" are semantically similar
   - Keyword matching would miss this connection

2. **Synonym Recognition**: Related terms are mapped to similar vector spaces.
   - "ML", "Machine Learning", and "artificial intelligence" have similar embeddings
   - Keyword matching requires explicit synonym lists

3. **Context Awareness**: Embeddings understand words in context.
   - "bank" (financial) vs "bank" (river) have different embeddings based on context
   - Keyword matching cannot distinguish context

4. **Phrase-Level Matching**: Captures multi-word concepts.
   - "machine learning" is treated as a single concept, not "machine" + "learning"
   - Keyword matching with n-grams is less flexible

5. **Language Agnostic**: Works across different phrasings of the same concept.
   - "built APIs" and "API development" are semantically similar
   - Keyword matching requires exact string matches

Model: all-MiniLM-L6-v2
- Lightweight (384 dimensions)
- Fast inference
- Good quality for semantic similarity
- No API costs (runs locally)
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity


class SemanticSimilarityEngine:
    """
    Semantic similarity engine using sentence-transformers.
    
    This class provides reusable utility functions for computing semantic
    similarity between texts using the all-MiniLM-L6-v2 model.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Initialize the semantic similarity engine.
        
        Args:
            model_name: Name of the sentence-transformers model
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """
        Load the sentence-transformers model (lazy loading).
        """
        if self.model is None:
            print(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into embeddings.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([]).reshape(0, self.model.get_sentence_embedding_dimension())
        
        # Encode all texts in a single batch for efficiency
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for better cosine similarity
        )
        
        return embeddings
    
    def compute_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        # Use sklearn's cosine_similarity
        similarity = cosine_similarity(
            embedding1.reshape(1, -1),
            embedding2.reshape(1, -1)
        )[0][0]
        
        return float(similarity)
    
    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix for multiple embeddings.
        
        Args:
            embeddings: numpy array of shape (n_texts, embedding_dim)
            
        Returns:
            numpy array of shape (n_texts, n_texts) with similarity scores
        """
        return cosine_similarity(embeddings)
    
    def compare_texts(self, text1: str, text2: str) -> float:
        """
        Compare two texts and return their semantic similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        embeddings = self.encode_texts([text1, text2])
        return self.compute_cosine_similarity(embeddings[0], embeddings[1])
    
    def find_most_similar(self, query: str, candidates: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find the most similar texts to a query from a list of candidates.
        
        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of top results to return
            
        Returns:
            List of (text, similarity_score) tuples, sorted by similarity
        """
        # Encode query and all candidates
        query_embedding = self.encode_texts([query])[0]
        candidate_embeddings = self.encode_texts(candidates)
        
        # Compute similarities
        similarities = []
        for i, candidate in enumerate(candidates):
            similarity = self.compute_cosine_similarity(query_embedding, candidate_embeddings[i])
            similarities.append((candidate, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


class ResumeJDSimilarityAnalyzer:
    """
    Specialized analyzer for comparing resumes and job descriptions.
    
    This class provides methods to compare different sections of resumes
    and job descriptions using semantic similarity.
    """
    
    def __init__(self, similarity_engine: SemanticSimilarityEngine = None):
        """
        Initialize the analyzer.
        
        Args:
            similarity_engine: SemanticSimilarityEngine instance (creates default if None)
        """
        self.engine = similarity_engine or SemanticSimilarityEngine()
    
    def compare_full_documents(self, resume_text: str, jd_text: str) -> Dict:
        """
        Compare full resume text with full job description text.
        
        Args:
            resume_text: Full resume text
            jd_text: Full job description text
            
        Returns:
            Dictionary with similarity score and details
        """
        similarity = self.engine.compare_texts(resume_text, jd_text)
        
        return {
            "comparison_type": "full_documents",
            "similarity_score": similarity,
            "interpretation": self._interpret_score(similarity)
        }
    
    def compare_responsibilities(self, resume_resp: List[str], jd_resp: List[str]) -> Dict:
        """
        Compare resume responsibilities with job description responsibilities.
        
        Args:
            resume_resp: List of responsibility strings from resume
            jd_resp: List of responsibility strings from job description
            
        Returns:
            Dictionary with similarity scores and best matches
        """
        if not resume_resp or not jd_resp:
            return {
                "comparison_type": "responsibilities",
                "average_similarity": 0.0,
                "best_match": None,
                "best_match_score": 0.0,
                "interpretation": "No responsibilities to compare"
            }
        
        # Compute similarity matrix
        all_texts = resume_resp + jd_resp
        embeddings = self.engine.encode_texts(all_texts)
        similarity_matrix = self.engine.compute_similarity_matrix(embeddings)
        
        # Extract resume-JD similarities (upper-right quadrant of matrix)
        n_resume = len(resume_resp)
        n_jd = len(jd_resp)
        cross_similarities = similarity_matrix[:n_resume, n_resume:n_resume+n_jd]
        
        # Calculate average similarity
        average_similarity = float(np.mean(cross_similarities))
        
        # Find best match
        max_idx = np.unravel_index(np.argmax(cross_similarities), cross_similarities.shape)
        best_resume_resp = resume_resp[max_idx[0]]
        best_jd_resp = jd_resp[max_idx[1]]
        best_match_score = float(cross_similarities[max_idx])
        
        return {
            "comparison_type": "responsibilities",
            "average_similarity": average_similarity,
            "best_match": {
                "resume": best_resume_resp,
                "jd": best_jd_resp,
                "score": best_match_score
            },
            "interpretation": self._interpret_score(average_similarity)
        }
    
    def compare_projects(self, resume_projects: List[str], jd_projects: List[str]) -> Dict:
        """
        Compare resume project descriptions with job description project requirements.
        
        Args:
            resume_projects: List of project description strings from resume
            jd_projects: List of project requirement strings from job description
            
        Returns:
            Dictionary with similarity scores and best matches
        """
        if not resume_projects or not jd_projects:
            return {
                "comparison_type": "projects",
                "average_similarity": 0.0,
                "best_match": None,
                "best_match_score": 0.0,
                "interpretation": "No projects to compare"
            }
        
        # Similar logic to responsibilities comparison
        all_texts = resume_projects + jd_projects
        embeddings = self.engine.encode_texts(all_texts)
        similarity_matrix = self.engine.compute_similarity_matrix(embeddings)
        
        n_resume = len(resume_projects)
        n_jd = len(jd_projects)
        cross_similarities = similarity_matrix[:n_resume, n_resume:n_resume+n_jd]
        
        average_similarity = float(np.mean(cross_similarities))
        
        max_idx = np.unravel_index(np.argmax(cross_similarities), cross_similarities.shape)
        best_resume_proj = resume_projects[max_idx[0]]
        best_jd_proj = jd_projects[max_idx[1]]
        best_match_score = float(cross_similarities[max_idx])
        
        return {
            "comparison_type": "projects",
            "average_similarity": average_similarity,
            "best_match": {
                "resume": best_resume_proj,
                "jd": best_jd_proj,
                "score": best_match_score
            },
            "interpretation": self._interpret_score(average_similarity)
        }
    
    def compare_adjacent_skills(self, resume_skills: List[str], jd_skills: List[str]) -> Dict:
        """
        Compare resume skills with job description skills using semantic similarity.
        
        This finds skills that are semantically related even if they don't match exactly.
        For example, "pandas" and "numpy" are adjacent skills in data science.
        
        Args:
            resume_skills: List of skills from resume
            jd_skills: List of required skills from job description
            
        Returns:
            Dictionary with matched, partial, and missing skills
        """
        if not jd_skills:
            return {
                "comparison_type": "adjacent_skills",
                "matched": [],
                "partial": [],
                "missing": [],
                "adjacent_pairs": []
            }
        
        if not resume_skills:
            return {
                "comparison_type": "adjacent_skills",
                "matched": [],
                "partial": [],
                "missing": jd_skills,
                "adjacent_pairs": []
            }
        
        # Encode all skills
        all_skills = resume_skills + jd_skills
        embeddings = self.engine.encode_texts(all_skills)
        similarity_matrix = self.engine.compute_similarity_matrix(embeddings)
        
        n_resume = len(resume_skills)
        n_jd = len(jd_skills)
        cross_similarities = similarity_matrix[:n_resume, n_resume:n_resume+n_jd]
        
        # Classify matches based on similarity thresholds
        matched = []
        partial = []
        missing = []
        adjacent_pairs = []
        
        for jd_idx, jd_skill in enumerate(jd_skills):
            # Find best matching resume skill
            best_similarity = 0.0
            best_resume_idx = -1
            
            for resume_idx in range(n_resume):
                similarity = cross_similarities[resume_idx][jd_idx]
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_resume_idx = resume_idx
            
            # Classify based on thresholds
            if best_similarity >= 0.85:
                matched.append(jd_skill)
                adjacent_pairs.append({
                    "jd_skill": jd_skill,
                    "resume_skill": resume_skills[best_resume_idx],
                    "similarity": float(best_similarity),
                    "match_type": "exact"
                })
            elif best_similarity >= 0.65:
                partial.append(jd_skill)
                adjacent_pairs.append({
                    "jd_skill": jd_skill,
                    "resume_skill": resume_skills[best_resume_idx],
                    "similarity": float(best_similarity),
                    "match_type": "partial"
                })
            else:
                missing.append(jd_skill)
        
        return {
            "comparison_type": "adjacent_skills",
            "matched": matched,
            "partial": partial,
            "missing": missing,
            "adjacent_pairs": adjacent_pairs
        }
    
    def comprehensive_analysis(self, resume_data: Dict, jd_data: Dict) -> Dict:
        """
        Perform comprehensive semantic similarity analysis between resume and JD.
        
        Args:
            resume_data: Dictionary containing resume sections (full_text, responsibilities, projects, skills)
            jd_data: Dictionary containing JD sections (full_text, responsibilities, projects, skills)
            
        Returns:
            Dictionary with all comparison results
        """
        results = {
            "full_document": self.compare_full_documents(
                resume_data.get("full_text", ""),
                jd_data.get("full_text", "")
            ),
            "responsibilities": self.compare_responsibilities(
                resume_data.get("responsibilities", []),
                jd_data.get("responsibilities", [])
            ),
            "projects": self.compare_projects(
                resume_data.get("projects", []),
                jd_data.get("projects", [])
            ),
            "adjacent_skills": self.compare_adjacent_skills(
                resume_data.get("skills", []),
                jd_data.get("skills", [])
            )
        }
        
        # Calculate overall semantic score
        weights = {
            "full_document": 0.3,
            "responsibilities": 0.3,
            "projects": 0.2,
            "adjacent_skills": 0.2
        }
        
        overall_score = (
            weights["full_document"] * results["full_document"]["similarity_score"] +
            weights["responsibilities"] * results["responsibilities"]["average_similarity"] +
            weights["projects"] * results["projects"]["average_similarity"] +
            weights["adjacent_skills"] * (len(results["adjacent_skills"]["matched"]) + 
                                          0.5 * len(results["adjacent_skills"]["partial"])) / 
                                          max(len(jd_data.get("skills", [])), 1)
        )
        
        results["overall_semantic_score"] = float(overall_score)
        results["overall_interpretation"] = self._interpret_score(overall_score)
        
        return results
    
    def _interpret_score(self, score: float) -> str:
        """
        Interpret a similarity score with a human-readable description.
        
        Args:
            score: Similarity score between 0 and 1
            
        Returns:
            Human-readable interpretation
        """
        if score >= 0.85:
            return "Excellent match - very high semantic similarity"
        elif score >= 0.70:
            return "Strong match - high semantic similarity"
        elif score >= 0.55:
            return "Moderate match - some semantic overlap"
        elif score >= 0.40:
            return "Weak match - limited semantic similarity"
        else:
            return "Poor match - little to no semantic similarity"


# ============================================================================
# SAMPLE USAGE AND OUTPUT
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SEMANTIC SIMILARITY ENGINE - SAMPLE USAGE")
    print("=" * 70)
    
    # Initialize the engine
    engine = SemanticSimilarityEngine()
    
    # Sample resume data
    resume_data = {
        "full_text": """
        Senior Software Engineer with 5 years of experience in building scalable
        web applications. Proficient in Python, Django, and cloud technologies.
        Led development of microservices architecture handling 1M+ daily users.
        """,
        "responsibilities": [
            "Designed and implemented RESTful APIs using Python and Django",
            "Led a team of 5 developers in building microservices",
            "Optimized database queries reducing response time by 40%",
            "Implemented CI/CD pipelines using Jenkins and Docker"
        ],
        "projects": [
            "Built an e-commerce platform using Django and React",
            "Developed a data processing pipeline using Apache Spark",
            "Created a real-time analytics dashboard with WebSocket support"
        ],
        "skills": ["python", "django", "react", "docker", "kubernetes", "aws", "sql"]
    }
    
    # Sample job description data
    jd_data = {
        "full_text": """
        We are looking for a Senior Backend Engineer to join our team.
        The ideal candidate has experience with Python web frameworks,
        cloud platforms, and building scalable systems.
        """,
        "responsibilities": [
            "Design scalable backend systems using Python",
            "Develop RESTful APIs and microservices",
            "Collaborate with cross-functional teams",
            "Optimize application performance and database queries"
        ],
        "projects": [
            "Experience building large-scale web applications",
            "Background in data processing and analytics"
        ],
        "skills": ["python", "flask", "aws", "microservices", "sql", "redis"]
    }
    
    # Create analyzer
    analyzer = ResumeJDSimilarityAnalyzer(engine)
    
    # Perform comprehensive analysis
    print("\nPerforming comprehensive semantic similarity analysis...\n")
    results = analyzer.comprehensive_analysis(resume_data, jd_data)
    
    # Print results
    print("-" * 70)
    print("FULL DOCUMENT COMPARISON")
    print("-" * 70)
    print(f"Similarity Score: {results['full_document']['similarity_score']:.4f}")
    print(f"Interpretation: {results['full_document']['interpretation']}")
    
    print("\n" + "-" * 70)
    print("RESPONSIBILITIES COMPARISON")
    print("-" * 70)
    print(f"Average Similarity: {results['responsibilities']['average_similarity']:.4f}")
    print(f"Interpretation: {results['responsibilities']['interpretation']}")
    if results['responsibilities']['best_match']:
        print(f"Best Match:")
        print(f"  Resume: {results['responsibilities']['best_match']['resume']}")
        print(f"  JD: {results['responsibilities']['best_match']['jd']}")
        print(f"  Score: {results['responsibilities']['best_match']['score']:.4f}")
    
    print("\n" + "-" * 70)
    print("PROJECTS COMPARISON")
    print("-" * 70)
    print(f"Average Similarity: {results['projects']['average_similarity']:.4f}")
    print(f"Interpretation: {results['projects']['interpretation']}")
    if results['projects']['best_match']:
        print(f"Best Match:")
        print(f"  Resume: {results['projects']['best_match']['resume']}")
        print(f"  JD: {results['projects']['best_match']['jd']}")
        print(f"  Score: {results['projects']['best_match']['score']:.4f}")
    
    print("\n" + "-" * 70)
    print("ADJACENT SKILLS COMPARISON")
    print("-" * 70)
    print(f"Matched Skills: {results['adjacent_skills']['matched']}")
    print(f"Partial Matches: {results['adjacent_skills']['partial']}")
    print(f"Missing Skills: {results['adjacent_skills']['missing']}")
    print("\nAdjacent Skill Pairs:")
    for pair in results['adjacent_skills']['adjacent_pairs']:
        print(f"  {pair['jd_skill']} <-> {pair['resume_skill']} ({pair['similarity']:.4f}) - {pair['match_type']}")
    
    print("\n" + "-" * 70)
    print("OVERALL SEMANTIC SCORE")
    print("-" * 70)
    print(f"Overall Score: {results['overall_semantic_score']:.4f}")
    print(f"Interpretation: {results['overall_interpretation']}")
    
    # Demonstrate finding most similar texts
    print("\n" + "=" * 70)
    print("FINDING MOST SIMILAR TEXTS")
    print("=" * 70)
    
    query = "building scalable web applications with Python"
    candidates = [
        "Developed REST APIs using Django framework",
        "Created microservices architecture for high-traffic systems",
        "Built data processing pipelines with Apache Spark",
        "Implemented CI/CD pipelines with Jenkins"
    ]
    
    print(f"\nQuery: {query}")
    print(f"Candidates: {len(candidates)}")
    print("\nTop 3 most similar:")
    
    top_matches = engine.find_most_similar(query, candidates, top_k=3)
    for i, (text, score) in enumerate(top_matches, 1):
        print(f"{i}. [{score:.4f}] {text}")
