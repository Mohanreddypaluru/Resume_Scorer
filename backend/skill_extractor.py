"""
Skill Extraction Engine for Resumes and Job Descriptions.

This module provides deterministic skill extraction from text using predefined
skill dictionaries, synonym matching, and fuzzy matching techniques.

Features:
- Predefined skill database with categories
- Synonym and related technology detection
- Normalized skill output
- Duplicate avoidance
- Preprocessing for robust matching
- Fuzzy matching suggestions
"""

import re
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import difflib


# ============================================================================
# PREDEFINED SKILL DATABASE
# ============================================================================

class SkillDatabase:
    """
    Centralized skill database with categories, synonyms, and related technologies.
    """
    
    SKILLS = {
        # Programming Languages
        "java": {
            "category": "programming_language",
            "synonyms": ["java", "java se", "java ee", "jdk", "jre"],
            "related": ["spring boot", "maven", "gradle", "junit", "hibernate"],
            "patterns": [r"\bjava\s*(?:se|ee)?\b", r"\bjdk\b", r"\bjre\b"]
        },
        "python": {
            "category": "programming_language",
            "synonyms": ["python", "python 3", "python3", "py"],
            "related": ["django", "flask", "fastapi", "pandas", "numpy", "scikit-learn"],
            "patterns": [r"\bpython\s*(?:3)?\b", r"\bpy\b(?=\s|$)"]
        },
        "javascript": {
            "category": "programming_language",
            "synonyms": ["javascript", "js", "ecmascript", "es6", "es2015"],
            "related": ["react", "angular", "vue", "node.js", "typescript"],
            "patterns": [r"\bjavascript\b", r"\bjs\b(?=\s|$)", r"\becmascript\b"]
        },
        
        # Frameworks
        "spring boot": {
            "category": "framework",
            "synonyms": ["spring boot", "springboot", "spring-boot"],
            "related": ["java", "spring", "maven", "gradle", "microservices"],
            "patterns": [r"\bspring\s*boot\b", r"\bspringboot\b"]
        },
        "flask": {
            "category": "framework",
            "synonyms": ["flask", "flask.py"],
            "related": ["python", "sqlalchemy", "jinja2", "werkzeug"],
            "patterns": [r"\bflask\b"]
        },
        "react": {
            "category": "framework",
            "synonyms": ["react", "react.js", "reactjs", "reactjs"],
            "related": ["javascript", "typescript", "redux", "next.js", "jsx"],
            "patterns": [r"\breact(?:\.js|js)?\b"]
        },
        
        # Databases
        "sql": {
            "category": "database",
            "synonyms": ["sql", "structured query language", "mysql", "postgresql", "postgres"],
            "related": ["database", "nosql", "mongodb", "redis", "oracle"],
            "patterns": [r"\bsql\b", r"\bmysql\b", r"\bpostgresql\b", r"\bpostgres\b"]
        },
        
        # AI/ML
        "machine learning": {
            "category": "ai_ml",
            "synonyms": ["machine learning", "ml", "machine-learning"],
            "related": ["deep learning", "artificial intelligence", "data science", "pandas", "scikit-learn"],
            "patterns": [r"\bmachine\s*learning\b", r"\bml\b(?=\s|$)"]
        },
        "deep learning": {
            "category": "ai_ml",
            "synonyms": ["deep learning", "dl", "deep-learning"],
            "related": ["machine learning", "neural networks", "tensorflow", "pytorch", "keras"],
            "patterns": [r"\bdeep\s*learning\b", r"\bdl\b(?=\s|$)"]
        },
        
        # DevOps/Cloud
        "docker": {
            "category": "devops",
            "synonyms": ["docker", "docker.io", "docker-compose"],
            "related": ["kubernetes", "containerization", "devops", "ci/cd"],
            "patterns": [r"\bdocker\b", r"\bdocker\s*compose\b"]
        },
        "aws": {
            "category": "cloud",
            "synonyms": ["aws", "amazon web services", "amazon aws"],
            "related": ["cloud", "ec2", "s3", "lambda", "rds", "cloud computing"],
            "patterns": [r"\baws\b", r"\bamazon\s*web\s*services\b"]
        },
        
        # Additional common skills
        "kubernetes": {
            "category": "devops",
            "synonyms": ["kubernetes", "k8s", "k8"],
            "related": ["docker", "containerization", "orchestration", "devops"],
            "patterns": [r"\bkubernetes\b", r"\bk8s\b"]
        },
        "git": {
            "category": "tool",
            "synonyms": ["git", "git version control"],
            "related": ["github", "gitlab", "version control", "devops"],
            "patterns": [r"\bgit\b"]
        },
        "linux": {
            "category": "os",
            "synonyms": ["linux", "gnu/linux", "unix"],
            "related": ["bash", "shell scripting", "command line", "devops"],
            "patterns": [r"\blinux\b", r"\bunix\b"]
        }
    }
    
    @classmethod
    def get_all_skills(cls) -> List[str]:
        """Get list of all normalized skill names."""
        return list(cls.SKILLS.keys())
    
    @classmethod
    def get_skill_info(cls, skill_name: str) -> Dict:
        """Get skill information including synonyms and related skills."""
        return cls.SKILLS.get(skill_name.lower(), {})
    
    @classmethod
    def get_skills_by_category(cls, category: str) -> List[str]:
        """Get all skills in a specific category."""
        return [skill for skill, info in cls.SKILLS.items() 
                if info.get("category") == category]


# ============================================================================
# PREPROCESSING
# ============================================================================

def preprocess_text(text: str) -> str:
    """
    Preprocess text for skill extraction.
    
    Args:
        text: Raw input text
        
    Returns:
        Preprocessed text
    """
    # Convert to lowercase for case-insensitive matching
    text = text.lower()
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that might interfere with matching
    # Keep hyphens, dots, and plus signs as they appear in tech names
    text = re.sub(r'[^\w\s\-\.\+]', ' ', text)
    
    # Normalize common variations
    text = re.sub(r'\.js\b', ' js', text)  # React.js -> react js
    text = re.sub(r'\.py\b', ' py', text)   # script.py -> script py
    
    return text.strip()


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into words and phrases.
    
    Args:
        text: Preprocessed text
        
    Returns:
        List of tokens
    """
    # Split into words
    words = text.split()
    
    # Also extract bigrams (2-word phrases)
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    
    # Also extract trigrams (3-word phrases)
    trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2)]
    
    return trigrams + bigrams + words


# ============================================================================
# DETERMINISTIC MATCHING
# ============================================================================

def exact_match(text: str, skill_db: Dict = None) -> Set[str]:
    """
    Extract skills using exact pattern matching.
    
    Args:
        text: Preprocessed text
        skill_db: Skill database (defaults to SkillDatabase.SKILLS)
        
    Returns:
        Set of matched skill names (normalized)
    """
    if skill_db is None:
        skill_db = SkillDatabase.SKILLS
    
    matched_skills = set()
    
    for skill_name, skill_info in skill_db.items():
        patterns = skill_info.get("patterns", [])
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matched_skills.add(skill_name)
                break  # Match found, move to next skill
    
    return matched_skills


def synonym_match(text: str, skill_db: Dict = None) -> Set[str]:
    """
    Extract skills by matching synonyms.
    
    Args:
        text: Preprocessed text
        skill_db: Skill database (defaults to SkillDatabase.SKILLS)
        
    Returns:
        Set of matched skill names (normalized)
    """
    if skill_db is None:
        skill_db = SkillDatabase.SKILLS
    
    matched_skills = set()
    
    for skill_name, skill_info in skill_db.items():
        synonyms = skill_info.get("synonyms", [])
        
        for synonym in synonyms:
            # Create pattern for synonym
            pattern = r'\b' + re.escape(synonym) + r'\b'
            
            if re.search(pattern, text, re.IGNORECASE):
                matched_skills.add(skill_name)
                break  # Match found, move to next skill
    
    return matched_skills


def related_match(text: str, skill_db: Dict = None) -> Set[str]:
    """
    Extract skills by matching related technologies.
    
    This is a weaker match - only used if no exact/synonym match found.
    
    Args:
        text: Preprocessed text
        skill_db: Skill database (defaults to SkillDatabase.SKILLS)
        
    Returns:
        Set of matched skill names (normalized)
    """
    if skill_db is None:
        skill_db = SkillDatabase.SKILLS
    
    matched_skills = set()
    
    for skill_name, skill_info in skill_db.items():
        related = skill_info.get("related", [])
        
        for related_term in related:
            # Create pattern for related term
            pattern = r'\b' + re.escape(related_term) + r'\b'
            
            if re.search(pattern, text, re.IGNORECASE):
                matched_skills.add(skill_name)
                break  # Match found, move to next skill
    
    return matched_skills


# ============================================================================
# FUZZY MATCHING
# ============================================================================

def fuzzy_match(text: str, skill_db: Dict = None, threshold: float = 0.8) -> Dict[str, float]:
    """
    Extract skills using fuzzy string matching.
    
    This helps catch typos and variations not covered by synonyms.
    
    Args:
        text: Preprocessed text
        skill_db: Skill database (defaults to SkillDatabase.SKILLS)
        threshold: Similarity threshold (0.0 to 1.0)
        
    Returns:
        Dictionary of skill names with similarity scores
    """
    if skill_db is None:
        skill_db = SkillDatabase.SKILLS
    
    # Get all possible skill names and synonyms
    all_terms = []
    term_to_skill = {}
    
    for skill_name, skill_info in skill_db.items():
        all_terms.append(skill_name)
        term_to_skill[skill_name] = skill_name
        
        for synonym in skill_info.get("synonyms", []):
            all_terms.append(synonym)
            term_to_skill[synonym] = skill_name
    
    # Tokenize text
    tokens = tokenize_text(text)
    
    # Find fuzzy matches
    fuzzy_matches = {}
    
    for token in tokens:
        for term in all_terms:
            # Calculate similarity ratio
            similarity = difflib.SequenceMatcher(None, token.lower(), term.lower()).ratio()
            
            if similarity >= threshold:
                skill_name = term_to_skill[term]
                
                # Keep the highest similarity for each skill
                if skill_name not in fuzzy_matches or similarity > fuzzy_matches[skill_name]:
                    fuzzy_matches[skill_name] = similarity
    
    return fuzzy_matches


# ============================================================================
# MAIN EXTRACTION ENGINE
# ============================================================================

class SkillExtractor:
    """
    Main skill extraction engine that combines multiple matching strategies.
    """
    
    def __init__(self, skill_db: Dict = None, use_fuzzy: bool = True, fuzzy_threshold: float = 0.85):
        """
        Initialize the skill extractor.
        
        Args:
            skill_db: Custom skill database (defaults to SkillDatabase.SKILLS)
            use_fuzzy: Whether to use fuzzy matching
            fuzzy_threshold: Similarity threshold for fuzzy matching
        """
        self.skill_db = skill_db or SkillDatabase.SKILLS
        self.use_fuzzy = use_fuzzy
        self.fuzzy_threshold = fuzzy_threshold
    
    def extract(self, text: str) -> Dict:
        """
        Extract skills from text using all available strategies.
        
        Args:
            text: Raw input text
            
        Returns:
            Dictionary containing:
            - skills: List of normalized skill names
            - skills_by_category: Skills grouped by category
            - fuzzy_matches: Skills found via fuzzy matching with scores
            - match_details: Details about how each skill was matched
        """
        # Preprocess text
        preprocessed = preprocess_text(text)
        
        # Extract using different strategies
        exact_matches = exact_match(preprocessed, self.skill_db)
        synonym_matches = synonym_match(preprocessed, self.skill_db)
        related_matches = related_match(preprocessed, self.skill_db)
        
        # Combine matches (priority: exact > synonym > related)
        all_matches = exact_matches | synonym_matches | related_matches
        
        # Fuzzy matching if enabled
        fuzzy_matches = {}
        if self.use_fuzzy:
            fuzzy_matches = fuzzy_match(preprocessed, self.skill_db, self.fuzzy_threshold)
            # Add fuzzy matches that weren't already found
            for skill, score in fuzzy_matches.items():
                if skill not in all_matches:
                    all_matches.add(skill)
        
        # Remove duplicates and normalize
        normalized_skills = sorted(list(all_matches))
        
        # Group by category
        skills_by_category = defaultdict(list)
        match_details = {}
        
        for skill in normalized_skills:
            skill_info = self.skill_db.get(skill, {})
            category = skill_info.get("category", "other")
            skills_by_category[category].append(skill)
            
            # Determine match type
            match_type = "exact" if skill in exact_matches else \
                       "synonym" if skill in synonym_matches else \
                       "related" if skill in related_matches else \
                       "fuzzy"
            
            match_details[skill] = {
                "category": category,
                "match_type": match_type,
                "fuzzy_score": fuzzy_matches.get(skill, None)
            }
        
        return {
            "skills": normalized_skills,
            "skills_by_category": dict(skills_by_category),
            "fuzzy_matches": fuzzy_matches,
            "match_details": match_details,
            "total_skills": len(normalized_skills)
        }
    
    def extract_skills_only(self, text: str) -> List[str]:
        """
        Extract only the list of skill names (simplified interface).
        
        Args:
            text: Raw input text
            
        Returns:
            List of normalized skill names
        """
        result = self.extract(text)
        return result["skills"]


# ============================================================================
# SAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Sample resume text
    sample_text = """
    Senior Software Engineer with 5+ years of experience.
    
    Skills:
    - Java, Spring Boot, Hibernate
    - Python, Django, Flask
    - JavaScript, React.js
    - SQL, PostgreSQL, MongoDB
    - Machine Learning with TensorFlow
    - Deep Learning using PyTorch
    - Docker, Kubernetes
    - AWS (EC2, S3, Lambda)
    - Git, Linux, Bash scripting
    """
    
    # Initialize extractor
    extractor = SkillExtractor(use_fuzzy=True, fuzzy_threshold=0.85)
    
    # Extract skills
    result = extractor.extract(sample_text)
    
    # Print results
    print("=" * 60)
    print("SKILL EXTRACTION RESULTS")
    print("=" * 60)
    
    print(f"\nTotal Skills Found: {result['total_skills']}")
    print(f"\nSkills: {', '.join(result['skills'])}")
    
    print("\n" + "-" * 60)
    print("Skills by Category:")
    print("-" * 60)
    for category, skills in result['skills_by_category'].items():
        print(f"{category}: {', '.join(skills)}")
    
    print("\n" + "-" * 60)
    print("Match Details:")
    print("-" * 60)
    for skill, details in result['match_details'].items():
        fuzzy_info = f" (fuzzy: {details['fuzzy_score']:.2f})" if details['fuzzy_score'] else ""
        print(f"{skill}: {details['match_type']}{fuzzy_info} [{details['category']}]")
    
    print("\n" + "-" * 60)
    print("Fuzzy Matches:")
    print("-" * 60)
    if result['fuzzy_matches']:
        for skill, score in result['fuzzy_matches'].items():
            print(f"{skill}: {score:.2f}")
    else:
        print("No fuzzy matches found")
    
    # Test with simplified interface
    print("\n" + "=" * 60)
    print("SIMPLIFIED INTERFACE TEST")
    print("=" * 60)
    skills = extractor.extract_skills_only(sample_text)
    print(f"Skills: {skills}")
