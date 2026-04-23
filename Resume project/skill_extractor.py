"""
Extract canonical skills from text using keyword and synonym matching.
Multi-word skills are matched before single-token synonyms for better accuracy.
"""

import re
from typing import Dict, List, Set

# Canonical skill names (Title Case) — broad list for resume/JD matching
SKILL_KEYWORDS: List[str] = [
    "REST API",
    "CI/CD",
    "Machine Learning",
    "Deep Learning",
    "Data Structures",
    "Computer Vision",
    "Natural Language Processing",
    "Agile",
    "Scrum",
    "GraphQL",
    "MongoDB",
    "PostgreSQL",
    "MySQL",
    "Redis",
    "Elasticsearch",
    "Apache Kafka",
    "RabbitMQ",
    "TensorFlow",
    "PyTorch",
    "Scikit-learn",
    "Pandas",
    "NumPy",
    "Kubernetes",
    "Docker",
    "Jenkins",
    "GitHub Actions",
    "Terraform",
    "Ansible",
    "Linux",
    "Unix",
    "Bash",
    "Shell Scripting",
    "Networking",
    "AWS",
    "Azure",
    "Google Cloud",
    "GCP",
    "Firebase",
    "React",
    "Angular",
    "Vue.js",
    "Node.js",
    "Express.js",
    "Next.js",
    "Django",
    "Flask",
    "FastAPI",
    "Spring Boot",
    "Java",
    "JavaScript",
    "TypeScript",
    "Python",
    "C++",
    "C#",
    "Go",
    "Rust",
    "Ruby",
    "PHP",
    "Swift",
    "Kotlin",
    "HTML",
    "CSS",
    "SASS",
    "Tailwind CSS",
    "Bootstrap",
    "SQL",
    "NoSQL",
    "Tableau",
    "Power BI",
    "Excel",
    "Statistics",
    "Algorithms",
    "Git",
    "Jira",
    "Microservices",
    "OAuth",
    "JWT",
    "WebSockets",
]

# Longer phrases first for greedy matching (handled by sorting keyword length)
SKILL_KEYWORDS_SORTED = sorted(set(SKILL_KEYWORDS), key=len, reverse=True)

# Lowercase alias / abbreviation -> canonical skill name (must match SKILL_KEYWORDS or added here)
SYNONYMS: Dict[str, str] = {
    "javascript": "JavaScript",
    "ecmascript": "JavaScript",
    "ts": "TypeScript",
    "typescript": "TypeScript",
    "py": "Python",
    "python3": "Python",
    "cpp": "C++",
    "c plus plus": "C++",
    "csharp": "C#",
    "c#": "C#",
    "golang": "Go",
    "go lang": "Go",
    "k8s": "Kubernetes",
    "kube": "Kubernetes",
    "kubernetes": "Kubernetes",
    "docker": "Docker",
    "aws": "AWS",
    "amazon web services": "AWS",
    "azure": "Azure",
    "gcp": "Google Cloud",
    "google cloud platform": "Google Cloud",
    "tf": "TensorFlow",
    "tensorflow": "TensorFlow",
    "pytorch": "PyTorch",
    "torch": "PyTorch",
    "react.js": "React",
    "reactjs": "React",
    "node": "Node.js",
    "nodejs": "Node.js",
    "node.js": "Node.js",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "mysql": "MySQL",
    "mongo": "MongoDB",
    "mongodb": "MongoDB",
    "ml": "Machine Learning",
    "machine learning": "Machine Learning",
    "dl": "Deep Learning",
    "deep learning": "Deep Learning",
    "nlp": "Natural Language Processing",
    "rest": "REST API",
    "restful": "REST API",
    "rest api": "REST API",
    "graphql": "GraphQL",
    "sql": "SQL",
    "nosql": "NoSQL",
    "git": "Git",
    "github": "Git",
    "gitlab": "Git",
    "linux": "Linux",
    "bash": "Bash",
    "shell": "Shell Scripting",
    "html": "HTML",
    "html5": "HTML",
    "css": "CSS",
    "css3": "CSS",
    "django": "Django",
    "flask": "Flask",
    "fastapi": "FastAPI",
    "java": "Java",
    "kotlin": "Kotlin",
    "swift": "Swift",
    "ruby": "Ruby",
    "php": "PHP",
    "rust": "Rust",
    "angular": "Angular",
    "vue": "Vue.js",
    "vue.js": "Vue.js",
    "vuejs": "Vue.js",
    "jenkins": "Jenkins",
    "terraform": "Terraform",
    "ansible": "Ansible",
    "tableau": "Tableau",
    "excel": "Excel",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "scikit-learn": "Scikit-learn",
    "sklearn": "Scikit-learn",
    "kafka": "Apache Kafka",
    "redis": "Redis",
    "elastic": "Elasticsearch",
    "elasticsearch": "Elasticsearch",
    "microservices": "Microservices",
    "jwt": "JWT",
    "oauth": "OAuth",
    "agile": "Agile",
    "scrum": "Scrum",
    "jira": "Jira",
    "statistics": "Statistics",
    "stats": "Statistics",
    "algorithms": "Algorithms",
    "dsa": "Data Structures",
    "data structures": "Data Structures",
    "networking": "Networking",
    "ci/cd": "CI/CD",
    "cicd": "CI/CD",
    "power bi": "Power BI",
    "powerbi": "Power BI",
}


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _find_keyword_skills(lower_text: str, found: Set[str]) -> None:
    """Match multi-word and single-word canonical keywords as substrings with word boundaries where possible."""
    for skill in SKILL_KEYWORDS_SORTED:
        pattern = re.escape(skill.lower())
        # Word boundary for short tokens to reduce false positives (e.g. "go" in "good")
        if len(skill) <= 3:
            if re.search(rf"\b{pattern}\b", lower_text):
                found.add(skill)
        else:
            if pattern in lower_text:
                found.add(skill)


def _find_synonym_skills(lower_text: str, found: Set[str]) -> None:
    """Match synonym phrases and tokens (longer keys first)."""
    sorted_syns = sorted(SYNONYMS.keys(), key=len, reverse=True)
    remaining = lower_text
    for key in sorted_syns:
        canonical = SYNONYMS[key]
        if len(key) <= 2:
            if re.search(rf"\b{re.escape(key)}\b", remaining):
                found.add(canonical)
        else:
            if key in remaining:
                found.add(canonical)


def extract_skills(text: str) -> List[str]:
    """
    Return a sorted, deduplicated list of canonical skills found in text.
    """
    if not text or not text.strip():
        return []

    lower_text = _normalize_whitespace(text).lower()
    found: Set[str] = set()

    _find_keyword_skills(lower_text, found)
    _find_synonym_skills(lower_text, found)

    # Deduplicate overlapping canonical names (e.g. REST from synonym vs keyword)
    return sorted(found)


def _first_occurrence(lower_text: str, canonical: str) -> int:
    """Earliest index of canonical name or any of its synonyms in text."""
    idx = lower_text.find(canonical.lower())
    if idx != -1:
        return idx
    best = 10**9
    for syn, canon in SYNONYMS.items():
        if canon != canonical:
            continue
        j = lower_text.find(syn)
        if j != -1:
            best = min(best, j)
    return best if best < 10**9 else 10**9


def extract_skills_ordered(text: str) -> List[str]:
    """
    Same skills as extract_skills, but ordered by first appearance in the text
    (useful for job descriptions to rank missing skills by importance).
    """
    if not text or not text.strip():
        return []

    lower_text = _normalize_whitespace(text).lower()
    found: Set[str] = set()
    _find_keyword_skills(lower_text, found)
    _find_synonym_skills(lower_text, found)
    return sorted(found, key=lambda s: (_first_occurrence(lower_text, s), s))
