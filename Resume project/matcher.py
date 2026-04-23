"""Match resume skills against job-description skills."""

from typing import List, Set


def matched_skills(resume_skills: List[str], jd_skills: List[str]) -> List[str]:
    """Return sorted intersection of resume and JD skills (canonical names)."""
    rs: Set[str] = set(resume_skills)
    js: Set[str] = set(jd_skills)
    return sorted(rs & js)
