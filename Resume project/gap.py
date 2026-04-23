"""
Skill gap analysis: match percentage, ordered missing skills, top critical gaps.
"""

from typing import Dict, List, Set

from matcher import matched_skills


def _jd_skill_order(jd_skills: List[str]) -> Dict[str, int]:
    """First occurrence index in JD list (importance proxy)."""
    order: Dict[str, int] = {}
    for i, s in enumerate(jd_skills):
        if s not in order:
            order[s] = i
    return order


def analyze_gap(
    resume_skills: List[str],
    jd_skills: List[str],
) -> Dict[str, object]:
    """
    matched: skills in both
    missing: JD skills not in resume, sorted by JD importance (order first)
    score: 0-100 match percentage vs JD; 100 if jd_skills empty
    critical_missing: top 3 missing skills
    """
    resume_set: Set[str] = set(resume_skills)
    jd_unique: List[str] = []
    seen: Set[str] = set()
    for s in jd_skills:
        if s not in seen:
            seen.add(s)
            jd_unique.append(s)

    if not jd_unique:
        return {
            "matched": [],
            "missing": [],
            "score": 100.0,
            "critical_missing": [],
        }

    matched = matched_skills(resume_skills, jd_unique)
    matched_set = set(matched)

    missing = [s for s in jd_unique if s not in resume_set]
    order = _jd_skill_order(jd_skills)
    missing.sort(key=lambda x: (order.get(x, 999), x))

    score = (len(matched) / len(jd_unique)) * 100.0

    critical_missing = missing[:3]

    return {
        "matched": matched,
        "missing": missing,
        "score": round(score, 2),
        "critical_missing": critical_missing,
    }
