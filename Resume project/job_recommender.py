"""Rank job roles from jobs.json against skill sets."""

import json
import os
from typing import Any, Dict, List, Set

_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "jobs.json")


def _load_roles() -> List[Dict[str, Any]]:
    with open(_DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("roles", []))


def _coverage_score(candidate: Set[str], required: List[str]) -> float:
    if not required:
        return 0.0
    req_set = set(required)
    inter = len(candidate & req_set)
    return round(100.0 * inter / len(req_set), 2)


def _why(title: str, candidate: Set[str], required: List[str], label: str) -> str:
    req_set = set(required)
    hit = sorted(candidate & req_set)
    miss = sorted(req_set - candidate)
    n_hit = len(hit)
    n_req = len(required)
    base = (
        f"{label} Matches {n_hit}/{n_req} required skills for {title}"
    )
    if hit:
        base += f" — strong in: {', '.join(hit[:6])}"
        if len(hit) > 6:
            base += ", ..."
    if miss:
        base += f" — gaps: {', '.join(miss[:4])}"
        if len(miss) > 4:
            base += ", ..."
    return base


def top_jobs(
    skills: Set[str],
    limit: int = 3,
    label: str = "Current profile:",
) -> List[Dict[str, Any]]:
    """Return top roles by coverage score with explanations."""
    roles = _load_roles()
    ranked: List[tuple[float, str, Dict[str, Any]]] = []
    for role in roles:
        title = str(role.get("title", "Role"))
        req = list(role.get("required_skills", []))
        score = _coverage_score(skills, req)
        ranked.append((score, title, role))
    ranked.sort(key=lambda x: (-x[0], x[1]))

    out: List[Dict[str, Any]] = []
    for score, title, role in ranked[:limit]:
        req = list(role.get("required_skills", []))
        why = _why(title, skills, req, label)
        out.append({"title": title, "match_score": score, "why": why})
    return out
