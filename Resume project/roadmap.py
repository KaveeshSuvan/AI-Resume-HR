"""Build learning roadmap from missing skills using roadmap_links.json."""

import json
import os
from typing import Any, Dict, List, Optional

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_ROADMAP_PATH = os.path.join(_DATA_DIR, "roadmap_links.json")

_DEFAULT_LINK = "https://www.google.com/search?q=learn+{skill}"
_DEFAULT_LEVEL = "Beginner"
_DEFAULT_DURATION = "2-4 weeks"

_links_cache: Optional[Dict[str, Any]] = None


def _load_links() -> Dict[str, Any]:
    global _links_cache
    if _links_cache is None:
        with open(_ROADMAP_PATH, encoding="utf-8") as f:
            _links_cache = json.load(f)
    return _links_cache


def build_roadmap(missing_skills: List[str]) -> Dict[str, Dict[str, str]]:
    """
    For each missing skill, return link, level, duration.
    Unknown skills get a generic search link and defaults.
    """
    data = _load_links()
    out: Dict[str, Dict[str, str]] = {}
    for skill in missing_skills:
        entry = data.get(skill)
        if entry and isinstance(entry, dict):
            out[skill] = {
                "link": str(entry.get("link", _DEFAULT_LINK.format(skill=skill.replace(" ", "+")))),
                "level": str(entry.get("level", _DEFAULT_LEVEL)),
                "duration": str(entry.get("duration", _DEFAULT_DURATION)),
            }
        else:
            q = skill.replace(" ", "+")
            out[skill] = {
                "link": _DEFAULT_LINK.format(skill=q),
                "level": _DEFAULT_LEVEL,
                "duration": _DEFAULT_DURATION,
            }
    return out
