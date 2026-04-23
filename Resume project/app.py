"""
AI Resume Analyzer & Career Roadmap Generator — Flask API.

Uses Ollama (local LLM) for intelligent analysis, with robust fallbacks.
"""

import io
import json
import os
import re
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PyPDF2 import PdfReader

# ── Your modules ──
from gap import analyze_gap
from skill_extractor import extract_skills, extract_skills_ordered
from job_recommender import top_jobs
from resume_improver import build_improved_resume
from roadmap import build_roadmap

# ── AI (safe import) ──
try:
    from ai_engine import ask_ai
except Exception:
    ask_ai = None


app = Flask(__name__)
CORS(app)

# ── Load curated roadmap links once ──
_ROADMAP_LINKS: Dict[str, Any] = {}
_ROADMAP_PATH = os.path.join(os.path.dirname(__file__), "data", "roadmap_links.json")
try:
    with open(_ROADMAP_PATH, encoding="utf-8") as _f:
        _ROADMAP_LINKS = json.load(_f)
except Exception:
    pass


# ── Curated platform-specific search URLs (NEVER plain Google) ──
_LEARN_PLATFORMS = [
    ("YouTube Tutorial", "https://www.youtube.com/results?search_query=learn+{q}+tutorial"),
    ("FreeCodeCamp", "https://www.freecodecamp.org/news/search/?query={q}"),
    ("MDN Web Docs", "https://developer.mozilla.org/en-US/search?q={q}"),
]


def _get_resource_for_skill(skill: str) -> Dict[str, Any]:
    """
    Return a rich resource entry for a skill.
    Priority: curated JSON → AI-generated → curated platform search (never plain Google).
    """
    entry = _ROADMAP_LINKS.get(skill)

    if entry and isinstance(entry, dict) and "link" in entry:
        # Curated entry with full metadata
        return {
            "link": str(entry["link"]),
            "alt_links": entry.get("alt_links", []),
            "level": str(entry.get("level", "Beginner")),
            "duration": str(entry.get("duration", "2-4 weeks")),
            "source": "curated",
        }
    elif entry and isinstance(entry, str):
        # Legacy format: just a URL string
        return {
            "link": entry,
            "alt_links": [],
            "level": "Beginner",
            "duration": "2-4 weeks",
            "source": "curated",
        }
    else:
        # Skill not in curated list — use platform-specific search
        q = urllib.parse.quote_plus(skill)
        primary_link = f"https://www.youtube.com/results?search_query=learn+{q}+full+course"
        alt_links = [url.format(q=q) for _, url in _LEARN_PLATFORMS[1:]]
        return {
            "link": primary_link,
            "alt_links": alt_links,
            "level": "Beginner",
            "duration": "2-4 weeks",
            "source": "auto",
        }


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(os.path.dirname(__file__), "ai-resume-analyzer.html")


# ── PDF extraction ──
def extract_pdf_text(file) -> str:
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()


# ── Request parser (supports JSON + PDF) ──
def _parse_request() -> Tuple[str, str]:
    if "resume_pdf" in request.files:
        file = request.files["resume_pdf"]
        resume_text = extract_pdf_text(file)
        job_desc = request.form.get("job_desc", "")
    else:
        data = request.get_json(silent=True) or {}
        resume_text = data.get("resume", "")
        job_desc = data.get("job_desc", "")

    return resume_text, job_desc


@app.route("/analyze", methods=["POST"])
def analyze() -> Any:
    try:
        resume_text, job_description = _parse_request()
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    if not resume_text.strip():
        return jsonify({"error": "Resume text is empty. Paste text or upload a PDF."}), 400

    # ── Skill extraction ──
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills_ordered(job_description) if job_description.strip() else []

    # ── Gap analysis ──
    gap_result = analyze_gap(resume_skills, jd_skills)
    missing = gap_result["missing"]
    matched = gap_result["matched"]
    critical = gap_result["critical_missing"]

    # ── Match score (clamped 0-100) ──
    match_score = max(0, min(float(gap_result["score"]), 100))

    # ── Build rich roadmap with real resource links ──
    roadmap_list = []
    for i, skill in enumerate(missing):
        resource = _get_resource_for_skill(skill)
        roadmap_list.append({
            "step": i + 1,
            "text": f"Learn {skill}",
            "skill": skill,
            "link": resource["link"],
            "alt_links": resource.get("alt_links", []),
            "level": resource["level"],
            "duration": resource["duration"],
            "source": resource["source"],
        })

    # ══════════════════════════════
    # 🤖 AI-POWERED ANALYSIS
    # ══════════════════════════════

    improved = None
    jobs_before = None
    jobs_after = None
    insight = None
    ai_roadmap_descriptions = {}

    try:
        if ask_ai:
            # ── AI-improved resume ──
            improved = ask_ai(f"""You are a professional resume writer. Rewrite this resume to be ATS-friendly and professionally formatted.

Rules:
- Use clear sections: PROFESSIONAL SUMMARY, EXPERIENCE, SKILLS, EDUCATION, PROJECTS
- Use strong action verbs and quantify achievements where possible
- Tailor it toward the job description if provided
- Output ONLY the resume text, no explanations or preamble

Resume:
{resume_text}

{"Job Description:" + chr(10) + job_description if job_description.strip() else ""}""")

            # ── Job suggestions BEFORE ──
            jobs_before_raw = ask_ai(f"""Based on this resume, suggest exactly 5 realistic job roles this person can apply for RIGHT NOW with their current skills.

Resume:
{resume_text}

Rules:
- Output ONLY job titles, one per line
- No numbering, no explanations, no bullet points
- Be specific (e.g. "Junior Frontend Developer" not just "Developer")
""")
            jobs_before = [j.strip().lstrip("0123456789.-) ") for j in jobs_before_raw.split("\n") if j.strip() and len(j.strip()) > 3][:5]

            # ── Job suggestions AFTER ──
            if missing:
                jobs_after_raw = ask_ai(f"""If this person learns these additional skills: {', '.join(missing[:8])}

And they already have: {', '.join(resume_skills[:10])}

Suggest exactly 5 better/higher-level job roles they could target.

Rules:
- Output ONLY job titles, one per line
- No numbering, no explanations, no bullet points
- Suggest roles that are a step UP from their current level
""")
                jobs_after = [j.strip().lstrip("0123456789.-) ") for j in jobs_after_raw.split("\n") if j.strip() and len(j.strip()) > 3][:5]

            # ── AI-generated learning descriptions for roadmap ──
            if missing:
                roadmap_ai_raw = ask_ai(f"""For each of these skills, write ONE short sentence (max 15 words) explaining WHY this skill is important for the target job and what it enables:

Skills: {', '.join(missing[:10])}

{"Target Job Description:" + chr(10) + job_description[:500] if job_description.strip() else ""}

Rules:
- Format: SkillName: short description
- One skill per line
- Be specific to the job context
- No numbering
""")
                for line in roadmap_ai_raw.split("\n"):
                    line = line.strip()
                    if ":" in line and len(line) > 5:
                        parts = line.split(":", 1)
                        skill_name = parts[0].strip().lstrip("0123456789.-) ")
                        desc = parts[1].strip()
                        ai_roadmap_descriptions[skill_name] = desc

            # ── Insight ──
            insight = ask_ai(f"""Analyze this resume against the job description in exactly 2 concise sentences.
First sentence: What the candidate does well.
Second sentence: The most important gap to address.

Resume:
{resume_text[:800]}

{"Job Description:" + chr(10) + job_description[:500] if job_description.strip() else "No specific job description provided."}

Rules:
- Exactly 2 sentences, no more
- Be specific, not generic
""")

        else:
            raise RuntimeError("AI not available")

    except Exception as e:
        print(f"AI ERROR: {e}")

        # ── FALLBACK: use existing modules ──
        if improved is None:
            improved = build_improved_resume(
                original_resume=resume_text,
                matched=matched,
                missing=missing,
                critical=critical,
                job_description=job_description,
            )

        if jobs_before is None:
            resume_skill_set = set(resume_skills)
            jobs_before_data = top_jobs(resume_skill_set, limit=5, label="Current profile:")
            jobs_before = [j["title"] for j in jobs_before_data]

        if jobs_after is None:
            if missing:
                future_skills = set(resume_skills) | set(missing)
                jobs_after_data = top_jobs(future_skills, limit=5, label="After upskilling:")
                jobs_after = [j["title"] for j in jobs_after_data]
            else:
                jobs_after = jobs_before

        if insight is None:
            insight = f"Found {len(resume_skills)} skills in your resume. {'You match ' + str(len(matched)) + '/' + str(len(matched) + len(missing)) + ' required skills from the job description.' if jd_skills else 'Add a job description for targeted analysis.'}"

    # ── Enrich roadmap with AI descriptions ──
    for item in roadmap_list:
        skill = item["skill"]
        if skill in ai_roadmap_descriptions:
            item["description"] = ai_roadmap_descriptions[skill]
        else:
            item["description"] = f"Essential skill for your target role. Estimated learning time: {item['duration']}."

    # ── FINAL RESPONSE ──
    payload = {
        "skills": resume_skills,
        "match_score": match_score,
        "missing_skills": missing,
        "roadmap": roadmap_list,
        "improved_resume": improved,
        "insight": insight,
        "job_suggestions_before": jobs_before or [],
        "job_suggestions_after": jobs_after or [],
    }

    return jsonify(payload), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)