"""Generate an improved resume as plain text with sections."""

from typing import List, Optional


def build_improved_resume(
    original_resume: str,
    matched: List[str],
    missing: List[str],
    critical: List[str],
    job_description: Optional[str],
) -> str:
    """
    Multi-section text: professional summary, suggested improvements, skills to add.
    """
    lines: List[str] = []

    # Professional summary
    lines.append("PROFESSIONAL SUMMARY")
    lines.append("-" * 40)
    if job_description and job_description.strip():
        skills_hint = ", ".join(matched[:8]) if matched else "relevant technical and professional skills"
        lines.append(
            f"Results-driven professional targeting roles aligned with the provided job description. "
            f"Brings demonstrated strength in {skills_hint}. "
            f"Committed to delivering quality work and closing skill gaps through focused learning."
        )
    else:
        skills_line = ", ".join(matched[:12]) if matched else "core professional competencies"
        lines.append(
            f"Versatile professional with experience and skills spanning {skills_line}. "
            f"Focused on growth, collaboration, and applying technical expertise to solve real problems."
        )
    lines.append("")

    # Suggested improvements
    lines.append("SUGGESTED IMPROVEMENTS")
    lines.append("-" * 40)
    suggestions = [
        "Quantify impact: add metrics (%, time saved, scale) next to key achievements.",
        "Align bullet points with keywords from the target role when applying.",
        "Group technical skills by category (e.g., Languages, Frameworks, Cloud).",
    ]
    if missing:
        suggestions.insert(
            0,
            f"Highlight or train toward these in-demand areas: {', '.join(missing[:5])}"
            + ("..." if len(missing) > 5 else "")
            + ".",
        )
    for s in suggestions:
        lines.append(f"• {s}")
    lines.append("")

    # Skills to add (from gap analysis)
    lines.append("SKILLS TO ADD (FROM GAP ANALYSIS)")
    lines.append("-" * 40)
    if critical:
        lines.append("Priority:")
        for sk in critical:
            lines.append(f"• {sk}")
        lines.append("")
    if missing:
        rest = [m for m in missing if m not in set(critical)]
        if rest:
            lines.append("Also consider:")
            for sk in rest[:15]:
                lines.append(f"• {sk}")
            if len(rest) > 15:
                lines.append("• ... (see full missing list in analysis)")
    else:
        lines.append("• No specific gaps vs. job description — keep strengthening your listed skills.")
    lines.append("")

    # Original resume preserved
    lines.append("ORIGINAL RESUME (REFERENCE)")
    lines.append("-" * 40)
    lines.append(original_resume.strip())

    return "\n".join(lines).strip()
