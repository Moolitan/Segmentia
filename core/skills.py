from __future__ import annotations

import os

from core.constants import ROOT


def load_skill_doc(skills_dir: str, skill_name: str) -> tuple[str | None, str | None]:
    skill_path = os.path.join(skills_dir, skill_name, "SKILL.md")
    if not os.path.isfile(skill_path):
        return None, None
    with open(skill_path, encoding="utf-8") as f:
        return skill_path, f.read()


def resolve_skills_dir(workspace: str, explicit: str | None) -> str:
    if explicit:
        return os.path.abspath(explicit)
    agents_skills = os.path.join(os.path.abspath(workspace), ".agents", "skills")
    if os.path.isdir(agents_skills):
        return agents_skills
    return os.path.join(ROOT, "skills")
