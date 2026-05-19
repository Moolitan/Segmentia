from pathlib import Path
import shutil

SRC_ROOT = Path("anthropic_skill_benchmark_8_repos")
DST_ROOT = Path("anthropic_skill_benchmark_8_repos_explicit_skills")

SKILL_MAP = {
    "doc_coauthoring_design_doc": ["doc-coauthoring"],
    "internal_comms_incident_update": ["internal-comms"],
    "web_artifact_with_theme": ["web-artifacts-builder", "theme-factory"],
    "mcp_server_and_spec": ["mcp-builder", "doc-coauthoring"],
    "launch_poster_page_pack": ["canvas-design", "web-artifacts-builder", "theme-factory"],
    "slack_launch_pack": ["internal-comms", "slack-gif-creator", "brand-guidelines"],
}

TURN_PREFIX = {
    1: "Before you begin, explicitly reference and use the relevant skills for this task.",
    2: "As you continue, keep using the relevant skills and follow their guidance.",
    3: "Please continue this task by explicitly referencing and using the relevant skills.",
    4: "Please keep using the relevant skills and follow them while refining the work.",
    5: "Continue the task and make sure you are still using the relevant skills as guidance.",
    6: "Please keep working with the relevant skills explicitly in mind.",
    7: "As you revise this further, continue to reference and use the relevant skills.",
    8: "For this final step, explicitly use the relevant skills and keep the output aligned with them.",
}


def skill_instruction(repo_name: str) -> str:
    skills = SKILL_MAP[repo_name]
    if len(skills) == 1:
        return f'Please explicitly reference and use the "{skills[0]}" skill for this turn.'
    if len(skills) == 2:
        return (
            f'Please explicitly reference and use the "{skills[0]}" skill and the "{skills[1]}" skill for this turn.'
        )
    quoted = ", ".join(f'"{s}"' for s in skills[:-1])
    return (
        f'Please explicitly reference and use the {quoted}, and "{skills[-1]}" skills for this turn.'
    )


def rewrite_turn(repo_name: str, turn_idx: int, original_text: str) -> str:
    prefix = TURN_PREFIX.get(turn_idx, "Please continue using the relevant skills.")
    skill_line = skill_instruction(repo_name)
    original_text = original_text.strip()
    return f"{prefix}\n{skill_line}\n\n{original_text}\n"


def main():
    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"Source benchmark root not found: {SRC_ROOT.resolve()}")

    if DST_ROOT.exists():
        shutil.rmtree(DST_ROOT)

    shutil.copytree(SRC_ROOT, DST_ROOT)

    for repo_dir in DST_ROOT.iterdir():
        if not repo_dir.is_dir():
            continue

        repo_name = repo_dir.name
        if repo_name not in SKILL_MAP:
            continue

        turns_dir = repo_dir / "turns"
        if not turns_dir.exists():
            continue

        for turn_file in sorted(turns_dir.glob("turn_*.txt")):
            turn_idx = int(turn_file.stem.split("_")[1])
            original = turn_file.read_text(encoding="utf-8")
            updated = rewrite_turn(repo_name, turn_idx, original)
            turn_file.write_text(updated, encoding="utf-8")

        readme_path = repo_dir / "README.md"
        if readme_path.exists():
            readme_text = readme_path.read_text(encoding="utf-8").rstrip()
            readme_text += "\n\n## Variant\nThis is the explicit-skill version of the benchmark prompts. Each turn directly instructs the agent to reference and use the relevant skill(s).\n"
            readme_path.write_text(readme_text + "\n", encoding="utf-8")

    print(f"Created explicit-skill benchmark repos under: {DST_ROOT.resolve()}")


if __name__ == "__main__":
    main()