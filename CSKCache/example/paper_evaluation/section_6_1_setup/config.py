"""Editable matrix for Section 6.1 setup validation."""

from pathlib import Path

from paper_evaluation.config import ROOT, TASK_PROMPT_ROOT


WORKLOADS = (
    (
        "internal-comms",
        ROOT / "skills/internal-comms/SKILL.md",
        TASK_PROMPT_ROOT / "internal-comms-incident-update.txt",
    ),
    (
        "mcp-builder",
        ROOT / "skills/mcp-builder/SKILL.md",
        TASK_PROMPT_ROOT / "mcp-builder-issue-tracker-server.txt",
    ),
    (
        "doc-coauthoring",
        ROOT / "skills/doc-coauthoring/SKILL.md",
        TASK_PROMPT_ROOT / "doc-coauthoring-retry-design-doc.txt",
    ),
    (
        "docx",
        ROOT / "skills/docx/SKILL.md",
        TASK_PROMPT_ROOT / "docx-convert-pdf-images.txt",
    ),
    (
        "paper-writing",
        ROOT
        / "skills/Auto-claude-code-research-in-sleep/skills/paper-writing/SKILL.md",
        TASK_PROMPT_ROOT / "paper-writing-segmentia-paper-plan.txt",
    ),
)

REQUIRED_MOUNT = Path("/mnt/990_pro")
EXPECTED_DEVICE = "/dev/nvme1n1"
