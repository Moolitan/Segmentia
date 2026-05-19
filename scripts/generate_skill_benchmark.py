from pathlib import Path
import json

ROOT = Path("anthropic_skill_benchmark_8_repos")

FILES = {
    # =========================================================
    # 1) doc_coauthoring_design_doc
    # =========================================================
    "doc_coauthoring_design_doc/README.md": """# doc_coauthoring_design_doc

## Goal
An 8-turn structured design-doc coauthoring task.

## Expected behavior
- Read the provided system background and failure examples
- Collaboratively plan and draft a short design doc
- Use a structured doc-writing workflow
- Refine the draft and perform a reviewer-style check at the end

## Context mode
seed_files

## Primary output
design_doc.md
""",

    "doc_coauthoring_design_doc/task.json": json.dumps({
        "repo_name": "doc_coauthoring_design_doc",
        "turn_count": 8,
        "skill_class": "1_skill",
        "context_mode": "seed_files",
        "expected_skills": ["doc-coauthoring"],
        "primary_outputs": ["design_doc.md"]
    }, indent=2),

    "doc_coauthoring_design_doc/expected_skills.json": json.dumps({
        "turn_1": ["doc-coauthoring"],
        "turn_2": ["doc-coauthoring"],
        "turn_3": ["doc-coauthoring"],
        "turn_4": ["doc-coauthoring"],
        "turn_5": ["doc-coauthoring"],
        "turn_6": ["doc-coauthoring"],
        "turn_7": ["doc-coauthoring"],
        "turn_8": ["doc-coauthoring"]
    }, indent=2),

    "doc_coauthoring_design_doc/seed_files/system_background.md": """# Experiment Platform Background

The platform is used to run batches of agent benchmark tasks.
Each task is launched by a scheduler and writes outputs to a results directory when finished.

Current failure handling:
- If a task crashes or a tool call fails, the task is marked as failed
- A user must manually relaunch the task
- There is no automatic retry mechanism
- A rerun usually creates a new run record, and linkage to the original failed run is weak

Known issues:
1. Temporary network hiccups can cause otherwise recoverable tasks to fail
2. Some tool calls occasionally time out, but succeed when rerun manually
3. Fully automatic unlimited retries would waste resources and may hide real bugs
""",

    "doc_coauthoring_design_doc/seed_files/failure_examples.md": """# Failure Examples

## Example 1
- Task type: web retrieval
- Failure reason: external page load timeout
- Manual rerun result: success

## Example 2
- Task type: file generation
- Failure reason: transient I/O error
- Manual rerun result: success

## Example 3
- Task type: code execution
- Failure reason: script logic bug
- Manual rerun result: still fails
""",

    "doc_coauthoring_design_doc/turns/turn_1.txt": """Please first read seed_files/system_background.md and seed_files/failure_examples.md.

Help me co-write a short design doc about adding automatic retry for failed benchmark tasks.
Start by guiding the structure and thinking process rather than writing the whole document at once.
""",

    "doc_coauthoring_design_doc/turns/turn_2.txt": """The target readers are teammates working on agent infrastructure.
Keep the total length within about 1.5 pages.

Please propose an outline that includes at least:
- Background
- Goals
- Proposed approach
- Risks
- Open questions
""",

    "doc_coauthoring_design_doc/turns/turn_3.txt": """Now write the "Background" and "Goals" sections.
Emphasize why automatic retry is needed now.
""",

    "doc_coauthoring_design_doc/turns/turn_4.txt": """Now write the "Proposed approach" and "Risks" sections.
Please explain why we should not do unlimited retries, and how to distinguish retryable vs non-retryable failures.
""",

    "doc_coauthoring_design_doc/turns/turn_5.txt": """Please revise the retry policy so it is more concrete.
Include:
- max retry count
- retry delay strategy
- what should be recorded in logs
""",

    "doc_coauthoring_design_doc/turns/turn_6.txt": """Now add a short rollout section.
How should we introduce this safely without affecting all workloads at once?
""",

    "doc_coauthoring_design_doc/turns/turn_7.txt": """Please review the full draft from a reviewer's perspective.
Point out 3 places that are still unclear or underspecified.
""",

    "doc_coauthoring_design_doc/turns/turn_8.txt": """Finally, provide:
1. the final concise design doc
2. a 5-bullet summary for teammates who will not read the full document
""",

    # =========================================================
    # 2) internal_comms_incident_update
    # =========================================================
    "internal_comms_incident_update/README.md": """# internal_comms_incident_update

## Goal
An 8-turn internal communications task centered on an internal service incident update.

## Expected behavior
- Read the factual incident notes
- Produce a calm internal update
- Tailor the message for leadership readability
- Add FAQ and final broadcast-ready text

## Context mode
seed_files

## Primary output
incident_update.md
""",

    "internal_comms_incident_update/task.json": json.dumps({
        "repo_name": "internal_comms_incident_update",
        "turn_count": 8,
        "skill_class": "1_skill",
        "context_mode": "seed_files",
        "expected_skills": ["internal-comms"],
        "primary_outputs": ["incident_update.md"]
    }, indent=2),

    "internal_comms_incident_update/expected_skills.json": json.dumps({
        "turn_1": ["internal-comms"],
        "turn_2": ["internal-comms"],
        "turn_3": ["internal-comms"],
        "turn_4": ["internal-comms"],
        "turn_5": ["internal-comms"],
        "turn_6": ["internal-comms"],
        "turn_7": ["internal-comms"],
        "turn_8": ["internal-comms"]
    }, indent=2),

    "internal_comms_incident_update/seed_files/incident_facts.md": """# Incident Facts

Time:
- Queue buildup started at around 09:20
- Service recovered at around 09:45

Initial suspected cause:
- Worker auto-scaling did not trigger as expected

Impact:
- 12 internal experiment tasks were delayed
- No external users were affected
- There is no sign of data loss

Current status:
- The service has recovered
- The team is checking scaling trigger logic and alert thresholds
""",

    "internal_comms_incident_update/turns/turn_1.txt": """Please first read seed_files/incident_facts.md.

Help me write an internal incident update.
Use a calm and objective tone.
""",

    "internal_comms_incident_update/turns/turn_2.txt": """Add a short "Impact" section and a short "Current status" section.
""",

    "internal_comms_incident_update/turns/turn_3.txt": """Now revise it so it is more suitable for leadership readers and contains fewer implementation details.
""",

    "internal_comms_incident_update/turns/turn_4.txt": """Add 4 FAQ items that people are likely to ask:
- Could this happen again?
- Was any data lost?
- Is everything recovered now?
- What will we do to prevent this next time?
""",

    "internal_comms_incident_update/turns/turn_5.txt": """Please also make a version that is suitable for a daily internal status digest.
It should be shorter and more summary-like.
""",

    "internal_comms_incident_update/turns/turn_6.txt": """Now add a brief section on next actions for the engineering team.
Keep it practical and not overly detailed.
""",

    "internal_comms_incident_update/turns/turn_7.txt": """Please rewrite the update so it feels more polished and consistent across:
- summary
- impact
- status
- next steps
- FAQ
""",

    "internal_comms_incident_update/turns/turn_8.txt": """Finally, provide:
1. one leadership-facing version
2. one internal chat version
3. one very short 3-sentence summary
""",

    # =========================================================
    # 3) web_artifact_with_theme
    # =========================================================
    "web_artifact_with_theme/README.md": """# web_artifact_with_theme

## Goal
An 8-turn artifact-building task that first creates a landing page, then applies a coherent visual theme.

## Expected behavior
- Read the event brief
- Build a structured landing page artifact
- Add more sections over turns
- Apply a consistent theme and improve hierarchy

## Context mode
seed_files

## Primary output
event_landing_page
""",

    "web_artifact_with_theme/task.json": json.dumps({
        "repo_name": "web_artifact_with_theme",
        "turn_count": 8,
        "skill_class": "2_skills",
        "context_mode": "seed_files",
        "expected_skills": ["web-artifacts-builder", "theme-factory"],
        "primary_outputs": ["event_landing_page"]
    }, indent=2),

    "web_artifact_with_theme/expected_skills.json": json.dumps({
        "turn_1": ["web-artifacts-builder"],
        "turn_2": ["web-artifacts-builder"],
        "turn_3": ["web-artifacts-builder"],
        "turn_4": ["theme-factory"],
        "turn_5": ["web-artifacts-builder", "theme-factory"],
        "turn_6": ["web-artifacts-builder"],
        "turn_7": ["theme-factory"],
        "turn_8": ["web-artifacts-builder", "theme-factory"]
    }, indent=2),

    "web_artifact_with_theme/seed_files/event_brief.md": """# Event Brief

Event name: Lab Tech Sharing Night
Time: Next Friday, 18:30 - 20:30
Location: Engineering Building, Room 301

The page should include:
- Event title and subtitle
- Time and location
- Registration button
- Short agenda
- Speaker section
- FAQ section

Style requirements:
- Clean
- Professional
- Technical-community feeling
- Not flashy
""",

    "web_artifact_with_theme/turns/turn_1.txt": """Please first read seed_files/event_brief.md.

Build an internal tech-event landing page that includes:
- title
- time
- location
- registration button
- a short agenda

It should feel like a proper event landing page, not just a plain demo.
""",

    "web_artifact_with_theme/turns/turn_2.txt": """Now add a speaker section.
Each speaker should have an avatar placeholder, name, and short bio.
""",

    "web_artifact_with_theme/turns/turn_3.txt": """Add an FAQ section with 4 placeholder questions.
""",

    "web_artifact_with_theme/turns/turn_4.txt": """Now apply a unified theme to the whole page.
The theme should feel clean, professional, and suitable for an internal technical event.
""",

    "web_artifact_with_theme/turns/turn_5.txt": """Refine the typography and color hierarchy so that the page sections, headings, and buttons feel more consistent and readable.
""",

    "web_artifact_with_theme/turns/turn_6.txt": """Please add a small registration-flow hint section near the call-to-action.
It should briefly explain what happens after someone clicks the registration button.
""",

    "web_artifact_with_theme/turns/turn_7.txt": """Please refine the visual theme further so the page feels slightly warmer and more welcoming,
while still staying professional and technical.
""",

    "web_artifact_with_theme/turns/turn_8.txt": """Finally, provide the finished landing page plus a short note summarizing:
- the main sections
- the theme direction
- the key design choices you made
""",

    # =========================================================
    # 4) mcp_server_and_spec
    # =========================================================
    "mcp_server_and_spec/README.md": """# mcp_server_and_spec

## Goal
An 8-turn task that plans an MCP server and then produces a short supporting spec.

## Expected behavior
- Read the issue API brief
- Plan tool surface and parameters
- Sketch minimal project structure
- Write a short spec
- Review the plan for likely real-world agent pitfalls

## Context mode
seed_files

## Primary output
mcp_server_spec.md
""",

    "mcp_server_and_spec/task.json": json.dumps({
        "repo_name": "mcp_server_and_spec",
        "turn_count": 8,
        "skill_class": "2_skills",
        "context_mode": "seed_files",
        "expected_skills": ["mcp-builder", "doc-coauthoring"],
        "primary_outputs": ["mcp_server_spec.md"]
    }, indent=2),

    "mcp_server_and_spec/expected_skills.json": json.dumps({
        "turn_1": ["mcp-builder"],
        "turn_2": ["mcp-builder"],
        "turn_3": ["mcp-builder"],
        "turn_4": ["doc-coauthoring"],
        "turn_5": ["mcp-builder", "doc-coauthoring"],
        "turn_6": ["mcp-builder"],
        "turn_7": ["doc-coauthoring"],
        "turn_8": ["mcp-builder", "doc-coauthoring"]
    }, indent=2),

    "mcp_server_and_spec/seed_files/issue_api_brief.md": """# Issue API Brief

Goal:
Build an MCP server for a simple issue-tracking service.

Available core REST APIs:
- GET /issues/{id}                  fetch a single issue
- GET /issues?status=&assignee=     filter issues
- POST /issues                      create an issue
- POST /issues/{id}/comments        add a comment to an issue

Minimum requirements:
1. The agent can query issues
2. The agent can create issues
3. The agent can add comments to issues

Constraints:
- No attachment upload for now
- No bulk modification for now
- No user-permission integration for now
""",

    "mcp_server_and_spec/turns/turn_1.txt": """Please first read seed_files/issue_api_brief.md.

I want to build an MCP server for this issue-tracking API.
First, help me decide what tools the MCP server should expose.
Do not go too deep into code yet.
""",

    "mcp_server_and_spec/turns/turn_2.txt": """The minimum tool set is:
- query issue
- create issue
- add comment to issue

Please think about parameter design for each tool.
""",

    "mcp_server_and_spec/turns/turn_3.txt": """Now sketch a minimal project structure, assuming the implementation will be in Python.
""",

    "mcp_server_and_spec/turns/turn_4.txt": """Based on the plan so far, write a short spec that includes:
- goal
- tool surface
- error handling principles
""",

    "mcp_server_and_spec/turns/turn_5.txt": """Now review this spec and point out 3 places where a real agent might still run into trouble when using the MCP server.
""",

    "mcp_server_and_spec/turns/turn_6.txt": """Please now think about pagination and filtering behavior.
I want the MCP server to stay usable even if the issue list becomes large.
""",

    "mcp_server_and_spec/turns/turn_7.txt": """Please revise the spec so it reads like a more realistic implementation note for teammates who may build it later.
""",

    "mcp_server_and_spec/turns/turn_8.txt": """Finally, provide:
1. the final concise spec
2. a compact implementation checklist
3. the top 3 design decisions that matter most
""",

    # =========================================================
    # 5) launch_poster_page_pack
    # =========================================================
    "launch_poster_page_pack/README.md": """# launch_poster_page_pack

## Goal
An 8-turn multi-artifact launch pack task: create a static poster, build a simple page, and unify them under one theme.

## Expected behavior
- Read the open day brief
- Create a poster concept/output
- Build a short web page
- Unify poster and page visually
- Check consistency at the end

## Context mode
seed_files

## Primary outputs
- open_day_poster
- open_day_page
""",

    "launch_poster_page_pack/task.json": json.dumps({
        "repo_name": "launch_poster_page_pack",
        "turn_count": 8,
        "skill_class": "3_skills",
        "context_mode": "seed_files",
        "expected_skills": ["canvas-design", "web-artifacts-builder", "theme-factory"],
        "primary_outputs": ["open_day_poster", "open_day_page"]
    }, indent=2),

    "launch_poster_page_pack/expected_skills.json": json.dumps({
        "turn_1": ["canvas-design"],
        "turn_2": ["web-artifacts-builder"],
        "turn_3": ["web-artifacts-builder"],
        "turn_4": ["theme-factory"],
        "turn_5": ["canvas-design", "web-artifacts-builder", "theme-factory"],
        "turn_6": ["canvas-design"],
        "turn_7": ["web-artifacts-builder"],
        "turn_8": ["canvas-design", "web-artifacts-builder", "theme-factory"]
    }, indent=2),

    "launch_poster_page_pack/seed_files/open_day_brief.md": """# Open Day Brief

Event: Lab Open Day
Target audience: students at the university who are interested in systems and AI
Time: April 20, 14:00 - 17:00
Location: Innovation Center, Floor 2

Deliverables:
1. A static poster suitable for sharing on social platforms
2. A simple web page showing the event intro, agenda, and registration entry

Style:
- modern
- clean
- academic
- not too commercial
""",

    "launch_poster_page_pack/turns/turn_1.txt": """Please first read seed_files/open_day_brief.md.

Start by creating a poster concept for this event.
The style should feel modern, clean, and academic.
""",

    "launch_poster_page_pack/turns/turn_2.txt": """Now create a short web page for the same event.
It should include:
- event introduction
- agenda
- registration entry
""",

    "launch_poster_page_pack/turns/turn_3.txt": """Make the page feel like a formal research-event page rather than a flashy marketing page.
""",

    "launch_poster_page_pack/turns/turn_4.txt": """Now unify the poster and the page under one visual theme.
Please make the colors and typography feel consistent.
""",

    "launch_poster_page_pack/turns/turn_5.txt": """Finally, check whether the poster and the web page really look like they belong to the same event pack, and keep the poster as a shareable static final asset.
""",

    "launch_poster_page_pack/turns/turn_6.txt": """Please refine the poster so that the hierarchy is clearer at a glance.
The event name, time, and location should feel easier to scan.
""",

    "launch_poster_page_pack/turns/turn_7.txt": """Now improve the web page so the agenda and registration entry feel more polished and better aligned with the poster.
""",

    "launch_poster_page_pack/turns/turn_8.txt": """Finally, provide:
1. the final poster asset
2. the final event page
3. a short summary of the visual system tying them together
""",

    # =========================================================
    # 6) slack_launch_pack
    # =========================================================
    "slack_launch_pack/README.md": """# slack_launch_pack

## Goal
An 8-turn internal launch-pack task: create an announcement, add a Slack GIF concept/output, then align everything with brand style.

## Expected behavior
- Read the feature brief
- Draft an internal launch announcement
- Adapt it for Slack
- Create a Slack GIF concept/output
- Align copy and visuals with brand notes

## Context mode
seed_files

## Primary outputs
- launch_announcement.md
- launch_gif
- final_launch_pack.md
""",

    "slack_launch_pack/task.json": json.dumps({
        "repo_name": "slack_launch_pack",
        "turn_count": 8,
        "skill_class": "3_skills",
        "context_mode": "seed_files",
        "expected_skills": ["internal-comms", "slack-gif-creator", "brand-guidelines"],
        "primary_outputs": ["launch_announcement.md", "launch_gif", "final_launch_pack.md"]
    }, indent=2),

    "slack_launch_pack/expected_skills.json": json.dumps({
        "turn_1": ["internal-comms"],
        "turn_2": ["internal-comms"],
        "turn_3": ["slack-gif-creator"],
        "turn_4": ["brand-guidelines"],
        "turn_5": ["internal-comms", "slack-gif-creator", "brand-guidelines"],
        "turn_6": ["internal-comms"],
        "turn_7": ["slack-gif-creator"],
        "turn_8": ["internal-comms", "slack-gif-creator", "brand-guidelines"]
    }, indent=2),

    "slack_launch_pack/seed_files/feature_launch_brief.md": """# Feature Launch Brief

New feature:
The experiment task dashboard now supports retry for failed tasks.

Main benefits:
- reduces manual reruns
- improves success rate for long-running tasks
- makes it easier to distinguish retryable vs non-retryable failures

Audience:
- internal research teammates
- platform engineering teammates

Tone:
- positive
- concise
- not overly promotional
""",

    "slack_launch_pack/seed_files/brand_notes.md": """# Brand Notes

Style requirements:
- simple
- clear
- trustworthy
- technical-team tone
- not flashy

Visual direction:
- calm colors
- simple shapes
- avoid excessive exclamation marks or hype language
""",

    "slack_launch_pack/turns/turn_1.txt": """Please first read seed_files/feature_launch_brief.md.

We want to announce a new internal feature launch:
the experiment task dashboard now supports retry for failed tasks.

Please draft a short internal announcement.
""",

    "slack_launch_pack/turns/turn_2.txt": """Now adapt it so it is more suitable for Slack.
Make the tone a little lighter, but still professional.
""",

    "slack_launch_pack/turns/turn_3.txt": """I also want a Slack GIF for this launch.
The theme should be "a failed task gets back up and tries again".
Keep it a little playful.
""",

    "slack_launch_pack/turns/turn_4.txt": """Please also read seed_files/brand_notes.md.

Now make the GIF direction and the announcement feel like they belong to the same launch pack.
""",

    "slack_launch_pack/turns/turn_5.txt": """Finally, tighten everything according to the brand notes so that the wording, colors, and overall style are more consistent.
Also output a short final launch-pack summary.
""",

    "slack_launch_pack/turns/turn_6.txt": """Please make the announcement slightly clearer for platform engineers who care about what the feature actually does.
Keep it short, but add a bit more substance.
""",

    "slack_launch_pack/turns/turn_7.txt": """Now refine the GIF direction so it feels more polished and less like a generic recovery animation.
It should still work well for Slack.
""",

    "slack_launch_pack/turns/turn_8.txt": """Finally, provide a polished final version suitable for actual internal rollout.
It should include:
- the main Slack announcement
- a shorter fallback version
- the GIF concept or output
- a brief note on how the pieces fit together
""",
}

def main():
    ROOT.mkdir(exist_ok=True)
    for rel_path, content in FILES.items():
        path = ROOT / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(f"Created benchmark repos under: {ROOT.resolve()}")

if __name__ == "__main__":
    main()