# Project Management Framework Instructions

## IMPORTANT: Date & Time

- Before adding Date/Time to any file make sure you are aware of the current system time (system `date` command) and use 24h time format.
- When Using Timestamps use YYYY-MM-DD HH:MM format, for Date only use YYYY-MM-DD

## Simone Project Management Overview

This project uses the Simone framework for project management (spec at https://github.com/Helmi/claude-simone). Key operational files are:

- Project status: `quactuary/.project_management/00_PROJECT_MANIFEST.md`

## Directory Structure

```plaintext
quactuary/.project_management/
├── 00_PROJECT_MANIFEST.md       # Project status and pointers
├── 01_PROJECT_DOCS/             # General Documentation to understand the Project
├── 02_REQUIREMENTS/             # Milestone-based requirements in subfolders
├── 03_SPRINTS/                  # Sprint plans and tasks
├── 04_GENERAL_TASKS/            # Non-sprint tasks
├── 05_ARCHITECTURE_DECISIONS/   # ADRs
├── 10_STATE_OF_PROJECT/         # Project review snapshots and state tracking
└── 99_TEMPLATES/                # Templates
```

## File Naming Conventions

- Milestones: `M<NN>_<Description>`
- Sprints: `S<NN>_M<NN>_<Focus_Area>`
- Sprint Tasks: `T<NN>_S<NN>_<Description>.md`
- Completed Sprint Tasks: `TX<NN>_S<NN>_<Description>.md`
- General Tasks: `T<NNN>_<Description>.md`
- Completed Tasks: `TX<NNN>_<Description>.md`
- ADRs: `ADR<NNN>_<Title>.md`
- PRD Amendments: `PRD_AMEND_<NN>_<Description>.md`

## General Guidelines

- Always use templates from `quactuary/.project_management/99_TEMPLATES/` as structural guides
- Flag unclear requirements or overly complex tasks for human review and actively ask the user.
- Be concise in logs and communications and be aware of current date and time
- Update project manifest when significant changes occur (especially on Tasks, Sprints or Milestones completed)
