# develop_log

Engineering progress log for torch_media. It lives in the main repo so progress entries stay
linked to the code and commits they describe.

## Layout

```
develop_log/
  README.md
  templates/
    progress.template.md
    task.template.md
  <YYYY-MM-DD>/                 # zero-padded date, e.g. 2026-05-30
    progressNN-<slug>.md        # one per topic; multiple per day allowed
    tasks/
      taskNN-<slug>.md          # executable work item, child of a progress
```

## Units

- A **progress** = one topic / unit of work (may span multiple days). It records the goal,
  key decisions + rationale, child tasks, gotchas, and carry-over TODOs.
- A **task** = a strict, executable work item for the implementer, child of exactly one progress.

## Naming & IDs

- Files: `progress01-remove-ffmpeg-dr_wav-io.md`, `tasks/task01-vendor-dr_wav-wav-path.md`.
- `id`: `<date>/progressNN` and `<date>/taskNN` — the stable handle used for cross-references.

## Cross-references

- A progress/task lists related ids in `refs:`.
- In prose, link with a relative path: `[progress02](./progress02-....md)`.
- If a progress replaces an earlier decision, set `supersedes: <id>`.

## status values

`active` · `blocked` · `done`

## Cross-day rule

A topic that continues across days stays in its original progress file — update `status` and
append to `Agent log`. Start a new progress only for a genuinely new direction, and link it via
`refs` / `supersedes`. Do not split one topic across several date folders.

## Language

English (these are code-agent-facing documents). Conversation with the human is in Chinese.

## Templates

Copy from `templates/progress.template.md` and `templates/task.template.md`.
