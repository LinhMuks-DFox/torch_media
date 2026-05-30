# Progress NN — <imperative title>
id: <YYYY-MM-DD>/progressNN
date: <YYYY-MM-DD>
author: <human | ai | human+ai>
status: active            # active | blocked | done
refs: []                  # related progress/task ids
supersedes:               # optional: id of a decision this replaces
commits: []               # related commit hashes (fill as work lands)
files:                    # source files this progress touches/affects
  - <path>

## Goal
<1-2 sentences: what this progress aims to achieve, and the boundary.>

## Context / Motivation
<Why now. Link to richer notes/refs instead of re-explaining them here.>

## Decisions

### D1 — <short title>
- Decision: <the chosen direction, stated as an implementation constraint>
- Why: <engineering rationale>
- Impact: <files / APIs / build affected; what to preserve or avoid>
- Alternatives considered: <option — why not> (omit if none)

## Tasks
- [ ] [taskNN — <title>](tasks/taskNN-<slug>.md)

## Issues / Gotchas
<bugs hit, tricky implementation details discovered during dev>

## Open / TODO (carry-over)
<unfinished items deferred to a later progress>

## Agent log
- <YYYY-MM-DD> [ai] <what was done / produced / next step>
