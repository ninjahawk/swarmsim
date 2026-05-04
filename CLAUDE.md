# Summer Research — Claude Context

## Project
PHY 351 Independent Summer Research  
Student: Nathan Langley  
Professor: Ian Beatty (ian@ianbeatty.com)  
Textbook: *Natural Complexity: A Modeling Handbook* by Paul Charbonneau (Princeton University Press, 2017)

## Key Files
- `logs.html` — open in Chrome to view/copy the time log and research log. This is the main interface for logging. Do not replace it with a different solution.
- `README.md` — GitHub repo overview
- `research_log.md` — placeholder, actual log data lives in `logs.html`

## GitHub
Remote: https://github.com/ninjahawk/Summer_Research.git  
Branch: main  
Share the repo URL with the professor for access — no collaborator invite needed.

## Logging Workflow
All logging is done by editing `logs.html` directly:
- **Time log** — add rows to `#timeBody` as `<tr data-date="YYYY-MM-DD">` with date, hours (decimal), summary
- **Research log** — add objects to the `researchData` array with `date` and `body` fields
- After any update, commit and push to GitHub

## Log Format
Time log hours are decimal rounded to nearest 0.25 (e.g. 30 min = 0.50).  
Research log entries are plain text, concise, first person.  
Header in research log: PHY 351 / Independent Summer Research / Nathan Langley — do not change.

## Rules from Professor
- Document all AI use in the research log (general purpose, not every prompt)
- Nathan must understand and own all decisions — "Claude recommended it" is not sufficient
- Share time log with professor via Google Sheets (copy from logs.html), research log via Google Docs

## Google Sheets / Docs
No API integration. Nathan copies from `logs.html` in Chrome and pastes manually into Google Sheets and Google Docs. Time log copies as a table (splits into columns on paste). Research log copies as plain text.

## Topic
Not yet selected. Professor suggested Chapter 6 (Forest Fires) as default. Nathan is deciding by Monday 2026-05-05. Other options: Sandpiles, Traffic Jams, Epidemics, Flocking, Pattern Formation, Percolation, Aggregation.
