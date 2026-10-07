# Vector United — Conv-Cup 2026

Final submission: **Vector United v5**. This repository contains the standalone
bot for the official AI Soccer Arena JSON-lines process interface. The Python
source and launch descriptor are unchanged from the evaluated V5 release.
No alternative bot versions are included.

## Requirements

- Python **3.11 or newer**.
- Python standard library only; no packages need to be installed.
- No model download, external service, internet connection or credentials.
- No simulator installation is required inside this repository. The organizer
  supplies the official engine and match runner separately.

This is a code-only planning policy. All policy logic is under `team_bot/`;
there are no external weights or missing model files.

## Organizer launch

Clone the repository and use its root `submission.json` with the official runner.
The runner must use the cloned repository root as the bot's working directory.
The descriptor specifies the exact final command:

```json
{
  "name": "Vector United v5",
  "command": ["python3", "-m", "team_bot.bot"]
}
```

For direct launch from this repository root:

```bash
python3 -m team_bot.bot
```

The process waits for the organizer's observation messages on standard input.
It is not a graphical simulator and produces no action until an observation
arrives. It reads one newline-delimited JSON observation at a time, writes one
action JSON line to standard output, flushes immediately, and exits on
`{"type":"match_end"}`. Diagnostics use standard error. Launch with no extra
flags to preserve the submitted V5 behavior.

## Required repository structure

```text
README.md
requirements.txt
submission.json
V5_LOCK.json
team_bot/
  __init__.py
  bot.py
  policy.py
  ... all supporting Python modules
```

`submission.json` is at the repository root. No outer `versions/v5/` directory
or nested ZIP needs to be located or extracted. `V5_LOCK.json` records SHA-256
checksums for the frozen runtime files and the previously evaluated V5 archive.
The launch command does not depend on any files outside this repository.

## Official validation and packaging

From the organizer's supplied `participants/` directory, replace
`/absolute/path/to/vector-united-conv-cup-2026` below with the cloned repository
path. These scripts belong to the official kit; they are not bot dependencies.

```bash
python3 validate_submission.py --submission /absolute/path/to/vector-united-conv-cup-2026/submission.json --matches-per-side 2 --seed 7000 --timeout 2
python3 package_submission.py /absolute/path/to/vector-united-conv-cup-2026 dist/vector-united-v5.zip
python3 check_submission.py dist/vector-united-v5.zip --report dist/vector-united-v5-report.json
```

Validation must finish on both sides with zero participant action errors, and
the archive checker must report `PASS`. The official packager puts the contents
of this repository at the ZIP root and excludes `.git` and Python caches.
Store the ZIP and validation outputs outside this repository. Do not include
virtual environments, logs, credentials, training data or other bots.

The exact original V5 policy was evaluated in 1,300 official process games
against thirteen rivals, over fifty obstacle layouts and both sides, with zero
participant action errors. The original V5 archive also passed the official
static checker and an extracted-archive process match on each side. Packaging
checks do not count toward the competitive evaluation results.

## Publish this folder

Create a GitHub repository named `vector-united-conv-cup-2026` accessible to the
organizers. Initialize Git in **this folder**. Explicitly stage the submission
files to avoid accidentally committing generated caches:

```bash
git init -b main
git add README.md requirements.txt submission.json V5_LOCK.json team_bot/*.py
git commit -m "Submit frozen Vector United v5 for Conv-Cup 2026"
git remote add origin https://github.com/YOUR_USERNAME/vector-united-conv-cup-2026.git
git push -u origin main
```

Submit the repository URL requested by the form. Keep this V5 source and
`submission.json` fixed after submission.
