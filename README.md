# Vector United submission — Exact Planner v23

Selected competition submission: **Exact Planner v23**, supplied by our teammate/friend.
This release replaces the previous Vector United V5 submission. The planner source,
parameters, launch descriptor and requirements are byte-identical to the evaluated
`exact-planner-v23` release. No alternate bots or simulator files are required here.

## Requirements and organizer launch

Python **3.11 or newer**, standard library only. There are no packages to install,
model downloads, external APIs, network calls or credentials.

Use the root `submission.json` with the organizer's official runner. Set the bot's
working directory to this repository's root. The prescribed command is:

```bash
python -m team_bot.bot --model team_bot/models/params.json
```

The process waits for newline-delimited JSON observations on standard input,
returns one flushed action JSON line per observation on standard output, and exits
on `{"type":"match_end"}`. Diagnostics go to standard error. It controls only its
own player through the prescribed action interface. The organizer provides the
engine and fixed arena configuration separately.

## Included files

```text
submission.json
requirements.txt
README.md
V23_LOCK.json
team_bot/
  __init__.py
  bot.py
  policy.py
  frame.py
  physics.py
  planner.py
  opponent_model.py
  models/
    params.json
```

`bot.py` implements the process interface. `policy.py` selects possession, defense
and loose-ball behavior; `frame.py` transforms the observation into its planning
frame; `physics.py` predicts ball motion locally; `planner.py` evaluates actions;
`opponent_model.py` estimates opponent behavior. `params.json` contains the shipped
planner settings and is required by the launch command. The planner does not need
neural weights or the other development repository. Its local physics predictor
does not modify the official engine. `V23_LOCK.json` records runtime file checksums.

## Official validation and packaging

From the official participant kit, substitute the cloned repository's absolute
path for `/path/to/vector-united-conv-cup-2026`:

```bash
python validate_submission.py --submission /path/to/vector-united-conv-cup-2026/submission.json --matches-per-side 1 --seed 7000 --timeout 2
python package_submission.py /path/to/vector-united-conv-cup-2026 dist/exact-planner-v23.zip
python check_submission.py dist/exact-planner-v23.zip --report dist/exact-planner-v23-report.json
```

The ZIP places `submission.json` and `team_bot/` at its root. Keep generated ZIPs,
validation logs and the arena outside this repository. The official packager
excludes Git metadata and Python caches.

## Selection evidence

The bounded comparison used 48 new official-process games and 20 unchanged prior
V7 games. Directly against V7, v23 recorded **4 wins, 7 draws and 1 loss** across
six layouts with both sides tested. Against the shared rival panel it recorded
**9 wins, 18 draws and 1 loss** in 28 matches. It was selected for defensive
resilience; frequent draws and its observed V5 weakness remain limitations.
These results do not guarantee performance against unseen entrants or establish
championship odds. The organizers determine the actual tournament and tie breaks.
