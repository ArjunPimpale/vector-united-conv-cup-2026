# Team IDK — Exact Planner v23

Conv-Cup '26 submission of team **IDK**: Arjun Pimpale (team leader) and Rohan Patil.
Selected bot: **Exact Planner v23**, developed by team member Rohan Patil.
It replaces the team's previous submission, Vector United V5. The planner source,
parameters, requirements and launch command are byte-identical to the evaluated
`exact-planner-v23` release. `submission.json` differs only in its public team name,
`IDK`. No alternate bots or simulator files are required here.

## Requirements and organizer launch

Python **3.11 or newer**, standard library only. There are no packages to install,
model downloads, external APIs, network calls or credentials.

Use the root `submission.json` with the organizer's official runner. Set the bot's
working directory to this repository's root. The prescribed command is:

```bash
python -m team_bot.bot --model team_bot/models/params.json
```

The process waits for newline-delimited JSON observations on standard input and
returns one flushed action JSON line per observation on standard output. A typical
decision takes under 20 ms. If a decision ever took longer than 1.7 s, the bot would
send nothing for that turn, so the runner's STAY substitution keeps the stream in
step. It exits on `{"type":"match_end"}`. Diagnostics go to standard error. It
controls only its own player through the prescribed action interface. The organizer
provides the engine and the fixed arena configuration separately.

## How it decides

Each observation is turned into a fixed frame in which we always attack up the
field. The bot then chooses a move (and, when it holds the ball, a kick) for this
turn only:

- **With the ball:** it searches every move x kick direction x power, using a local
  replica of the published ball physics. It shoots when it can prove the opponent
  cannot reach the ball's path before it crosses the line, or when its model of this
  opponent predicts no block. Otherwise it dribbles or passes toward space that it
  reaches before the opponent.
- **Without the ball:** it positions between the ball and its own goal, covering the
  lanes the opponent could actually shoot along, and tackles when eligible.
- **Loose ball:** it races for the ball when it expects to arrive first. Otherwise it
  holds the lane to its goal.
- **Breaking deadlocks:** when the score is level or behind and play stops gaining
  ground for 80 turns, it carries the ball toward the best shooting spot, or presses
  the opponent's carrier. After a goal the kickoff position is always the same, so a
  short sequence that scored from a kickoff is repeated while every observation
  matches it exactly. It is dropped at the first difference, and forgotten if it ever
  ends in a goal against us.

The bot uses only the fields of each observation and its memory of earlier turns in
the same match. It never modifies the game state or the engine. It has no random
number generator, so the same seed and actions always reproduce the same match.

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

- `bot.py` implements the process interface.
- `policy.py` selects possession, defense and loose-ball behavior.
- `frame.py` transforms the observation into its planning frame.
- `physics.py` predicts ball motion locally. This local predictor does not modify
  the official engine.
- `planner.py` evaluates actions.
- `opponent_model.py` estimates opponent behavior.
- `params.json` contains the shipped planner settings. Always launch with it: if it
  cannot be read, the bot falls back to weaker built-in defaults instead of stopping.
- `V23_LOCK.json` records runtime file checksums.

The planner does not need neural weights or the other development repository.

## Official validation and packaging

From the official participant kit, substitute the cloned repository's absolute
path for `/path/to/vector-united-conv-cup-2026`:

```bash
python validate_submission.py --submission /path/to/vector-united-conv-cup-2026/submission.json --matches-per-side 2 --seed 7000 --timeout 2
python package_submission.py /path/to/vector-united-conv-cup-2026 dist/idk-exact-planner-v23.zip
python check_submission.py dist/idk-exact-planner-v23.zip --report dist/idk-exact-planner-v23-report.json
```

The ZIP places `submission.json` and `team_bot/` at its root. It holds 12 files,
including this README and `V23_LOCK.json`, so it is not byte-identical to the
evaluated release archive. Record the SHA-256 that the checker prints. Keep
generated ZIPs, validation logs and the arena outside this repository.

The official packager excludes Git metadata and Python caches but includes every
other file. The checker rejects files whose type is not on the kit's allowed list,
so do not add files such as `.gitignore` or `LICENSE` here.

## Selection evidence

From the team leader's own comparison:

> The bounded comparison used 48 new official-process games and 20 unchanged prior
> V7 games. Directly against V7, v23 recorded **4 wins, 7 draws and 1 loss** across
> six layouts with both sides tested. Against the shared rival panel it recorded
> **9 wins, 18 draws and 1 loss** in 28 matches. It was selected for defensive
> resilience; frequent draws and its observed V5 weakness remain limitations.

From the developer's own holdout, with every opponent's clock frozen.
It used 360 paired matches on fresh seeds against the previous v22 build, at the
shipped time budget, with both sides played. v23 took **79.0%** of available points
(v22: 75.0%; paired difference +4.0, 95% CI +2.1 to +6.1):

| Opponent | v23 share of points |
|---|---|
| Our earlier V3 bot | 57.5% |
| Our earlier V5 bot | 49.2% |
| idk-release4 bot | 84.2% |
| Earlier v16 build | 85.8% |
| Kit RL bot | 100% |
| Kit reference bot | 97.5% |

The official validator reported zero action errors.

These results do not guarantee performance against unseen entrants or establish
championship odds. The organizers determine the actual tournament and tie breaks.
