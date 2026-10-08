"""JSON-lines protocol wrapper.

Contract (from ``soccer_env.match_runner.AgentProcess``):
  * one JSON object per line on stdin, one action line on stdout, flushed;
  * a reply over 8192 bytes, invalid JSON, EOF or a 2.0 s timeout all become
    ``{"move": "STAY"}`` plus a counted action error;
  * a *late* line is never drained -- ``asyncio.wait_for`` cancels the read but
    the stream buffer survives, so the line is served as the answer to the next
    observation and the agent stays one step behind for the rest of the match.

So once the deadline has passed we print nothing at all: STAY has already been
substituted and silence keeps the pipe in sync. Diagnostics go to stderr only.
"""
from __future__ import annotations

import argparse
import json
import sys
from time import perf_counter

from .policy import Policy

HARD_DEADLINE = 1.70   # seconds; the runner's limit is 2.0
SAFE_ACTION = '{"move":"STAY"}'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    try:
        policy = Policy.load(args.model)
    except Exception as error:
        print(f"planner: falling back to defaults: {error}", file=sys.stderr, flush=True)
        policy = Policy()

    write = sys.stdout.write
    flush = sys.stdout.flush

    for line in sys.stdin:
        received = perf_counter()
        try:
            message = json.loads(line)
            kind = message.get("type")
            if kind == "match_end":
                return
            if kind != "observation":
                continue
            try:
                action = policy.choose_action(message["observation"])
                payload = json.dumps(action, separators=(",", ":"))
                if len(payload) > 4000:
                    payload = SAFE_ACTION
            except Exception as error:
                print(f"planner decision error: {error}", file=sys.stderr, flush=True)
                payload = SAFE_ACTION
            if perf_counter() - received > HARD_DEADLINE:
                # Too late to be read as this turn's answer; stay silent.
                print("planner: dropped a late reply", file=sys.stderr, flush=True)
                continue
            write(payload)
            write("\n")
            flush()
        except Exception as error:
            print(f"planner protocol error: {error}", file=sys.stderr, flush=True)
            if perf_counter() - received <= HARD_DEADLINE:
                write(SAFE_ACTION)
                write("\n")
                flush()


if __name__ == "__main__":
    main()
