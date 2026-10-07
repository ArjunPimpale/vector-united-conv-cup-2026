"""One action per observation; diagnostics are exclusively on stderr."""
import json
import sys
import argparse

from .policy import Policy


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--experimental-setup',action='store_true')
    parser.add_argument('--diagnostics',action='store_true')
    parser.add_argument('--physical-only',action='store_true')
    parser.add_argument('--no-adaptation',action='store_true')
    parser.add_argument('--adaptive',action='store_true')
    args=parser.parse_args()
    policy = Policy(setup_enabled=args.experimental_setup)
    if args.physical_only:policy.coverage_enabled=policy.memory_enabled=policy.adaptation_enabled=False
    if args.adaptive:policy.adaptation_enabled=True
    if args.no_adaptation:policy.adaptation_enabled=False
    for line in sys.stdin:
        try:
            message = json.loads(line)
            if message.get("type") == "match_end":
                break
            if message.get("type") != "observation":
                continue
            action = policy.choose_action(message["observation"])
            if args.diagnostics:
                print(json.dumps({'iteration':message['observation']['state']['iteration'],
                                  **policy.diagnostic}),file=sys.stderr,flush=True)
        except Exception as error:
            print(f"policy error: {error}", file=sys.stderr, flush=True)
            action = {"move": "STAY"}
        print(json.dumps(action, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
