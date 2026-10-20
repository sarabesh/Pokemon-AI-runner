"""Random baseline: mashes buttons and reports the same metrics every other agent is scored on.

    .venv/Scripts/python random_agent.py roms/pokemon_red.gb --steps 2000
    .venv/Scripts/python random_agent.py roms/pokemon_red.gb --state runs/discovery/states/after_intro.state \
        --controls runs/discovery/controls.json --seeds 5
    .venv/Scripts/python random_agent.py roms/pokemon_red.gb --watch   # open a window
"""
import argparse
import json
import random
import statistics
from pathlib import Path

from env import BUTTONS, PokemonEnv
from metrics import Tracker


def run(args, seed: int, actions: list, out: Path) -> dict:
    rng = random.Random(seed)
    env = PokemonEnv(args.rom, headless=not args.watch, speed=1 if args.watch else 0)
    if args.state:
        env.load(args.state)
    else:
        env.tick(60 * 10)  # skip boot logos
    tracker = Tracker()
    tracker.update(env.state())

    for i in range(args.steps):
        state = env.step(rng.choice(actions))
        tracker.update(state)
        if seed == 0 and i % 200 == 0:
            env.screenshot().save(out / f"frame_{i:05d}.png")
    env.close()
    return tracker.summary()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rom")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--state", help="start from a saved state (e.g. after_intro from discovery)")
    p.add_argument("--controls", help="controls.json from discover.py; uses its action_set")
    p.add_argument("--watch", action="store_true", help="show the emulator window at normal speed")
    p.add_argument("--out", default="runs/random")
    args = p.parse_args()

    actions = BUTTONS
    if args.controls:
        actions = json.loads(Path(args.controls).read_text())["manual"]["action_set"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in range(args.seeds):
        r = run(args, seed, actions, out)
        print(f"seed {seed}: {r}")
        results.append(r)

    mean = {k: statistics.mean(r[k] for r in results) for k in results[0]}
    report = {"agent": "random", "actions": actions, "steps": args.steps, "start_state": args.state,
              "per_seed": results, "mean": mean}
    (out / "baseline.json").write_text(json.dumps(report, indent=2))
    print("mean:", mean)


if __name__ == "__main__":
    main()
