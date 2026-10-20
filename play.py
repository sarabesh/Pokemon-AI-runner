"""Step-by-step control from the command line, so Claude Code (or you) can act as the parent LLM.

The game state persists between calls in runs/cc/current.state. Every command prints the RAM
state and writes the current screen to runs/cc/screen.png (2x size).

    python play.py reset                       # boot a fresh game
    python play.py do "a*5 down a wait60"      # buttons, with *N repeats and waitN frames
    python play.py save after_intro            # -> runs/cc/states/after_intro.state
    python play.py load after_intro
    python play.py show                        # just print state + screenshot
"""
import argparse
import json
import re
import shutil
from pathlib import Path

from env import BUTTONS, PokemonEnv

ROOT = Path(__file__).parent
SESSION = ROOT / "runs" / "cc"
CURRENT = SESSION / "current.state"


def run_sequence(env: PokemonEnv, seq: str) -> list:
    log = []
    for tok in seq.split():
        if m := re.fullmatch(r"wait(\d+)", tok):
            env.tick(int(m[1]))
            log.append(tok)
            continue
        m = re.fullmatch(r"([a-z]+)(?:\*(\d+))?", tok)
        if not m or m[1] not in BUTTONS:
            raise SystemExit(f"bad token {tok!r}; use {BUTTONS}, button*N, or waitN")
        for _ in range(int(m[2] or 1)):
            env.step(m[1])
        log.append(tok)
    return log


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["reset", "do", "save", "load", "show"])
    p.add_argument("arg", nargs="?")
    p.add_argument("--rom", default=str(ROOT / "roms" / "pokemon_red.gb"))
    args = p.parse_args()

    SESSION.mkdir(parents=True, exist_ok=True)
    (SESSION / "states").mkdir(exist_ok=True)
    env = PokemonEnv(args.rom)

    if args.cmd == "reset":
        env.tick(60 * 3)
    elif args.cmd == "load":
        shutil.copy(SESSION / "states" / f"{args.arg}.state", CURRENT)
        env.load(str(CURRENT))
    else:
        if not CURRENT.exists():
            raise SystemExit("no session yet; run: python play.py reset")
        env.load(str(CURRENT))

    if args.cmd == "do":
        run_sequence(env, args.arg or "")
    elif args.cmd == "save":
        env.save(str(SESSION / "states" / f"{args.arg}.state"))

    env.save(str(CURRENT))
    img = env.screenshot()
    img.resize((img.width * 2, img.height * 2)).save(SESSION / "screen.png")
    print(json.dumps(env.state()))
    with (SESSION / "history.log").open("a") as f:
        f.write(f"{args.cmd} {args.arg or ''}\n")
    env.close()


if __name__ == "__main__":
    main()
