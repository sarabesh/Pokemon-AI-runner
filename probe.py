"""Scripted (free) part of discovery: get through the intro and measure what each button does.

No LLM here. It produces runs/probe/report.json + screenshots, which the parent LLM (Claude
Code, via play.py) reads to write the controls manual.

    python probe.py
"""
import json
from pathlib import Path

from env import BUTTONS, PokemonEnv

ROOT = Path(__file__).parent
OUT = ROOT / "runs" / "probe"
REDS_ROOM = 38  # RedsHouse2F
WRAM = range(0xC000, 0xE000)


def wram(env):
    return bytes(env.mem[a] for a in WRAM)


def can_move(env, tmp: Path) -> bool:
    """Does pressing a direction change position? Restores the state afterwards."""
    env.save(str(tmp))
    moved = False
    for d in ["down", "up", "left", "right"]:
        before = env.state()["pos"]
        env.step(d)
        env.step(d)
        if env.state()["pos"] != before:
            moved = True
            break
    env.load(str(tmp))
    return moved


def skip_intro(env, tmp: Path, max_presses=4000) -> int:
    """Mash A (START now and then, to confirm name entry) until we can walk in Red's room.

    The map id is already Red's room during Oak's speech, so every 25 presses we try closing
    any menu/dialogue and walking, and roll back if that fails (B would undo name entry)."""
    check = tmp.with_name("check.state")
    for i in range(max_presses):
        if i % 25 == 24 and env.state()["map_id"] == REDS_ROOM:
            env.save(str(check))
            for _ in range(4):
                env.step("b")
            if can_move(env, tmp):
                check.unlink()
                return i
            env.load(str(check))
        env.step("start" if i % 6 == 5 else "a")
    raise RuntimeError("did not reach Red's room")


def diff(a: dict, b: dict) -> dict:
    return {k: [a[k], b[k]] for k in b if a[k] != b[k]}


def probe_buttons(env, start: Path, tmp: Path) -> dict:
    results = {}
    for b in BUTTONS:
        env.load(str(start))
        s0, m0 = env.state(), wram(env)
        env.step(b)
        s1, m1 = env.state(), wram(env)
        env.screenshot().save(OUT / f"after_{b}.png")
        changed = [hex(0xC000 + i) for i in range(len(m0)) if m0[i] != m1[i]]
        results[b] = {
            "state_change": diff(s0, s1),
            "wram_bytes_changed": len(changed),
            "can_still_walk_after": can_move(env, tmp),
        }
        # how many presses does it take to get back to walking?
        for n in range(1, 6):
            if results[b]["can_still_walk_after"]:
                break
            env.step("b")
            if can_move(env, tmp):
                results[b]["b_presses_to_recover"] = n
                break
    return results


def probe_movement(env, start: Path) -> dict:
    out = {}
    for d in ["up", "down", "left", "right"]:
        env.load(str(start))
        p0 = env.state()["pos"]
        env.step(d)
        p1 = env.state()["pos"]
        env.step(d)
        p2 = env.state()["pos"]
        out[d] = {"after_1_press": [p1[0] - p0[0], p1[1] - p0[1]],
                  "after_2_presses": [p2[0] - p0[0], p2[1] - p0[1]]}
    return out


def menu_signals(env, start: Path) -> list:
    """WRAM bytes stable while walking but different once the START menu is open."""
    env.load(str(start))
    walking = []
    for d in ["left", "right", "left", "right"]:
        env.step(d)
        walking.append(wram(env))
    env.step("start")
    menu = wram(env)
    stable = [i for i in range(len(menu)) if len({w[i] for w in walking}) == 1]
    return [{"addr": hex(0xC000 + i), "walking": walking[0][i], "menu": menu[i]}
            for i in stable if menu[i] != walking[0][i]][:40]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start, tmp = OUT / "after_intro.state", OUT / "tmp.state"
    env = PokemonEnv(str(ROOT / "roms" / "pokemon_red.gb"))
    env.tick(60 * 3)

    presses = skip_intro(env, tmp)
    env.save(str(start))
    env.screenshot().save(OUT / "after_intro.png")
    print(f"intro done after {presses} presses: {env.state()}")

    report = {
        "intro_presses": presses,
        "after_intro_state": env.state(),
        "movement": probe_movement(env, start),
        "buttons": probe_buttons(env, start, tmp),
        "menu_signal_candidates": menu_signals(env, start),
    }
    tmp.unlink(missing_ok=True)
    env.close()
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("movement", "buttons")}, indent=1))


if __name__ == "__main__":
    main()
