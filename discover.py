"""Phase 0: a parent LLM (Claude) explores the game and writes a controls manual.

Claude drives the emulator through tools. Every press returns a screenshot plus the
RAM-derived state and what changed, so it can work out what each button does in each
context (title screen, dialogue, menus, overworld, battle). It saves checkpoints along
the way (e.g. right after the intro, the starting point for RL) and finishes by writing
controls.json: the reduced action set and per-context button semantics.

    .venv/Scripts/python discover.py roms/pokemon_red.gb
    .venv/Scripts/python discover.py roms/pokemon_red.gb --max-turns 60 --effort low
"""
import argparse
import base64
import io
import json
import re
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from env import BUTTONS, PokemonEnv
from metrics import Tracker

load_dotenv(Path(__file__).parent / ".env")  # ANTHROPIC_API_KEY

SYSTEM = """You are the "scientist" in an experiment where a small RL model will later learn to play \
Pokemon Red. Your job is NOT to beat the game. Your job is to discover how the controls work and \
write a manual the RL setup can use.

You control a Game Boy emulator through tools. Each press returns a 2x screenshot, the game state \
read from RAM (map id, x/y position, party, badges, in_battle, money) and what changed.

Do this, roughly in order:
1. Get through the title screen and intro (Oak's speech, naming yourself and the rival). Pick \
default names to save time. Once you are standing in your bedroom and can walk, call \
save_checkpoint with name "after_intro" - this is where RL training will start.
2. In the overworld, test each button: which ones move you, how many tiles per press, what \
START/SELECT/B do, what A does facing an object or person. Watch the pos/map_id changes.
3. Leave the house to reach Pallet Town, then try to trigger Oak's grass event and get a starter \
Pokemon and the first rival battle. Save a checkpoint before and during your first battle \
("before_first_battle", "in_battle").
4. In battle, learn the menu layout (FIGHT/PKMN/ITEM/RUN), how to select a move, and what is \
needed to advance text.

Be efficient: use the `times` argument to repeat presses (e.g. advancing long dialogue), and \
don't re-test things you already understand. When you have enough (or are told you are close \
to the turn limit), call finish with the manual. Key questions the manual must answer: which \
buttons should the RL agent be allowed to use (dropping ones that mostly waste time, like menus \
that trap a random policy), and what RAM signals reliably tell which context the game is in."""

TOOLS = [
    {
        "name": "press",
        "description": "Press a Game Boy button, optionally several times in a row, then return "
        "the screenshot, RAM state and what changed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "button": {"type": "string", "enum": BUTTONS},
                "times": {"type": "integer", "minimum": 1, "maximum": 30,
                          "description": "How many times to press it (default 1)."},
            },
            "required": ["button"],
            "additionalProperties": False,
        },
    },
    {
        "name": "wait",
        "description": "Let the game run without input for some frames (60 frames = 1 second), "
        "then return the screenshot and state.",
        "input_schema": {
            "type": "object",
            "properties": {"frames": {"type": "integer", "minimum": 1, "maximum": 600}},
            "required": ["frames"],
            "additionalProperties": False,
        },
    },
    {
        "name": "save_checkpoint",
        "description": "Save the emulator state under a short name so training can start from here.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "pattern": "^[a-z0-9_]{1,40}$"},
                "description": {"type": "string"},
            },
            "required": ["name", "description"],
            "additionalProperties": False,
        },
    },
    {
        "name": "finish",
        "description": "End exploration and write the controls manual.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action_set": {
                    "type": "array", "items": {"type": "string", "enum": BUTTONS},
                    "description": "Buttons the RL agent should be allowed to press.",
                },
                "dropped_buttons": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"button": {"type": "string"}, "reason": {"type": "string"}},
                        "required": ["button", "reason"],
                        "additionalProperties": False,
                    },
                },
                "contexts": {
                    "type": "array",
                    "description": "One entry per game context (overworld, dialogue, menu, battle, ...).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "how_to_detect": {"type": "string",
                                              "description": "RAM signals and/or visual cues."},
                            "buttons": {"type": "object", "additionalProperties": {"type": "string"},
                                        "description": "button -> what it does here"},
                        },
                        "required": ["name", "how_to_detect", "buttons"],
                        "additionalProperties": False,
                    },
                },
                "movement": {"type": "string",
                             "description": "How movement works: tiles per press, turning, blocking."},
                "reward_ideas": {"type": "array", "items": {"type": "string"},
                                 "description": "RAM-based reward signals worth using for RL."},
                "pitfalls": {"type": "array", "items": {"type": "string"},
                             "description": "Traps a random or naive policy will fall into."},
            },
            "required": ["action_set", "dropped_buttons", "contexts", "movement",
                         "reward_ideas", "pitfalls"],
            "additionalProperties": False,
        },
    },
]


def image_block(img) -> dict:
    img = img.resize((img.width * 2, img.height * 2))  # 320x288, easier to read
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = base64.standard_b64encode(buf.getvalue()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


def diff(before: dict, after: dict) -> dict:
    return {k: [before[k], after[k]] for k in after if before[k] != after[k]}


class Session:
    def __init__(self, env: PokemonEnv, out: Path):
        self.env, self.out = env, out
        self.tracker = Tracker()
        self.checkpoints = {}
        self.manual = None
        self.presses = 0

    def observe(self, before: dict, note: str) -> list:
        after = self.env.state()
        self.tracker.update(after)
        text = json.dumps({"note": note, "state": after, "changed": diff(before, after)})
        return [{"type": "text", "text": text}, image_block(self.env.screenshot())]

    def run_tool(self, name: str, args: dict):
        before = self.env.state()
        if name == "press":
            times = max(1, min(int(args.get("times", 1)), 30))
            for _ in range(times):
                self.env.step(args["button"])
            self.presses += times
            return self.observe(before, f"pressed {args['button']} x{times}")
        if name == "wait":
            self.env.tick(max(1, min(int(args["frames"]), 600)))
            return self.observe(before, f"waited {args['frames']} frames")
        if name == "save_checkpoint":
            if not re.fullmatch(r"[a-z0-9_]{1,40}", args["name"]):
                raise ValueError("name must be lowercase letters, digits, underscores")
            path = self.out / "states" / f"{args['name']}.state"
            path.parent.mkdir(parents=True, exist_ok=True)
            self.env.save(str(path))
            self.checkpoints[args["name"]] = {"path": str(path), "description": args["description"],
                                              "state": before}
            return f"saved {path}"
        if name == "finish":
            self.manual = args
            return "manual recorded"
        raise ValueError(f"unknown tool {name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rom")
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--max-turns", type=int, default=120, help="LLM calls before forcing finish")
    p.add_argument("--watch", action="store_true", help="show the emulator window")
    p.add_argument("--out", default="runs/discovery")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = PokemonEnv(args.rom, headless=not args.watch, speed=1 if args.watch else 0)
    env.tick(60 * 3)
    session = Session(env, out)
    client = anthropic.Anthropic()

    messages = [{"role": "user", "content": [
        {"type": "text", "text": "The game just booted. Here is the current screen and state. Begin."},
        *session.observe(env.state(), "boot"),
    ]}]
    transcript = out / "transcript.jsonl"
    usage = {"input": 0, "output": 0, "cache_read": 0}

    for turn in range(args.max_turns):
        if turn == args.max_turns - 5:
            messages.append({"role": "system",
                             "content": "5 turns left. Wrap up and call finish with the manual now."})
        try:
            response = client.beta.messages.create(
                model=args.model,
                max_tokens=16000,
                system=SYSTEM,
                tools=TOOLS,
                messages=messages,
                thinking={"type": "adaptive"},
                output_config={"effort": args.effort},
                cache_control={"type": "ephemeral"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.RateLimitError as e:
            wait_s = int(e.response.headers.get("retry-after", "30"))
            print(f"rate limited, sleeping {wait_s}s")
            time.sleep(wait_s)
            continue

        u = response.usage
        usage["input"] += u.input_tokens
        usage["output"] += u.output_tokens
        usage["cache_read"] += u.cache_read_input_tokens or 0

        if response.stop_reason == "refusal":
            print("model refused:", response.stop_details)
            break

        messages.append({"role": "assistant", "content": response.content})
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"[{turn}] {block.text.strip()}")

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            if session.manual:
                break
            messages.append({"role": "user", "content": "Keep going: use the tools, and call "
                             "finish when you have the manual."})
            continue

        results = []
        for tu in tool_uses:
            try:
                content = session.run_tool(tu.name, tu.input)
                results.append({"type": "tool_result", "tool_use_id": tu.id, "content": content})
            except Exception as e:  # bad args etc. - let the model correct itself
                results.append({"type": "tool_result", "tool_use_id": tu.id,
                                "content": f"Error: {e}", "is_error": True})
            with transcript.open("a") as f:
                f.write(json.dumps({"turn": turn, "tool": tu.name, "input": tu.input,
                                    "state": env.state()}) + "\n")
        messages.append({"role": "user", "content": results})
        print(f"[{turn}] {', '.join(f'{t.name}{t.input}' for t in tool_uses)} -> {env.state()['pos']}")

        if session.manual:
            break

    env.save(str(out / "final.state"))
    env.screenshot().save(out / "final.png")
    env.close()

    report = {
        "model": args.model,
        "turns": turn + 1,
        "button_presses": session.presses,
        "usage_tokens": usage,
        "metrics": session.tracker.summary(),
        "checkpoints": session.checkpoints,
        "manual": session.manual,
    }
    (out / "controls.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("turns", "button_presses", "usage_tokens", "metrics")},
                     indent=2))
    print(f"wrote {out / 'controls.json'}" + ("" if session.manual else " (no manual - hit turn limit)"))


if __name__ == "__main__":
    main()
