"""Gymnasium wrapper for PPO training.

Observation (dict):
  screen:  last 3 frames, grayscale, downsampled 2x -> uint8 (3, 72, 80)
  vec:     RAM feature vector (context, position, map, party HP/levels, novelty, ...)
  explore: (obs version 3) 21x21 tiles around the player on the current map, 1 = visited
           this episode, so the policy can see where it hasn't been yet
Action: index into the action set from controls.json (a, b, up, down, left, right).

Obs versions (models only load with the version they were trained on):
  1  vec 12              runs "first", "v2"
  2  vec 13 (+ PP)       run "v3"
  3  vec 20 + explore    v4 onward: + enemy HP/level, wild/trainer, story progress, badges

Progress checkpoints: with progress_dir set, every story event saves the emulator state
there, and resets start from those saved states with probability progress_prob, so later
parts of the game get practised without replaying the start every episode.
"""
import json
import os
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env import ENEMY_HP, PokemonEnv
from metrics import Tracker
from reward import Reward

ROOT = Path(__file__).parent
DEFAULT_START = str(ROOT / "states" / "after_rival.state")
ENEMY_MAX_HP, ENEMY_LEVEL, BATTLE_TYPE = 0xCFF4, 0xCFF3, 0xD057
CONTEXTS = ["overworld", "window", "battle"]
FRAMES = 3
EXPLORE_R = 10  # explore window radius -> 21x21
VEC_SIZE = {1: 12, 2: 13, 3: 20}


def load_actions(path=ROOT / "controls.json"):
    return json.loads(Path(path).read_text())["manual"]["action_set"]


def obs_version_of(space) -> int:
    """Which observation layout a trained model expects (pass model.observation_space)."""
    if "explore" in space.spaces:
        return 3
    return 2 if space["vec"].shape[0] == VEC_SIZE[2] else 1


class PokemonGymEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, rom=str(ROOT / "roms" / "pokemon_red.gb"), start_states=None,
                 max_steps=2048, reward_weights=None, actions=None, seed=None,
                 headless=True, speed=0, reward_version="v5", obs_version=3,
                 progress_dir=None, progress_prob=0.0, progress_keep=4):
        super().__init__()
        self.rom = rom
        self.headless, self.speed = headless, speed
        self.obs_version = obs_version
        self.start_states = start_states or [DEFAULT_START]
        self.max_steps = max_steps
        self.actions = actions or load_actions()
        self.reward = Reward(reward_weights, reward_version)
        self.rank = seed or 0
        self.rng = random.Random(seed)
        self.progress_dir = Path(progress_dir) if progress_dir else None
        self.progress_prob, self.progress_keep = progress_prob, progress_keep
        if self.progress_dir:
            self.progress_dir.mkdir(parents=True, exist_ok=True)
        self.game = None

        self.action_space = spaces.Discrete(len(self.actions))
        obs = {
            "screen": spaces.Box(0, 255, (FRAMES, 72, 80), np.uint8),
            "vec": spaces.Box(-1.0, 1.0, (VEC_SIZE[obs_version],), np.float32),
        }
        if obs_version >= 3:
            n = 2 * EXPLORE_R + 1
            obs["explore"] = spaces.Box(0.0, 1.0, (n, n), np.float32)
        self.observation_space = spaces.Dict(obs)

    # --- helpers
    def _word(self, addr):
        m = self.game.mem
        return (m[addr] << 8) | m[addr + 1]

    def _frame(self):
        return self.game.pyboy.screen.ndarray[::2, ::2, 0].copy()  # palette is grayscale

    def _vec(self, state):
        party = state["party"]
        hp = sum(p["hp"] for p in party)
        x, y = state["pos"]
        v = [float(state["context"] == c) for c in CONTEXTS] + [
            x / 64, y / 64, state["map_id"] / 255,
            len(party) / 6, sum(p["level"] for p in party) / 100,
            min(hp / 100, 1.0), float(hp == 0 and bool(party)),
            min(len(self.reward.tiles) / 1000, 1.0), self.steps / self.max_steps,
        ]
        if self.obs_version >= 2:
            pp, pp_max = state["pp"]
            v.append(pp / pp_max if pp_max else 1.0)
        if self.obs_version >= 3:
            battle = self.game.mem[BATTLE_TYPE]
            enemy_max = self._word(ENEMY_MAX_HP)
            v += [
                state["enemy_hp"] / enemy_max if battle and enemy_max else 0.0,
                self.game.mem[ENEMY_LEVEL] / 100 if battle else 0.0,
                float(battle == 1), float(battle == 2),       # wild / trainer
                min((state["events"] - self.start_events) / 10, 1.0),  # progress this episode
                min(state["events"] / 50, 1.0),                # overall story progress
                state["badges"] / 8,
            ]
        return np.clip(np.array(v, np.float32), -1, 1)

    def _explore(self, state):
        m, (x, y) = state["map_id"], state["pos"]
        tiles = self.reward.tiles
        r = range(-EXPLORE_R, EXPLORE_R + 1)
        return np.array([[float((m, x + dx, y + dy) in tiles) for dx in r] for dy in r],
                        np.float32)

    def _obs(self, state):
        obs = {"screen": np.stack(self.frames), "vec": self._vec(state)}
        if self.obs_version >= 3:
            obs["explore"] = self._explore(state)
        return obs

    # --- progress checkpoints
    def _progress_files(self):
        return list(self.progress_dir.glob("ev*.state")) if self.progress_dir else []

    def _save_progress(self):
        ev = self.state["events"]
        name = f"ev{ev:03d}_m{self.state['map_id']:03d}_r{self.rank}_{time.time_ns()}.state"
        tmp = self.progress_dir / (name + ".tmp")
        self.game.save(str(tmp))
        os.replace(tmp, self.progress_dir / name)  # atomic: other envs never see half a file
        same = sorted(self.progress_dir.glob(f"ev{ev:03d}_*.state"), key=lambda p: p.stat().st_mtime)
        for old in same[:-self.progress_keep]:
            try:
                old.unlink()
            except OSError:
                pass  # another env removed or is loading it

    def _pick_start(self):
        files = self._progress_files()
        if files and self.rng.random() < self.progress_prob:
            levels = sorted({f.name[2:5] for f in files}, reverse=True)[:3]  # 3 furthest stages
            pool = [f for f in files if f.name[2:5] in levels]
            return str(self.rng.choice(pool)), True
        return self.rng.choice(self.start_states), False

    # --- gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
        if self.game is None:
            self.game = PokemonEnv(self.rom, headless=self.headless, speed=self.speed)
        start, from_progress = self._pick_start()
        try:
            self.game.load(start)
        except (OSError, ValueError):  # checkpoint pruned by another env meanwhile
            start, from_progress = self.rng.choice(self.start_states), False
            self.game.load(start)
        self.from_progress = from_progress
        self.steps = 0
        self.reward.reset()
        self.tracker = Tracker()
        self.state = self.game.state()
        self.start_events = self.state["events"]
        self.enemy_hp = self.state["enemy_hp"]
        self.reward(self.state, self.state, self.enemy_hp, self.enemy_hp)  # mark start tile visited
        self.reward.totals = {k: 0.0 for k in self.reward.totals}  # ...without counting it
        self.tracker.update(self.state)
        f = self._frame()
        self.frames = [f] * FRAMES
        return self._obs(self.state), {}

    def step(self, action):
        prev, prev_enemy = self.state, self.enemy_hp
        self.state = self.game.step(self.actions[int(action)])
        self.enemy_hp = self.state["enemy_hp"]
        self.steps += 1
        r = self.reward(prev, self.state, prev_enemy, self.enemy_hp)
        self.tracker.update(self.state)
        self.frames = self.frames[1:] + [self._frame()]
        if self.progress_dir and "event" in self.reward.last_parts:
            self._save_progress()

        truncated = self.steps >= self.max_steps
        info = {}
        if truncated:
            info["episode_metrics"] = {
                **self.tracker.summary(),
                "start_events": self.start_events,
                "events_gained": self.state["events"] - self.start_events,
                "from_progress": int(self.from_progress),
                **{f"r_{k}": v for k, v in self.reward.totals.items()},
            }
        return self._obs(self.state), r, False, truncated, info

    def render(self):
        return self.game.pyboy.screen.ndarray[:, :, :3].copy()

    def close(self):
        if self.game:
            self.game.close()
            self.game = None
