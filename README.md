# pokemon-ai

An experiment in having AI play **Pokemon Red**: a *parent LLM* works out how the game works and
designs the rewards, and a *small local model* (a ~0.6M-parameter CNN trained with PPO) learns to
actually play by trial and error on a laptop GPU.

```
Parent LLM (Claude Code, occasional)          Local player (PPO, runs for hours)
 ├─ discovers controls & RAM signals    ──►    12 headless Game Boys in parallel
 ├─ writes / revises reward.py          ──►    ~500 button presses per second
 └─ reads episodes.csv, diagnoses       ◄──    per-episode metrics + reward breakdown
```

No paid API is required: discovery was done with a scripted probe plus Claude Code (subscription)
driving the emulator, and training is free on a local GPU.

## Why Pokemon Red + PyBoy

- **Pokemon Red (Game Boy, Gen 1)** – the simplest mainline game: 160x144 screen, 8 buttons,
  turn-based battles, and the whole game is disassembled ([pret/pokered](https://github.com/pret/pokered)),
  so every RAM address (position, map, party, HP, XP, PP, story flags) is known. Prior work to compare
  against: Claude Plays Pokemon, PWhiddy's PokemonRedExperiments.
- **[PyBoy](https://github.com/Baekalfen/PyBoy)** – a Game Boy emulator as a pip package. Runs headless
  far faster than real time, takes button presses from Python, exposes screen and RAM, save/load states.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
# GPU build of PyTorch (optional, CUDA 12.6):
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

**ROM:** put your own legally dumped Pokemon Red (US) ROM at `roms/pokemon_red.gb`. ROMs are not in
this repo and are git-ignored. The RAM addresses assume the US release.

`states/` holds emulator save states used as start points (`after_intro`, `got_starter`,
`battle_menu`, `after_rival`, `wild_battle`); training starts from `after_rival` (Charmander in
hand, standing in Oak's lab after the first rival battle).

## Quick start

```bash
# train (reward v5, ~2.5 h for 5M steps on an RTX 3050 laptop)
.venv/Scripts/python train.py --run v5 --timesteps 5000000

# live charts, all runs side by side
.venv/Scripts/tensorboard --logdir runs/train

# score a model (3 x 2000 presses from after_rival) and watch it play slowly
.venv/Scripts/python evaluate.py runs/train/v5/final.zip
.venv/Scripts/python watch.py runs/train/v5/final.zip --speed 1 --delay 1
```

## How it works

### 1. Discovery (`probe.py`, `play.py`, `controls.json`)

- `probe.py` (no LLM) mashes through the intro in ~124 presses, then tests every button and diffs RAM.
- Claude Code drove `play.py` (one command per call, persistent state, screenshot each time) from the
  bedroom through Oak's lab, the starter choice and the first rival battle, saving checkpoints.
- Findings in `controls.json`: drop START/SELECT (menu traps, no effect), detect context from RAM
  (battle flag 0xD057; any text box/menu draws border tile 0x79 into the tile map 0xC3A0), mashing A
  wins early battles, and a list of reward ideas and pitfalls.
- `discover.py` is an optional paid alternative that does the same via the Claude API.

Random baseline (`random_agent.py`, 2000 presses, mean of 3 seeds) – dropping START/SELECT alone
more than doubles exploration:

| Start | Actions | Unique tiles | Maps | Party level sum |
|---|---|---|---|---|
| after_intro | all 8 | 46 | 1.3 | 0 |
| after_intro | reduced 6 | 124 | 3.3 | 3.3 |
| after_rival | all 8 | 65 | 1.7 | 6 |
| after_rival | reduced 6 | 150 | 3.0 | 6 |

### 2. Environment (`env.py`, `gym_env.py`)

One step = one button press (8 frames held + 24 frames wait ≈ 0.5 s of game time). `env.py` reads
state from RAM: map, position, party levels/HP/XP/PP (max PP from the ROM move table), badges, money,
battle type, enemy HP, story event flags (`wEventFlags` 0xD747, 2560 bits) and context.

Observations (the model only loads with the version it was trained on; `evaluate.py`, `watch.py`
and `train.py --resume` detect it):

| Obs version | Inputs | Runs |
|---|---|---|
| 1 | last 3 frames (grayscale 72x80) + 12 RAM features | `first`, `v2` |
| 2 | + party PP fraction (13) | `v3` |
| 3 | + enemy HP fraction, enemy level, wild/trainer, story events (episode / overall), badges (20) + **21x21 explore map** of visited tiles around the player | `v4`, `v5` |

### 3. Training (`train.py`)

stable-baselines3 PPO, `MultiInputPolicy` (Nature CNN on frames + MLP), 12 parallel envs,
n_steps 512, batch 512, 3 epochs, gamma 0.998, ent_coef 0.01. Per-episode metrics and every reward
component go to `runs/train/<run>/episodes.csv` and TensorBoard; checkpoints every ~100k steps.

**Progress checkpoints:** every story event during training saves the emulator state to
`runs/train/<run>/progress_states/`, and with `--progress-prob 0.5` (default) half the episodes start
from one of the 3 furthest stages reached, so later parts of the game get practised.

### 4. Rewards (`reward.py`) and what each version taught us

| Component | v1 | v2 | v3 | v4 | v5 (default) |
|---|---|---|---|---|---|
| new tile (indoor / outdoor) | 0.1 | 0.1 / 0.2 | 0.1 / 0.2 | 0.1 / 0.2 | **0.2 / 0.6** |
| new map (indoor / outdoor) | 2.0 | 1.0 / 5.0 | 1.0 / 5.0 | 1.0 / 5.0 | **3.0 / 30.0** |
| level up (per level) | 1.0 | 5.0 | – | – | – |
| damage (per enemy HP) | 0.05 | 0.3 | – | – | – |
| enemy fainted | – | 5.0 | – | – | – |
| Pokemon XP | – | – | 2.0 per sqrt(XP) | 0.15 per XP below level 12 (+6/badge) | **0.05** per XP below **level 10** (+6/badge) |
| fled a wild battle | – | – | – | -3.0 | **-1.0** |
| story event flag | – | 10.0 | 30.0 | 40.0 + visited tiles/maps reset | **100.0** + reset |
| badge | 20.0 | 20.0 | 50.0 | 50.0 | **300.0** |
| blackout | -5.0 | -5.0 | -10.0 | -10.0 | -10.0 |
| no effect (overworld press that changes nothing: wall bump, A/B at air) | – | – | – | – | **-1.0** |
| stuck (per press after N presses without progress) | – | – | -0.05, N=300 | -0.05, N=500 | -0.05, N=300 |
| low PP (per press while party PP < 25%) | – | – | -0.05 | -0.05 | -0.05 |

Outdoor = map ids 0x00–0x24 (towns and Routes 1–25).

| Run | Reward | Steps / episode | Result (last training episodes) |
|---|---|---|---|
| `first` | v1 | 2M / 2048 | explores ~638 tiles, ~11 maps (4x random); never fights, level stays 6 |
| `v2` | v2 | 2M / 2048 | grinds Route 1: ~35 battles, level 11, but only ~80 tiles; ~85% of reward from fights |
| `v3` | v3 | 2M / 2048 | flees ~every battle, stalls at Viridian (road opens only after the parcel is delivered), policy near random |
| `v4` | v4 | 5M / 8192 | best explorer (~790 tiles), wins 26 / flees 3, but farms Route 1 to the level cap; gets the parcel, doesn't deliver it |
| `v5` | v5 | – | not trained yet: battles pay less, exploration and story goals far more, wall bumps punished |

Lessons so far: the agent optimises exactly what's paid (explore-only → never fights; per-fight
reward → endless grinding; tiny fight reward → always flees). Story progress that needs
backtracking only happens when backtracking itself pays and the agent can see its progress.

## Files

| File | What it does |
|---|---|
| `env.py` | `PokemonEnv`: PyBoy wrapper – `step(button)`, `screenshot()`, `state()`, `context()`, save/load |
| `gym_env.py` | Gymnasium env: observations, reward hookup, progress checkpoints, obs versions |
| `reward.py` | reward presets v1–v5; overrides via `train.py --reward-weights '{"no_effect": -0.3}'` |
| `metrics.py` | tiles, maps, levels, badges, battles started / won / fled, party XP |
| `train.py` | PPO training on parallel headless games |
| `evaluate.py` | fixed-protocol scoring of a model (comparable across runs) |
| `watch.py` | watch a model play in a window (`--delay` to slow it down; writes no saves) |
| `probe.py` | scripted discovery: intro skip, per-button effects, menu RAM signals |
| `play.py` | step-by-step CLI control (how Claude Code played the game) |
| `controls.json` | discovered controls manual |
| `random_agent.py` | random baseline |
| `discover.py` | optional: discovery via the Claude API (`ANTHROPIC_API_KEY` in `.env`) |
| `states/` | start-point save states |

Generated output goes to `runs/` (git-ignored): training runs, probe results, play sessions.
