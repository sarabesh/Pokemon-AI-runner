"""Score a trained policy with the same protocol as random_agent.py, and compare.

    python evaluate.py runs/train/first/final.zip
    python evaluate.py runs/train/first/final.zip --state states/after_rival.state --steps 2000
"""
import argparse
import json
import statistics
from pathlib import Path

from stable_baselines3 import PPO

from gym_env import ROOT, PokemonGymEnv, obs_version_of


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--state", default=str(ROOT / "states" / "after_rival.state"))
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--baseline", help="random_agent baseline.json to compare against")
    args = p.parse_args()

    model = PPO.load(args.model, device="cpu")
    obs_version = obs_version_of(model.observation_space)  # older models, older inputs
    out = Path(args.model).with_suffix("")
    out.mkdir(exist_ok=True)
    results = []
    for seed in range(args.seeds):
        env = PokemonGymEnv(start_states=[args.state], max_steps=args.steps, seed=seed,
                            obs_version=obs_version)
        obs, _ = env.reset(seed=seed)
        model.set_random_seed(seed)
        for i in range(args.steps):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, _, _, truncated, info = env.step(action)
            if seed == 0 and i % 200 == 0:
                env.game.screenshot().save(out / f"frame_{i:05d}.png")
            if truncated:
                break
        r = info["episode_metrics"]
        print(f"seed {seed}: {r}")
        results.append(r)
        env.close()

    keys = ["unique_tiles", "maps_visited", "party_level_sum", "badges", "battles_started",
            "battles_won", "battles_fled", "party_xp"]
    mean = {k: statistics.mean(r[k] for r in results) for k in keys}
    report = {"agent": "ppo", "model": args.model, "steps": args.steps, "start_state": args.state,
              "per_seed": results, "mean": mean}
    (out / "eval.json").write_text(json.dumps(report, indent=2))
    print("ppo mean:   ", mean)
    if args.baseline:
        base = json.loads(Path(args.baseline).read_text())["mean"]
        print("random mean:", {k: base.get(k) for k in keys})


if __name__ == "__main__":
    main()
