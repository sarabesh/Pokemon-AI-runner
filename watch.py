"""Watch a trained policy play in an emulator window.

    python watch.py runs/train/first/final.zip
    python watch.py runs/train/first/final.zip --speed 1 --delay 1   # slow: 1 s pause per press
    python watch.py runs/train/first/final.zip --speed 3 --steps 4000
    python watch.py runs/train/first/checkpoints/ppo_1000008_steps.zip --deterministic

Prints the action, game context, position and reward as it goes. Close the window or Ctrl+C to stop.
"""
import argparse
import time

from stable_baselines3 import PPO

from gym_env import ROOT, PokemonGymEnv, obs_version_of


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--state", default=str(ROOT / "states" / "after_rival.state"))
    p.add_argument("--steps", type=int, default=2048)
    p.add_argument("--speed", type=int, default=2, help="1 = real time, 0 = as fast as possible")
    p.add_argument("--delay", type=float, default=0.0,
                   help="extra pause in seconds after every button press (e.g. 1.0 to follow along); "
                        "also prints every press")
    p.add_argument("--deterministic", action="store_true", help="always take the top action")
    p.add_argument("--reward", default="v5", choices=["v1", "v2", "v3", "v4", "v5"], help="which reward to display")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    model = PPO.load(args.model, device="cpu")
    obs_version = obs_version_of(model.observation_space)  # older models, older inputs
    model.set_random_seed(args.seed)
    env = PokemonGymEnv(start_states=[args.state], max_steps=args.steps, seed=args.seed,
                        headless=False, speed=args.speed, reward_version=args.reward,
                        obs_version=obs_version)
    obs, _ = env.reset(seed=args.seed)
    total = 0.0
    try:
        for i in range(args.steps):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, r, _, truncated, info = env.step(action)
            total += r
            s = env.state
            if args.delay:
                time.sleep(args.delay)  # the window stays on the last frame meanwhile
            if args.delay or r or i % 50 == 0:
                print(f"{i:5d} {env.actions[int(action)]:5s} {s['context']:9s} map {s['map_id']:3d} "
                      f"pos {s['pos']}  r {r:+.2f}  total {total:.1f}")
            if truncated:
                print("episode done:", info["episode_metrics"])
                break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
