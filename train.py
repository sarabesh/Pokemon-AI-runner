"""PPO training of the local player on parallel headless games.

    python train.py --run first --timesteps 2000000 --reward v1
    python train.py --run v2 --timesteps 2000000 --reward v2
    python train.py --run v3 --timesteps 2000000 --reward v3 --episode-steps 2048 --progress-prob 0
    python train.py --run v4 --timesteps 5000000 --reward v4
    python train.py --run v5 --timesteps 5000000             # reward v5 (default)
    python train.py --run first --resume runs/train/first/checkpoints/ppo_400000_steps.zip

Writes to runs/train/<run>/: progress_states/ (auto-saved story checkpoints), episodes.csv (per-episode metrics + reward components, for the
parent LLM to read), checkpoints/, final.zip, and TensorBoard logs.
"""
import argparse
import csv
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from gym_env import ROOT, PokemonGymEnv, obs_version_of


class EpisodeLog(BaseCallback):
    """Write each finished episode's metrics to CSV and TensorBoard."""

    def __init__(self, path: Path):
        super().__init__()
        self.path, self.writer, self.file = path, None, None

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            m = info.get("episode_metrics")
            if not m:
                continue
            row = {"timesteps": self.num_timesteps, "return": info.get("episode", {}).get("r"), **m}
            if self.writer is None:
                new = not self.path.exists()
                # on --resume keep the existing columns, even if the reward preset changed
                fields = list(row) if new else next(csv.reader(self.path.open()))
                self.file = self.path.open("a", newline="")
                self.writer = csv.DictWriter(self.file, fieldnames=fields, extrasaction="ignore")
                if new:
                    self.writer.writeheader()
            self.writer.writerow(row)
            self.file.flush()
            for k, v in m.items():
                self.logger.record_mean(f"game/{k}", v)
        return True

    def _on_training_end(self):
        if self.file:
            self.file.close()


def make_env(rank, args):
    def _init():
        return PokemonGymEnv(start_states=args.start_states, max_steps=args.episode_steps,
                             reward_weights=json.loads(args.reward_weights),
                             reward_version=args.reward, obs_version=args.obs_version,
                             progress_dir=args.progress_dir, progress_prob=args.progress_prob,
                             seed=rank)
    return _init


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="ppo")
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--envs", type=int, default=12)
    p.add_argument("--episode-steps", type=int, default=8192,
                   help="presses per episode; 8192 fits Pallet -> Viridian -> Pallet -> Viridian")
    p.add_argument("--start-states", nargs="+",
                   default=[str(ROOT / "states" / "after_rival.state")])
    p.add_argument("--reward", default="v5", choices=["v1", "v2", "v3", "v4", "v5"],
                   help="reward preset in reward.py")
    p.add_argument("--progress-prob", type=float, default=0.5,
                   help="chance an episode starts from an auto-saved story checkpoint (0 = off)")
    p.add_argument("--reward-weights", default="{}", help='JSON overrides, e.g. \'{"new_tile": 0.2}\'')
    p.add_argument("--resume", help="checkpoint .zip to continue from")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    out = ROOT / "runs" / "train" / args.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    # older models have older observation layouts (see gym_env.py); keep the env matching
    args.obs_version = 3
    if args.resume:
        args.obs_version = obs_version_of(PPO.load(args.resume, device="cpu").observation_space)
    args.progress_dir = str(out / "progress_states") if args.progress_prob > 0 else None
    env = VecMonitor(SubprocVecEnv([make_env(i, args) for i in range(args.envs)]))
    if args.resume:
        model = PPO.load(args.resume, env=env, device=args.device)
        model.tensorboard_log = str(out / "tb")  # log under this run, not the run it came from
    else:
        model = PPO(
            "MultiInputPolicy", env,
            n_steps=512, batch_size=512, n_epochs=3,
            gamma=0.998, gae_lambda=0.95, ent_coef=0.01, learning_rate=3e-4,
            tensorboard_log=str(out / "tb"), device=args.device, verbose=1,
        )

    callbacks = CallbackList([
        EpisodeLog(out / "episodes.csv"),
        CheckpointCallback(save_freq=max(100_000 // args.envs, 1), save_path=str(out / "checkpoints"),
                           name_prefix="ppo"),
    ])
    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks,
                    reset_num_timesteps=not args.resume, tb_log_name="ppo")
    finally:
        model.save(out / "final.zip")
        env.close()
    print(f"saved {out / 'final.zip'}")


if __name__ == "__main__":
    main()
