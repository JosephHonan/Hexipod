"""
Train the hexapod walking policy with PPO on the GPU (MJX + Brax).

    python sim/rl/train.py --quick          # ~5M steps smoke test: does the pipeline run?
    python sim/rl/train.py                  # full run (50M steps)
    python sim/rl/train.py --envs 2048      # use fewer parallel robots if the GPU runs out of memory

Watch it learn live in a second terminal:  python sim/rl/watch.py

Checkpoints land in sim/rl/runs/<name>/ (params saved after every evaluation).
The first evaluation is slow: JAX compiles the whole pipeline once (a few minutes).
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")  # leave VRAM for Windows + display
import argparse
import functools
import json
import shutil
import time
from pathlib import Path

import jax
import jax_compat  # noqa: F401  (must come before brax training)
from brax.io import model as brax_model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

import hexapod_env as H

ap = argparse.ArgumentParser()
ap.add_argument("--timesteps", type=int, default=50_000_000)
ap.add_argument("--envs", type=int, default=4096, help="parallel simulated robots")
ap.add_argument("--quick", action="store_true", help="5M-step smoke test with 1024 robots")
ap.add_argument("--name", default=time.strftime("run_%Y%m%d_%H%M%S"))
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--evals", type=int, default=20, help="checkpoints saved during training")
args = ap.parse_args()
if args.quick:
    args.timesteps, args.envs = 5_000_000, 1024

out = Path(__file__).resolve().parent / "runs" / args.name
out.mkdir(parents=True, exist_ok=True)
print(f"JAX devices: {jax.devices()}")
print(f"Training {args.timesteps:,} steps with {args.envs} parallel robots -> {out}")

env, eval_env = H.HexapodEnv(), H.HexapodEnv()
config = dict(
    num_timesteps=args.timesteps, num_evals=args.evals, episode_length=1000, action_repeat=1,
    normalize_observations=True, reward_scaling=1.0, discounting=0.97,
    learning_rate=3e-4, entropy_cost=1e-2, unroll_length=20, batch_size=256,
    num_minibatches=32, num_updates_per_batch=4, num_envs=args.envs, seed=args.seed,
)
(out / "config.json").write_text(json.dumps(
    {"ppo": config, "network": H.NETWORK, "env": {
        "control_dt": H.CONTROL_DT, "gait_period": H.GAIT_PERIOD, "gait_lift": H.GAIT_LIFT,
        "residual_scale": H.RESIDUAL_SCALE, "cmd_range": H.CMD_RANGE,
        "reward_weights": H.REWARD_WEIGHTS}}, indent=2))

t0 = time.time()


def progress(step, metrics):
    r = metrics.get("eval/episode_reward", float("nan"))
    ep_len = metrics.get("eval/avg_episode_length", float("nan"))
    spd = metrics.get("eval/episode_fwd_speed", float("nan"))
    print(f"[{(time.time() - t0) / 60:6.1f} min] step {step:>11,}  "
          f"eval reward {r:8.1f}  episode length {ep_len:6.0f}/1000  "
          f"avg fwd speed {spd / max(ep_len, 1):.3f} m/s", flush=True)


ckpt_dir = out / "checkpoints"
ckpt_dir.mkdir(exist_ok=True)


def save(step, make_policy, params):
    """Keep every checkpoint (for replays) plus 'params' = latest (for watch.py/play.py)."""
    tmp = out / "params.tmp"
    brax_model.save_params(str(tmp), params)
    shutil.copy(tmp, ckpt_dir / f"step_{step:011d}")
    os.replace(tmp, out / "params")          # atomic: a watcher never reads a half-written file
    (out / "latest.txt").write_text(str(step))


train_fn = functools.partial(
    ppo.train, **config,
    network_factory=functools.partial(ppo_networks.make_ppo_networks, **H.NETWORK),
    progress_fn=progress, policy_params_fn=save,
)
make_policy, params, _ = train_fn(environment=env, eval_env=eval_env)
brax_model.save_params(str(out / "params"), params)
print(f"Done in {(time.time() - t0) / 60:.1f} min. Watch it with:\n"
      f"  python sim/rl/play.py --run {args.name} --view")
