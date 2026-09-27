"""
Watch the hexapod learn.

LIVE (run in a second terminal while train.py is running):
    python sim/rl/watch.py
  Shows the newest training run. Until the first checkpoint exists the robot
  uses the plain scripted tripod. Every time train.py saves a checkpoint, the
  robot switches to the new brain and the terminal reports how it walks.

REPLAY (after training, or during it):
    python sim/rl/watch.py --replay
  Plays every checkpoint in order, a few seconds each, so you can see the
  gait change from first to last.

Options: --run <name>  --speed 0.12  --secs 6 (replay seconds per checkpoint)

Runs on the CPU on purpose, so it never steals GPU memory from training.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")   # must be set before jax is imported

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402
from brax.io import model as brax_model  # noqa: E402
from brax.training.acme import running_statistics  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402

import hexapod_env as H  # noqa: E402

RUNS = Path(__file__).resolve().parent / "runs"

ap = argparse.ArgumentParser()
ap.add_argument("--run", help="run folder under sim/rl/runs/ (default: newest)")
ap.add_argument("--replay", action="store_true", help="play all checkpoints in order")
ap.add_argument("--speed", type=float, default=0.10, help="commanded speed, m/s")
ap.add_argument("--secs", type=float, default=6.0, help="replay: seconds per checkpoint")
args = ap.parse_args()


def newest_run():
    runs = [p for p in RUNS.glob("*") if p.is_dir()] if RUNS.exists() else []
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


print("Waiting for a training run to appear in sim/rl/runs/ ...", flush=True)
while True:
    run_dir = RUNS / args.run if args.run else newest_run()
    if run_dir is not None and run_dir.exists():
        break
    time.sleep(2)
print(f"Watching run: {run_dir.name}")

# ---- policy (same network shape as training) ----
env_obs, env_act = 63, 18
nets = ppo_networks.make_ppo_networks(
    env_obs, env_act, preprocess_observations_fn=running_statistics.normalize, **H.NETWORK)
make_policy = ppo_networks.make_inference_fn(nets)


@jax.jit
def act(params, obs):
    return make_policy(params, deterministic=True)(obs, jax.random.PRNGKey(0))[0]


gait = jax.jit(H.tripod_targets)

# ---- plain MuJoCo copy of the training robot (same solver settings as hexapod_env) ----
m = mujoco.MjModel.from_xml_path(str(H.MODELS / "hexapod.xml"))
m.opt.iterations, m.opt.ls_iterations = 4, 8
d = mujoco.MjData(m)
torso = m.body("torso").id
home = m.key("home").qpos.copy()
lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
n_sub = round(H.CONTROL_DT / m.opt.timestep)


class Robot:
    def reset(self):
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        self.phase, self.last = 0.0, np.zeros(18)
        self.x0, self.t = d.qpos[0], 0.0

    def obs(self):
        R = d.xmat[torso].reshape(3, 3)
        ph = 2 * np.pi * self.phase
        return np.concatenate([
            R.T @ np.array([0.0, 0.0, -1.0]), d.qvel[3:6] * 0.25,
            d.qpos[7:] - home[7:], d.qvel[6:] * 0.1, self.last,
            [args.speed * 5.0], [np.sin(ph), np.cos(ph)]]).astype(np.float32)

    def step(self, params):
        a = np.zeros(18) if params is None else np.asarray(act(params, self.obs()))
        a = np.clip(a, -1, 1)
        self.phase = (self.phase + H.CONTROL_DT / H.GAIT_PERIOD) % 1.0
        target = np.asarray(gait(self.phase, args.speed)) + H.RESIDUAL_SCALE * a
        d.ctrl[:] = np.clip(target, lo, hi)
        for _ in range(n_sub):
            mujoco.mj_step(m, d)
        self.last, self.t = a, self.t + H.CONTROL_DT
        R = d.xmat[torso].reshape(3, 3)
        return (R.T @ np.array([0, 0, -1.0]))[2] > -0.7 or d.qpos[2] < 0.08   # fell?

    def report(self, label):
        spd = (d.qpos[0] - self.x0) / max(self.t, 1e-9)
        print(f"  {label}: {spd:.3f} m/s ({100 * spd / max(args.speed, 1e-9):.0f}% of command), "
              f"drift {d.qpos[1] * 100:+.1f} cm", flush=True)


def load(path):
    for _ in range(5):                    # file may be mid-copy; retry briefly
        try:
            return brax_model.load_params(str(path))
        except Exception:
            time.sleep(0.5)
    return None


robot = Robot()
robot.reset()
viewer = mujoco.viewer.launch_passive(m, d)
viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
viewer.cam.trackbodyid = torso
viewer.cam.distance, viewer.cam.elevation = 1.2, -25


def tick(params):
    wall = time.time()
    if robot.step(params):
        print("  -> fell, resetting", flush=True)
        robot.reset()
    viewer.sync()
    time.sleep(max(0.0, H.CONTROL_DT - (time.time() - wall)))


if args.replay:
    ckpts = sorted((run_dir / "checkpoints").glob("step_*"))
    if not ckpts:
        sys.exit(f"No checkpoints in {run_dir / 'checkpoints'} yet.")
    print(f"Replaying {len(ckpts)} checkpoints, {args.secs:.0f} s each. Close the window to stop.")
    while viewer.is_running():
        for ck in [None] + ckpts:           # None = scripted tripod, the starting point
            label = "scripted tripod (before training)" if ck is None else f"step {int(ck.name[5:]):,}"
            params = None if ck is None else load(ck)
            robot.reset()
            while viewer.is_running() and robot.t < args.secs:
                tick(params)
            if not viewer.is_running():
                break
            robot.report(label)
else:
    latest, params, shown = run_dir / "latest.txt", None, None
    print("Showing the scripted tripod until the first checkpoint arrives "
          "(the first one takes a few minutes while JAX compiles).", flush=True)
    last_check = 0.0
    while viewer.is_running():
        if time.time() - last_check > 1.0:
            last_check = time.time()
            if latest.exists() and latest.read_text().strip() != shown:
                new = load(run_dir / "params")
                if new is not None:
                    if robot.t > 2.0:        # sum up the brain being replaced
                        robot.report(f"step {int(shown):,}" if shown else "scripted tripod")
                    shown, params = latest.read_text().strip(), new
                    print(f"New checkpoint: step {int(shown):,}", flush=True)
                    robot.reset()
        tick(params)
        if robot.t >= 10.0:                  # report every 10 s and restart the walk
            robot.report(f"step {int(shown):,}" if shown else "scripted tripod")
            robot.reset()

sys.stdout.flush()
os._exit(0)                                   # avoids the WSLg OpenGL teardown segfault
