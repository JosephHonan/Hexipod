"""
Run a trained policy and compare it with the scripted tripod gait.

    python sim/rl/play.py --run <run_name>                 # headless, prints metrics
    python sim/rl/play.py --run <run_name> --view          # watch it in the viewer
    python sim/rl/play.py --run <run_name> --speed 0.12    # pick the commanded speed
"""
import argparse
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax.io import model as brax_model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from mujoco import mjx

import hexapod_env as H

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="folder name under sim/rl/runs/")
ap.add_argument("--speed", type=float, default=0.10)
ap.add_argument("--seconds", type=float, default=10.0)
ap.add_argument("--view", action="store_true")
args = ap.parse_args()

run_dir = Path(__file__).resolve().parent / "runs" / args.run
env = H.HexapodEnv()
nets = ppo_networks.make_ppo_networks(
    env.observation_size, env.action_size,
    preprocess_observations_fn=running_statistics.normalize, **H.NETWORK)
policy = jax.jit(ppo_networks.make_inference_fn(nets)(
    brax_model.load_params(str(run_dir / "params")), deterministic=True))
reset, step = jax.jit(env.reset), jax.jit(env.step)

state = reset(jax.random.PRNGKey(0))
state = state.replace(info={**state.info, "cmd": jnp.float32(args.speed)})
print("Compiling (first run takes a minute)...", flush=True)

m, d = env.mj_model, mujoco.MjData(env.mj_model)
viewer = None
if args.view:
    import mujoco.viewer
    viewer = mujoco.viewer.launch_passive(m, d)

key = jax.random.PRNGKey(1)
start_x = float(state.pipeline_state.qpos[0])
limit = m.actuator_forcerange[:, 1]
peak, max_tilt, falls = np.zeros(m.nu), 0.0, 0
n = int(args.seconds / H.CONTROL_DT)
for _ in range(n):
    if viewer is not None and not viewer.is_running():
        break
    wall = time.time()
    key, k = jax.random.split(key)
    action, _ = policy(state.obs, k)
    state = step(state, action)
    peak = np.maximum(peak, np.abs(np.asarray(state.pipeline_state.actuator_force)))
    qw = abs(float(state.pipeline_state.qpos[3]))
    max_tilt = max(max_tilt, float(np.degrees(2 * np.arccos(min(1.0, qw)))))
    if float(state.done):
        falls += 1
        break
    if viewer is not None:
        mjx.get_data_into(d, m, state.pipeline_state)
        viewer.sync()
        time.sleep(max(0.0, H.CONTROL_DT - (time.time() - wall)))

qpos = np.asarray(state.pipeline_state.qpos)
w = int(np.argmax(peak / limit))
print(f"Policy '{args.run}' at {args.speed:.3f} m/s for {args.seconds:.0f} s:")
print(f"  average speed:   {(qpos[0] - start_x) / args.seconds:.3f} m/s "
      f"({100 * (qpos[0] - start_x) / args.seconds / max(args.speed, 1e-9):.0f}% of command)")
print(f"  sideways drift:  {qpos[1] * 100:.1f} cm   max tilt: {max_tilt:.1f} deg")
print(f"  peak servo load: {peak[w]:.2f} N*m on {m.actuator(w).name} ({100 * peak[w] / limit[w]:.0f}% of limit)")
print("  FELL" if falls else "  stayed up the whole run")
print("Scripted tripod baseline at 0.10 m/s: 95% of command, 0.0 cm drift, 45% peak servo load")

if viewer is not None:
    print("Close the viewer window to exit.")
    while viewer.is_running():
        time.sleep(0.1)
    sys.stdout.flush()
    os._exit(0)
