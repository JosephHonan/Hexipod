"""
Scripted tripod gait for the IEE Hexapod (no AI).

Legs LF, RM, LR swing while RF, LM, RR push, then they swap. Foot paths are
planned in the body frame and converted to joint angles with inverse
kinematics. Targets update at 50 Hz, the same rate the RL policy will use.

    python sim/gaits/tripod_gait.py             # headless run, prints metrics
    python sim/gaits/tripod_gait.py --view      # watch it walk in real time
    python sim/gaits/tripod_gait.py --speed 0.08 --period 1.2 --lift 0.04
"""
import argparse
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

MODELS = Path(__file__).resolve().parents[1] / "models"
sys.path.insert(0, str(MODELS))
import build_hexapod as P  # geometry lives in one place  # noqa: E402

LEG_NAMES = list(P.LEGS)                       # LF LM LR RF RM RR
GROUP_A = {"LF", "RM", "LR"}                   # tripod A; the rest are tripod B
CONTROL_DT = 0.02                              # 50 Hz


def leg_ik(x, y, z):
    """Foot target (leg frame, origin at coxa joint) -> (coxa, femur, tibia) rad."""
    C, F, T = P.COXA_LEN, P.FEMUR_LEN, P.TIBIA_LEN
    q1 = math.atan2(y, x)
    u = math.hypot(x, y) - C                  # horizontal reach from femur joint
    w = z
    d2 = u * u + w * w
    d2 = min(max(d2, (F - T) ** 2 + 1e-9), (F + T) ** 2 - 1e-9)  # stay reachable
    cb = (d2 - F * F - T * T) / (2 * F * T)
    beta = -math.acos(max(-1.0, min(1.0, cb)))                   # knee-up solution
    q2 = math.atan2(w, u) - math.atan2(T * math.sin(beta), F + T * math.cos(beta))
    q3 = beta + math.pi / 2
    return q1, q2, q3


def leg_fk(q1, q2, q3):
    """Joint angles -> foot position in leg frame (used for self-test)."""
    r = P.COXA_LEN + P.FEMUR_LEN * math.cos(q2) + P.TIBIA_LEN * math.sin(q2 + q3)
    z = P.FEMUR_LEN * math.sin(q2) - P.TIBIA_LEN * math.cos(q2 + q3)
    return r * math.cos(q1), r * math.sin(q1), z


def foot_offset(phase, step_len, lift):
    """Phase in [0,1): first half stance (foot slides back), second half swing."""
    if phase < 0.5:
        s = phase / 0.5
        return step_len * (0.5 - s), 0.0
    s = (phase - 0.5) / 0.5
    x = step_len * (-0.5 + 0.5 * (1 - math.cos(math.pi * s)))
    return x, lift * math.sin(math.pi * s)


def gait_targets(t, speed, period, lift):
    """Joint targets for all 18 actuators at time t (actuator order)."""
    step_len = speed * period / 2            # distance covered per stance
    reach = P.COXA_LEN + P.FEMUR_LEN          # nominal foot reach from coxa
    ctrl = []
    for name in LEG_NAMES:
        a = math.radians(P.LEGS[name])
        phase = (t / period + (0.0 if name in GROUP_A else 0.5)) % 1.0
        dx_body, dz = foot_offset(phase, step_len, lift)
        # body-frame forward offset -> leg frame (rotate by -a)
        lx = reach + dx_body * math.cos(a)
        ly = -dx_body * math.sin(a)
        ctrl.extend(leg_ik(lx, ly, -P.TIBIA_LEN + dz))
    return np.array(ctrl)


def run(args):
    m = mujoco.MjModel.from_xml_path(str(MODELS / "hexapod.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    n_sub = round(CONTROL_DT / m.opt.timestep)
    limit = m.actuator_forcerange[:, 1]
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]

    peak, max_tilt, clipped = np.zeros(m.nu), 0.0, 0
    start_xy = None
    viewer = mujoco.viewer.launch_passive(m, d) if args.view else None

    steps = int((args.settle + args.duration) / CONTROL_DT)
    for k in range(steps):
        t = k * CONTROL_DT
        if viewer is not None and not viewer.is_running():
            break
        walking = t >= args.settle
        if walking and start_xy is None:
            start_xy, start_t = d.qpos[:2].copy(), t
        ramp = min(1.0, (t - args.settle) / 1.0) if walking else 0.0   # 1 s speed ramp
        target = gait_targets(t - args.settle, args.speed * ramp, args.period,
                              args.lift if walking else 0.0)
        clipped += int(np.any((target < lo) | (target > hi)))
        d.ctrl[:] = np.clip(target, lo, hi)

        wall = time.time()
        for _ in range(n_sub):
            mujoco.mj_step(m, d)
            peak = np.maximum(peak, np.abs(d.actuator_force))
        max_tilt = max(max_tilt, math.degrees(2 * math.acos(min(1.0, abs(d.qpos[3])))))
        if viewer is not None:
            viewer.sync()
            time.sleep(max(0.0, CONTROL_DT - (time.time() - wall)))

    if start_xy is None:
        print("Stopped before walking started.")
        return viewer

    moved = d.qpos[:2] - start_xy
    elapsed = t + CONTROL_DT - start_t
    names = [m.actuator(i).name for i in range(m.nu)]
    w = int(np.argmax(peak / limit))
    fell = d.qpos[2] < 0.6 * (P.TIBIA_LEN + P.FOOT_RADIUS) or max_tilt > 30

    print(f"Walked {elapsed:.1f} s  (command {args.speed:.3f} m/s, period {args.period} s, "
          f"lift {args.lift * 1000:.0f} mm)")
    print(f"  forward distance: {moved[0] * 100:.1f} cm   sideways drift: {moved[1] * 100:.1f} cm")
    print(f"  average speed:    {moved[0] / elapsed:.3f} m/s "
          f"({100 * moved[0] / elapsed / max(args.speed, 1e-9):.0f}% of command)")
    print(f"  body height end:  {d.qpos[2] * 1000:.0f} mm   max tilt: {max_tilt:.1f} deg")
    print(f"  peak servo load:  {peak[w]:.2f} N*m on {names[w]} ({100 * peak[w] / limit[w]:.0f}% of limit)")
    if clipped:
        print(f"  WARNING: {clipped} control steps hit a joint limit (targets clipped)")
    print("FAIL: robot fell" if fell else
          "PASS: robot walked" if moved[0] > 0.5 * args.speed * elapsed else
          "WEAK: robot stayed up but made little progress (feet slipping?)")
    return viewer


def self_test():
    for target in [(0.155, 0.0, -0.14), (0.16, 0.03, -0.12), (0.13, -0.04, -0.15)]:
        got = leg_fk(*leg_ik(*target))
        assert np.allclose(got, target, atol=1e-6), (target, got)
    assert np.allclose(leg_ik(0.155, 0.0, -0.14), (0, 0, 0), atol=1e-6)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speed", type=float, default=0.10, help="forward speed, m/s")
    ap.add_argument("--period", type=float, default=1.0, help="full gait cycle, s")
    ap.add_argument("--lift", type=float, default=0.03, help="foot lift height, m")
    ap.add_argument("--duration", type=float, default=10.0, help="walking time, s")
    ap.add_argument("--settle", type=float, default=1.0, help="standing time before walking, s")
    ap.add_argument("--view", action="store_true", help="open the viewer and run in real time")
    args = ap.parse_args()
    if args.view:
        import mujoco.viewer  # noqa: F401
    self_test()
    viewer = run(args)
    if viewer is not None:
        # Leave the final pose on screen until the window is closed, then exit
        # hard: tearing down the OpenGL context under WSLg can segfault.
        print("Close the viewer window to exit.")
        while viewer.is_running():
            time.sleep(0.1)
        sys.stdout.flush()
        import os
        os._exit(0)
