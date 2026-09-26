"""
Sanity-check the hexapod model: load it, hold the home pose for 3 s,
and report whether the robot stands and how hard the servos work.

    python sim/models/check_model.py
"""
from pathlib import Path

import mujoco
import numpy as np

XML = Path(__file__).with_name("hexapod.xml")

m = mujoco.MjModel.from_xml_path(str(XML))
d = mujoco.MjData(m)

print(f"Loaded {XML.name}: {m.nbody} bodies, {m.njnt} joints, {m.nu} actuators, "
      f"total mass {m.body_subtreemass[1]:.2f} kg")

mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
z0 = d.qpos[2]
limit = m.actuator_forcerange[:, 1]
peak = np.zeros(m.nu)

for _ in range(int(3.0 / m.opt.timestep)):
    mujoco.mj_step(m, d)
    peak = np.maximum(peak, np.abs(d.actuator_force))

z1 = d.qpos[2]
tilt = np.degrees(2 * np.arccos(np.clip(abs(d.qpos[3]), 0, 1)))
print(f"Body height: start {z0 * 1000:.0f} mm -> after 3 s {z1 * 1000:.0f} mm, tilt {tilt:.1f} deg")

names = [m.actuator(i).name for i in range(m.nu)]
worst = int(np.argmax(peak / limit))
print(f"Peak servo load: {peak[worst]:.2f} N*m on {names[worst]} "
      f"({100 * peak[worst] / limit[worst]:.0f}% of limit)")

ok = z1 > 0.6 * z0 and tilt < 10
print("PASS: robot holds its stance" if ok else "FAIL: robot sagged or tipped; check gains/masses")
