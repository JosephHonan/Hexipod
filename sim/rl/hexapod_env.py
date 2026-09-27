"""
MJX training environment for the IEE Hexapod.

The policy does NOT learn to walk from scratch. Every control step (50 Hz)
the scripted tripod gait from sim/gaits/tripod_gait.py produces joint targets,
and the network outputs an 18-value correction added on top:

    joint target = tripod gait(phase, commanded speed) + RESIDUAL_SCALE * action

Observations (63 values, everything the real robot can measure):
    3  gravity direction in body frame     (IMU)
    3  body angular velocity               (IMU gyro)
    18 joint positions relative to home    (STS3215 feedback)
    18 joint velocities                    (STS3215 feedback)
    18 previous action
    1  commanded forward speed
    2  gait phase as sin/cos
"""
import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
from brax.envs.base import Env, State
from mujoco import mjx

MODELS = Path(__file__).resolve().parents[1] / "models"
sys.path.insert(0, str(MODELS))
import build_hexapod as P  # noqa: E402

# ------------------------------ settings ------------------------------
CONTROL_DT = 0.02            # 50 Hz, same as the real robot's policy loop
GAIT_PERIOD = 1.0            # s per full tripod cycle
GAIT_LIFT = 0.03             # m foot lift
RESIDUAL_SCALE = 0.3         # rad: max correction the policy can add per joint
CMD_RANGE = (0.0, 0.15)      # m/s forward speed commands sampled per episode
OBS_NOISE = 0.02             # sensor noise std (after scaling)
STAND_HEIGHT = 0.148         # m, measured by tripod_gait.py
NETWORK = dict(policy_hidden_layer_sizes=(128, 128, 128),
               value_hidden_layer_sizes=(256, 256, 256))

REWARD_WEIGHTS = dict(
    track=1.5,        # follow commanded forward speed
    straight=0.5,     # no sideways slide, no spinning
    tilt=-2.0,        # keep body level
    height=-20.0,     # keep ride height
    torque=-2e-3,     # energy / servo heat
    action_rate=-0.02,  # smooth commands (servos and gears like this)
)
TERMINATION_PENALTY = 10.0
# ----------------------------------------------------------------------

_LEG_ANG = jnp.array([math.radians(a) for a in P.LEGS.values()])
_OFFSETS = jnp.array([0.0 if n in {"LF", "RM", "LR"} else 0.5 for n in P.LEGS])
_REACH = P.COXA_LEN + P.FEMUR_LEN


def leg_ik(x, y, z):
    """Vectorized IK: foot targets (leg frames) -> coxa, femur, tibia angles."""
    C, F, T = P.COXA_LEN, P.FEMUR_LEN, P.TIBIA_LEN
    q1 = jnp.arctan2(y, x)
    u = jnp.hypot(x, y) - C
    d2 = jnp.clip(u * u + z * z, (F - T) ** 2 + 1e-9, (F + T) ** 2 - 1e-9)
    beta = -jnp.arccos(jnp.clip((d2 - F * F - T * T) / (2 * F * T), -1.0, 1.0))
    q2 = jnp.arctan2(z, u) - jnp.arctan2(T * jnp.sin(beta), F + T * jnp.cos(beta))
    return q1, q2, beta + jnp.pi / 2


def tripod_targets(phase, cmd):
    """Scripted tripod gait -> 18 joint targets, actuator order."""
    step_len = cmd * GAIT_PERIOD / 2
    ph = (phase + _OFFSETS) % 1.0
    stance = ph < 0.5
    s_st, s_sw = ph / 0.5, (ph - 0.5) / 0.5
    dx = jnp.where(stance, step_len * (0.5 - s_st),
                   step_len * (-0.5 + 0.5 * (1 - jnp.cos(jnp.pi * s_sw))))
    dz = jnp.where(stance, 0.0, GAIT_LIFT * jnp.sin(jnp.pi * s_sw))
    lx = _REACH + dx * jnp.cos(_LEG_ANG)
    ly = -dx * jnp.sin(_LEG_ANG)
    q1, q2, q3 = leg_ik(lx, ly, -P.TIBIA_LEN + dz)
    return jnp.stack([q1, q2, q3], axis=1).reshape(-1)


class HexapodEnv(Env):
    def __init__(self):
        mj = mujoco.MjModel.from_xml_path(str(MODELS / "hexapod.xml"))
        mj.opt.iterations = 4          # constraint solver: plenty for this contact count
        mj.opt.ls_iterations = 8
        try:
            self._mjx = mjx.put_model(mj)
        except Exception as e:         # older MJX builds lack some features
            print(f"[hexapod_env] MJX rejected the model ({e}); "
                  "dropping joint frictionloss and using Euler integration.")
            mj.dof_frictionloss[:] = 0.0
            mj.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
            self._mjx = mjx.put_model(mj)
        self.mj_model = mj
        self._n_sub = round(CONTROL_DT / mj.opt.timestep)
        self._home = jnp.array(mj.key("home").qpos)
        self._torso = mj.body("torso").id
        self._lo = jnp.array(mj.actuator_ctrlrange[:, 0])
        self._hi = jnp.array(mj.actuator_ctrlrange[:, 1])

    # --- brax Env interface ---
    @property
    def observation_size(self):
        return 63

    @property
    def action_size(self):
        return 18

    @property
    def backend(self):
        return "mjx"

    def reset(self, rng):
        rng, k_q, k_cmd, k_ph, k_obs = jax.random.split(rng, 5)
        qpos = self._home.at[7:].add(jax.random.uniform(k_q, (18,), minval=-0.05, maxval=0.05))
        data = mjx.make_data(self._mjx).replace(
            qpos=qpos, qvel=jnp.zeros(self._mjx.nv),
            ctrl=jnp.clip(qpos[7:], self._lo, self._hi))
        data = mjx.forward(self._mjx, data)
        info = {
            "rng": rng,
            "cmd": jax.random.uniform(k_cmd, (), minval=CMD_RANGE[0], maxval=CMD_RANGE[1]),
            "phase": jax.random.uniform(k_ph, ()),
            "last_act": jnp.zeros(18),
        }
        metrics = {k: jnp.zeros(()) for k in (*REWARD_WEIGHTS, "fwd_speed")}
        return State(data, self._obs(data, info, k_obs), jnp.zeros(()), jnp.zeros(()),
                     metrics, info)

    def step(self, state, action):
        action = jnp.clip(action, -1.0, 1.0)
        info = state.info
        phase = (info["phase"] + CONTROL_DT / GAIT_PERIOD) % 1.0
        target = tripod_targets(phase, info["cmd"]) + RESIDUAL_SCALE * action
        data = state.pipeline_state.replace(ctrl=jnp.clip(target, self._lo, self._hi))
        data = jax.lax.fori_loop(0, self._n_sub, lambda _, d: mjx.step(self._mjx, d), data)

        rng, k_obs = jax.random.split(info["rng"])
        new_info = {**info, "rng": rng, "phase": phase, "last_act": action}

        R = data.xmat[self._torso]
        grav = R.T @ jnp.array([0.0, 0.0, -1.0])
        vel = R.T @ data.qvel[0:3]
        yaw_rate = data.qvel[5]
        terms = {
            "track": jnp.exp(-jnp.square(vel[0] - info["cmd"]) / 0.005),
            "straight": jnp.exp(-(jnp.square(vel[1]) + jnp.square(yaw_rate)) / 0.05),
            "tilt": jnp.square(grav[0]) + jnp.square(grav[1]),
            "height": jnp.square(data.qpos[2] - STAND_HEIGHT),
            "torque": jnp.sum(jnp.square(data.actuator_force)),
            "action_rate": jnp.sum(jnp.square(action - info["last_act"])),
        }
        reward = sum(REWARD_WEIGHTS[k] * v for k, v in terms.items())
        fell = (grav[2] > -0.7) | (data.qpos[2] < 0.08)
        done = fell.astype(jnp.float32)
        reward = reward - TERMINATION_PENALTY * done
        metrics = {**state.metrics, **terms, "fwd_speed": vel[0]}
        return state.replace(pipeline_state=data, obs=self._obs(data, new_info, k_obs),
                             reward=reward, done=done, metrics=metrics, info=new_info)

    def _obs(self, data, info, key):
        R = data.xmat[self._torso]
        sensors = jnp.concatenate([
            R.T @ jnp.array([0.0, 0.0, -1.0]),     # gravity in body frame
            data.qvel[3:6] * 0.25,                  # gyro (free joint ang vel is body frame)
            data.qpos[7:] - self._home[7:],         # joint positions
            data.qvel[6:] * 0.1,                    # joint velocities
        ])
        sensors = sensors + OBS_NOISE * jax.random.normal(key, sensors.shape)
        ph = 2 * jnp.pi * info["phase"]
        return jnp.concatenate([sensors, info["last_act"], jnp.array([info["cmd"] * 5.0]),
                                jnp.array([jnp.sin(ph), jnp.cos(ph)])])
