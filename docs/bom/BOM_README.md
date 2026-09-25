# Bill of Materials

`IEE_Hexapod_Parts_List.xlsx` — full parts list with pricing, category subtotals, and sourcing notes.
Blue cells are editable inputs (Qty, Est. Unit Price); line totals, subtotals, and the grand total are formulas.
The **Status** column marks each line as Kept, NEW, or Changed, and the **Change Log** tab records what was
removed in the latest revision and why.

Current estimated total: **~$1,485** (servo pricing is the biggest variable — see notes in the sheet).

## Architecture (revised Sept 2026)

The hardware is designed around deploying a locomotion policy trained in simulation (MuJoCo + PPO), so the
robot has to reproduce the sim's observations, actions, and timing as closely as possible.

- **Raspberry Pi 5 (8GB)** — main compute. Runs the locomotion policy (small MLP on CPU via ONNX Runtime),
  navigation, and orchestration.
- **Raspberry Pi AI HAT+ (26 TOPS, Hailo-8)** — perception accelerator for camera-based models. Replaces the
  earlier Coral USB plan (better supported on Pi 5, ~6x the throughput). Occupies the PCIe slot, so the Pi
  boots from microSD.
- **Teensy 4.1 real-time controller** — runs a fixed-rate loop (100–200 Hz) for the servo buses and IMU,
  exchanges state/commands with the Pi over USB serial, and acts as a watchdog that puts the robot in a safe
  crouch if the Pi stops responding.
- **18x Feetech STS3215 smart servos (12V, 30 kg-cm)** — 3 DOF/leg x 6 legs, on three half-duplex serial bus
  chains (two legs per chain). Report position, speed, load, voltage, and temperature, which the RL policy
  needs as observations. Replaces the earlier open-loop PWM servos + PCA9685 drivers.
- **Sensors** — BNO085 IMU (body orientation/angular velocity for the policy), Pi Camera Module 3,
  RPLIDAR A1 (provisional), and 4x VL53L0X time-of-flight sensors for cliff/step detection (provisional).
- **Power** — 3S LiPo (6000mAh) feeding the 12V servo rail directly, with a 40A main fuse, physical kill
  switch, and INA226 voltage/current monitor. A separate 5V 5A buck converter isolates the Pi from servo
  current spikes.

## Still provisional

LIDAR, ToF sensor count/placement, and chassis cost will be finalized once the chassis and leg layout are
designed. Servo torque should be re-checked against final femur length from the leg CAD.

Previous BOM versions (Coral TPU / PCA9685 / PWM servo architecture) are preserved in this file's Git history.
