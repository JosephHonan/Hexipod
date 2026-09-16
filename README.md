# IEE Hexapod

A six-legged (hexapod) walking robot project.

## Status

Early setup — control platform (Arduino/C++, Raspberry Pi/Python, or ROS/ROS2) is not finalized yet. This repo is structured to accommodate any of those without a rework.

## Repository layout

```
firmware/       Low-level microcontroller code (e.g. Arduino sketches, PlatformIO projects)
software/       Higher-level control code (e.g. Python, ROS/ROS2 packages, gait planners)
hardware/
  cad/          Mechanical design files (leg linkages, chassis, mounts)
  electrical/   Wiring diagrams, PCB designs, BOM
sim/            Simulation assets/configs (e.g. Gazebo, PyBullet, MATLAB)
docs/           Design notes, kinematics derivations, build logs
scripts/        Utility/setup scripts (build, flash, calibration)
tests/          Unit/integration tests
.github/workflows/  CI configs (once a platform is picked)
```

## Getting started

Once the control platform is chosen, add setup instructions here (toolchain, dependencies, how to flash/run).

## License

TBD.
