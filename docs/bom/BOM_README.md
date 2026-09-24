# Bill of Materials

`IEE_Hexapod_Parts_List.xlsx` — full parts list with pricing, category subtotals, and sourcing notes.
Blue cells are editable inputs (Qty, Est. Unit Price); line totals and the grand total are formulas.

Current architecture assumption: Raspberry Pi 5 + Coral USB Accelerator (Edge TPU) for on-device
AI-driven movement, 18x digital servos (3 DOF/leg x 6 legs) via two PCA9685 driver boards, plus a
2D LIDAR (RPLIDAR A1) and per-leg time-of-flight sensors for obstacle/cliff detection.

Sensor counts/placement are provisional — revisit once the chassis and leg layout are designed.
