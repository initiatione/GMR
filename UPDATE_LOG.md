# Update Log

## 2026-04-13

- Target environment: `conda robot`
- Installed runtime packages for current GMR local tuning: `mink`, `rich`, `imageio`, `torch`, `smplx`
- Installed project in editable mode: `python -m pip install -e . --no-deps`
- Cleared broken pip cache once and retried installation after `proxsuite` wheel CRC failure
- Verified GMR import in `robot` environment
- Verified `bvh_lafan1 -> t800` smoke test with `ik_config_manager/TPOSE.bvh`
- Smoke test result: successfully created `GeneralMotionRetargeting(src_human='bvh_lafan1', tgt_robot='t800')` and produced first-frame `qpos` with length `32`
- Optional component still not installed: `xrobotoolkit_sdk` (not required for current local BVH/T800 ik_config tuning)
- Added a visible and collidable floor plane to `assets/t800/mujoco/t800_full_gmr.xml` for local MuJoCo debugging and IK tuning.
- Added `contact exclude` rules in `assets/t800/mujoco/t800_full_gmr.xml` to suppress the initial `LINK_BASE` vs `LINK_HIP_ROLL_{L,R}` self-collision that caused first-frame joint explosion.
- Added a gradient skybox, brighter headlight settings, and a top scene light to `assets/t800/mujoco/t800_full_gmr.xml` to improve local IK tuning visibility.
- Copied local LAFAN1 motion file `aiming1_subject1.bvh` to workspace root as `D:\human_robot\aiming1_subject1.bvh` for T800 ik_config tuning.
- Added `scripts/adjust_xml_transparency.py`, a generic XML transparency utility that reads an existing XML and writes a new transparent XML file; verified by generating `D:\human_robot\t800_full_gmr_transparent.xml` from `assets/t800/mujoco/t800_full_gmr.xml` with alpha `0.22`.
- Added `t800_transparent` to the `bvh_to_robot.py` robot choices and renamed the transparent MuJoCo model title to `t800_full_gmr_transparent` for easier debug-window identification.
