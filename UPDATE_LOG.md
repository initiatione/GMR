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
