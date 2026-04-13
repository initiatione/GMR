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
