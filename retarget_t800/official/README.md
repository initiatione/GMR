# Official T800 Retarget Outputs

This directory separates accepted outputs for the training-data pipeline from local experiment artifacts.

## training_ready

Use these files as the current accepted GMR outputs for the next conversion stage:

- `training_ready/raw_pkl/accepted_raw/zhiquan_quanji_001_raw.pkl`
- `training_ready/raw_pkl/accepted_raw/540huixuantitui_001_raw.pkl`
- `training_ready/core32_npy/zhiquan_quanji_001_core32.npy`
- `training_ready/core32_npy/540huixuantitui_001_core32.npy`

The raw `.pkl` files are the authoritative retarget outputs for continuing toward standard training `.npz` conversion. The `core32_npy` files are compact intermediate checks: `root_pos(3) + root_rot_xyzw(4) + dof_pos(25)`.

`training_ready/raw_pkl/grounded_global_candidates/` contains grounded variants generated from accepted raw files. Treat them as postprocess candidates until replay/contact checks explicitly promote them.

## experiments

Files under `experiments/` are diagnostic or historical artifacts only:

- `short_windows/`: short-frame root, scale, and route probes.
- `manual_tuning/`: manual IK tuning snapshots.
- `right_hand_diagnostics/`: sampled full-motion right-hand diagnosis variants.

Do not feed experiment files into training conversion unless a later change explicitly promotes them.
