# Training-Ready Official Retarget Outputs

Current accepted official BVH -> T800 outputs:

## raw_pkl

- `raw_pkl/accepted_raw/zhiquan_quanji_001_raw.pkl`
- `raw_pkl/accepted_raw/540huixuantitui_001_raw.pkl`

Use these as the source of truth for the next standard training `.npz` conversion stage.

`raw_pkl/grounded_global_candidates/` contains grounded postprocess candidates. Keep them separate from accepted raw inputs until a later replay/contact validation promotes one.

## core32_npy

- `core32_npy/zhiquan_quanji_001_core32.npy`
- `core32_npy/540huixuantitui_001_core32.npy`

These are compact intermediate files for schema/FK bridge checks. They preserve GMR's `root_rot_xyzw` order and are not the final training `.npz` format.
