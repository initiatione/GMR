import argparse
import pathlib
import os
import time

import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.motion_contact_postprocess import apply_contact_aware_postprocess, build_contact_aware_config
from general_motion_retargeting.motion_grounding import align_motion_root_to_ground
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

from rich import print

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smplx_file",
        help="SMPLX motion file to load.",
        type=str,
        # required=True,
        default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male1General_c3d/General_A1_-_Stand_stageii.npz",
        # default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male2MartialArtsKicks_c3d/G8_-__roundhouse_left_stageii.npz"
        # default="/home/yanjieze/projects/g1_wbc/TWIST-dev/motion_data/AMASS/KIT_572_dance_chacha11_stageii.npz"
        # default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male2MartialArtsPunches_c3d/E1_-__Jab_left_stageii.npz",
        # default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male1Running_c3d/Run_C24_-_quick_side_step_left_stageii.npz",
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1", 
                "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", "berkeley_humanoid_lite", "booster_k1",
                "pnd_adam_lite", "openloong", "tienkung", "fourier_gr3"],
        default="unitree_g1",
    )
    
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )

    parser.add_argument(
        "--record_video",
        default=False,
        action="store_true",
        help="Record the video.",
    )

    parser.add_argument(
        "--rate_limit",
        default=False,
        action="store_true",
        help="Limit the rate of the retargeted robot motion to keep the same as the human motion.",
    )

    parser.add_argument(
        "--foot-ground-align",
        dest="foot_ground_align",
        action="store_true",
        # 默认关闭，避免无提示地改训练分布。
        # 只有用户显式开启时，才把导出的 PKL 做贴地修复。
        help="Ground the saved PKL using actual robot support geoms.",
    )
    parser.add_argument(
        "--foot-ground-mode",
        choices=["per_frame", "global"],
        default="per_frame",
        help="Vertical grounding strategy applied to saved motion.",
    )
    parser.add_argument(
        "--foot-ground-clearance",
        type=float,
        default=0.002,
        help="Target minimum support clearance above ground in meters for saved motion.",
    )
    parser.add_argument(
        "--contact-aware-postprocess",
        action="store_true",
        default=False,
        help="Apply stance detection, stance-foot XY lock, support-geom grounding, and root_z smoothing to the saved PKL.",
    )
    parser.add_argument(
        "--contact-profile",
        choices=["conservative", "balanced", "aggressive"],
        default="balanced",
        # profile 是面对日常使用者的主入口；下面那些 threshold/window 参数只保留给 expert 调参。
        help="Preset strength for contact-aware postprocess. Expert flags below can still override individual thresholds.",
    )
    parser.add_argument(
        "--contact-stance-height-threshold",
        type=float,
        default=None,
        help="Expert override: frames with support min_z below this threshold can be treated as stance candidates.",
    )
    parser.add_argument(
        "--contact-stance-speed-threshold",
        type=float,
        default=None,
        help="Expert override: maximum planar foot speed (m/s) for stance detection.",
    )
    parser.add_argument(
        "--contact-stance-min-frames",
        type=int,
        default=None,
        help="Expert override: minimum segment length to keep a stance phase.",
    )
    parser.add_argument(
        "--contact-ground-mode",
        choices=["per_frame", "global"],
        default=None,
        help="Expert override: grounding mode used inside the contact-aware postprocess.",
    )
    parser.add_argument(
        "--contact-ground-clearance",
        type=float,
        default=None,
        help="Expert override: target minimum support clearance above ground in meters inside the contact-aware postprocess.",
    )
    parser.add_argument(
        "--contact-root-z-smoothing-window",
        type=int,
        default=None,
        help="Expert override: moving-average window used to smooth root_z correction inside the contact-aware postprocess.",
    )

    args = parser.parse_args()


    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
    
    
    # Load SMPLX trajectory
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER
    )
    
    # align fps
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
    
   
    # Initialize the retargeting system
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )
    
    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=aligned_fps,
                                            transparent_robot=0,
                                            record_video=args.record_video,
                                            video_path=f"videos/{args.robot}_{args.smplx_file.split('/')[-1].split('.')[0]}.mp4",)
    

    curr_frame = 0
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []
    
    # Start the viewer
    i = 0

    while True:
        if args.loop:
            i = (i + 1) % len(smplx_data_frames)
        else:
            i += 1
            if i >= len(smplx_data_frames):
                break
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
        
        # Update task targets.
        smplx_data = smplx_data_frames[i]

        # retarget
        qpos = retarget.retarget(smplx_data)

        # visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retarget.scaled_human_data,
            # human_motion_data=smplx_data,
            human_pos_offset=np.array([0.0, 0.0, 0.0]),
            show_human_body_name=False,
            rate_limit=args.rate_limit,
            follow_camera=False,
        )
        if args.save_path is not None:
            qpos_list.append(qpos)
            
    if args.save_path is not None:
        import pickle
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        local_body_pos = None
        body_names = None
        
        motion_data = {
            "fps": aligned_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
        }
        if args.contact_aware_postprocess:
            # 和 BVH 导出保持一致：profile 先给出一套稳定默认值，必要时再被 expert override 覆盖。
            motion_data, contact_stats = apply_contact_aware_postprocess(
                motion_data=motion_data,
                model_or_path=retarget.xml_file,
                config=build_contact_aware_config(
                    profile=args.contact_profile,
                    stance_height_threshold=args.contact_stance_height_threshold,
                    stance_speed_threshold=args.contact_stance_speed_threshold,
                    stance_min_frames=args.contact_stance_min_frames,
                    ground_clearance=args.contact_ground_clearance,
                    ground_mode=args.contact_ground_mode,
                    root_z_smoothing_window=args.contact_root_z_smoothing_window,
                ),
                inplace=False,
            )
            print("[contact_aware_postprocess]", contact_stats)
        elif args.foot_ground_align:
            # 旧接口仍保留，用于只想做最小 grounding 修复的场景。
            # 这是导出阶段的显式后处理，不参与 IK 本身。
            motion_data, grounding_stats = align_motion_root_to_ground(
                motion_data=motion_data,
                model_or_path=retarget.xml_file,
                clearance=args.foot_ground_clearance,
                mode=args.foot_ground_mode,
                inplace=False,
            )
            print("[foot_ground_align]", grounding_stats)
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")
            
      
    
    robot_motion_viewer.close()
