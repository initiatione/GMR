import argparse
import pathlib
import time
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.motion_grounding import align_motion_root_to_ground
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from rich import print
from tqdm import tqdm
import os
import numpy as np

def estimate_ground_offset(retargeter: GMR, motion_frames):
    """Estimate a source-side global z offset from the human motion itself.

    这是“重定向前”的 auto ground：
    - 观测的是源人体关键点最低点；
    - 调整的是输入 human motion 的整体高度。

    它和后面的 `foot_ground_align` 不是一回事：
    - auto_ground: 先把源动作大致摆正；
    - foot_ground_align: 重定向结束后，再按机器人真实支撑碰撞体做一次校准。
    """
    lowest_z = np.inf
    for human_data in motion_frames:
        human_data = retargeter.to_numpy(human_data)
        human_data = retargeter.scale_human_data(
            human_data,
            retargeter.human_root_name,
            retargeter.human_scale_table,
        )
        human_data = retargeter.offset_human_data(
            human_data,
            retargeter.pos_offsets1,
            retargeter.rot_offsets1,
        )
        for pos, _ in human_data.values():
            if pos[2] < lowest_z:
                lowest_z = pos[2]

    return float(lowest_z)

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bvh_file",
        help="BVH motion file to load.",
        required=True,
        type=str,
    )
    
    parser.add_argument(
        "--format",
        choices=["lafan1", "nokov"],
        default="lafan1",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )
    
    parser.add_argument(
        "--robot",
        # 这里把 T800 也接入到 BVH retarget 入口。
        # 这样在命令行里就可以直接使用 `--robot t800` 或 `--robot t800_transparent`
        # 走完整的 GMR BVH 重定向流程。
        choices=["unitree_g1", "unitree_g1_with_hands", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01", "pal_talos", "t800", "t800_transparent"],
        default="unitree_g1",
    )
    
    
    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default="videos/example.mp4",
    )

    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    
    parser.add_argument(
        "--motion_fps",
        default=30,
        type=int,
    )

    parser.add_argument(
        "--auto_ground",
        action="store_true",
        default=False,
        help="Automatically offset source motion to ground before retargeting.",
    )

    parser.add_argument(
        "--auto_ground_margin",
        type=float,
        default=0.0,
        help="Target clearance above ground (meters) when --auto_ground is enabled.",
    )

    parser.add_argument(
        "--debug_log_path",
        type=str,
        default=None,
        help="可选：把每帧 IK 数值误差与 target/current frame 信息写成 jsonl 调试日志。",
    )

    parser.add_argument(
        "--debug_log_every_n",
        type=int,
        default=1,
        help="调试日志采样间隔。默认每帧都记录；设为 10 表示每 10 帧记录一次。",
    )

    parser.add_argument(
        "--foot-ground-align",
        dest="foot_ground_align",
        action="store_true",
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
    
    args = parser.parse_args()
    
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    
    # Load SMPLX trajectory
    lafan1_data_frames, actual_human_height = load_bvh_file(args.bvh_file, format=args.format)
    
    
    # Initialize the retargeting system
    retargeter = GMR(
        src_human=f"bvh_{args.format}",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
        debug_log_path=args.debug_log_path,
        debug_log_every_n=args.debug_log_every_n,
    )

    if args.auto_ground:
        estimated_lowest_z = estimate_ground_offset(retargeter, lafan1_data_frames)
        if not np.isfinite(estimated_lowest_z):
            raise RuntimeError("Failed to estimate ground offset from BVH frames.")
        applied_ground_offset = estimated_lowest_z - args.auto_ground_margin
        retargeter.set_ground_offset(applied_ground_offset)
        print(
            "[auto_ground] "
            f"min_z={estimated_lowest_z:.6f}, "
            f"margin={args.auto_ground_margin:.6f}, "
            f"applied_ground_offset={applied_ground_offset:.6f}"
        )

    motion_fps = args.motion_fps
    
    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=motion_fps,
                                            transparent_robot=0,
                                            record_video=args.record_video,
                                            video_path=args.video_path,
                                            # video_width=2080,
                                            # video_height=1170
                                            )
    
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    print(f"mocap_frame_rate: {motion_fps}")
    
    # Create tqdm progress bar for the total number of frames
    pbar = tqdm(total=len(lafan1_data_frames), desc="Retargeting")
    
    # Start the viewer
    i = 0
    


    while True:
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
            
        # Update progress bar
        pbar.update(1)

        # Update task targets.
        smplx_data = lafan1_data_frames[i]

        # retarget
        qpos = retargeter.retarget(smplx_data, frame_index=i)
        

        # visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retargeter.scaled_human_data,
            rate_limit=args.rate_limit,
            follow_camera=True,
            # human_pos_offset=np.array([0.0, 0.0, 0.0])
        )

        if args.loop:
            i = (i + 1) % len(lafan1_data_frames)
        else:
            i += 1
            if i >= len(lafan1_data_frames):
                break
   
        
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
            "fps": motion_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
        }
        if args.foot_ground_align:
            # 注意：这里只在“保存 pkl”时做后处理，不会影响前面的 viewer 回放。
            # 这样方便你先看原始 retarget 效果，再决定是否对训练数据做 grounding 修复。
            motion_data, grounding_stats = align_motion_root_to_ground(
                motion_data=motion_data,
                model_or_path=retargeter.xml_file,
                clearance=args.foot_ground_clearance,
                mode=args.foot_ground_mode,
                inplace=False,
            )
            print("[foot_ground_align]", grounding_stats)
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    # Close progress bar
    pbar.close()
    
    robot_motion_viewer.close()
       
