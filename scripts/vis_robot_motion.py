import argparse
import os
from pathlib import Path
import sys
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting.data_loader import load_robot_motion
from general_motion_retargeting.robot_motion_viewer import RobotMotionViewer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
                        
    parser.add_argument("--robot_motion_path", type=str, required=True)

    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, 
                        default="videos/example.mp4")
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_end", type=int, default=None)
                        
    args = parser.parse_args()
    
    robot_type = args.robot
    robot_motion_path = args.robot_motion_path
    
    if not os.path.exists(robot_motion_path):
        raise FileNotFoundError(f"Motion file {robot_motion_path} not found")
    
    motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list = load_robot_motion(robot_motion_path)
    frame_count = len(motion_root_pos)
    frame_start = int(args.frame_start)
    frame_end = frame_count if args.frame_end is None else int(args.frame_end)
    if frame_start < 0:
        raise ValueError("--frame_start must be greater than or equal to 0.")
    if frame_end > frame_count:
        raise ValueError(f"--frame_end ({frame_end}) exceeds motion frame count ({frame_count}).")
    if frame_start >= frame_end:
        raise ValueError(f"--frame_start ({frame_start}) must be smaller than --frame_end ({frame_end}).")
    
    env = RobotMotionViewer(robot_type=robot_type,
                            motion_fps=motion_fps,
                            camera_follow=False,
                            record_video=args.record_video, video_path=args.video_path)
    
    frame_idx = frame_start
    while True:
        env.step(motion_root_pos[frame_idx], 
                motion_root_rot[frame_idx], 
                motion_dof_pos[frame_idx], 
                rate_limit=True)
        frame_idx += 1
        if frame_idx >= frame_end:
            if args.record_video:
                break
            frame_idx = frame_start
    env.close()
