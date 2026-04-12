
import mink
import mujoco as mj
import numpy as np
import json
import os
from scipy.spatial.transform import Rotation as R
from .params import ROBOT_XML_DICT, IK_CONFIG_DICT
from rich import print

class GeneralMotionRetargeting:
    """General Motion Retargeting (GMR).
    """
    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        actual_human_height: float = None,
        solver: str="daqp", # change from "quadprog" to "daqp".
        damping: float=5e-1, # change from 1e-1 to 1e-2.
        verbose: bool=True,
        use_velocity_limit: bool=False,
        debug_log_path: str | None = None,
        debug_log_every_n: int = 1,
    ) -> None:

        # load the robot model
        self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
        if verbose:
            print("Use robot model: ", self.xml_file)
        self.model = mj.MjModel.from_xml_path(self.xml_file)
        
        # Print DoF names in order
        print("[GMR] Robot Degrees of Freedom (DoF) names and their order:")
        self.robot_dof_names = {}
        self.robot_dof_names_in_order = []
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            self.robot_dof_names[dof_name] = i
            self.robot_dof_names_in_order.append(dof_name)
            if verbose:
                print(f"DoF {i}: {dof_name}")
            
            
        print("[GMR] Robot Body names and their IDs:")
        self.robot_body_names = {}
        for i in range(self.model.nbody):  # 'nbody' is the number of bodies
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i
            if verbose:
                print(f"Body ID {i}: {body_name}")
        
        print("[GMR] Robot Motor (Actuator) names and their IDs:")
        self.robot_motor_names = {}
        for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            self.robot_motor_names[motor_name] = i
            if verbose:
                print(f"Motor ID {i}: {motor_name}")

        # Load the IK config
        self.ik_config_path = IK_CONFIG_DICT[src_human][tgt_robot]
        with open(self.ik_config_path) as f:
            ik_config = json.load(f)
        if verbose:
            print("Use IK config: ", self.ik_config_path)
        
        # compute the scale ratio based on given human height and the assumption in the IK config
        if actual_human_height is not None:
            ratio = actual_human_height / ik_config["human_height_assumption"]
        else:
            ratio = 1.0
            
        # adjust the human scale table
        for key in ik_config["human_scale_table"].keys():
            ik_config["human_scale_table"][key] = ik_config["human_scale_table"][key] * ratio
    

        # used for retargeting
        self.ik_match_table1 = ik_config["ik_match_table1"]
        self.ik_match_table2 = ik_config["ik_match_table2"]
        self.human_root_name = ik_config["human_root_name"]
        self.robot_root_name = ik_config["robot_root_name"]
        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
        self.human_scale_table = ik_config["human_scale_table"]
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])

        self.max_iter = 10

        self.solver = solver
        self.damping = damping
        self.tgt_robot = tgt_robot
        self.src_human = src_human

        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.task1_body_to_frame = {}
        self.task2_body_to_frame = {}
        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}
        self.latest_task_targets1 = {}
        self.latest_task_targets2 = {}

        self.task_errors1 = {}
        self.task_errors2 = {}

        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        if use_velocity_limit:
            VELOCITY_LIMITS = {k: 3*np.pi for k in self.robot_motor_names.keys()}
            self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS)) 
            
        self.setup_retarget_configuration()
        
        self.ground_offset = 0.0
        self.debug_log_path = debug_log_path
        self.debug_log_every_n = max(1, int(debug_log_every_n))
        self.debug_frame_counter = 0

        # 如果开启了调试日志，这里先清空旧文件，确保本次运行得到的是一份干净的 jsonl 记录。
        if self.debug_log_path is not None:
            debug_log_dir = os.path.dirname(self.debug_log_path)
            if debug_log_dir:
                os.makedirs(debug_log_dir, exist_ok=True)
            with open(self.debug_log_path, "w", encoding="utf-8") as _:
                pass
            if verbose:
                print(f"[GMR] Debug log enabled: {self.debug_log_path}")

    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)
    
        self.tasks1 = []
        self.tasks2 = []
        
        for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task1[body_name] = task
                self.task1_body_to_frame[body_name] = frame_name
                self.pos_offsets1[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets1[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks1.append(task)
                self.task_errors1[task] = []
        
        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task2[body_name] = task
                self.task2_body_to_frame[body_name] = frame_name
                self.pos_offsets2[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets2[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks2.append(task)
                self.task_errors2[task] = []

  
    def update_targets(self, human_data, offset_to_ground=False):
        # scale human data in local frame
        human_data = self.to_numpy(human_data)
        human_data = self.scale_human_data(human_data, self.human_root_name, self.human_scale_table)
        human_data = self.offset_human_data(human_data, self.pos_offsets1, self.rot_offsets1)
        human_data = self.apply_ground_offset(human_data)
        if offset_to_ground:
            human_data = self.offset_human_data_to_ground(human_data)
        self.scaled_human_data = human_data

        if self.use_ik_match_table1:
            for body_name in self.human_body_to_task1.keys():
                task = self.human_body_to_task1[body_name]
                pos, rot = human_data[body_name]
                self.latest_task_targets1[body_name] = {
                    "frame_name": self.task1_body_to_frame[body_name],
                    "target_pos": np.asarray(pos).copy(),
                    "target_quat": np.asarray(rot).copy(),
                }
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
        
        if self.use_ik_match_table2:
            for body_name in self.human_body_to_task2.keys():
                task = self.human_body_to_task2[body_name]
                pos, rot = human_data[body_name]
                self.latest_task_targets2[body_name] = {
                    "frame_name": self.task2_body_to_frame[body_name],
                    "target_pos": np.asarray(pos).copy(),
                    "target_quat": np.asarray(rot).copy(),
                }
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
            
            
    def retarget(self, human_data, offset_to_ground=False, frame_index: int | None = None):
        # Update the task targets
        self.update_targets(human_data, offset_to_ground)

        if frame_index is None:
            frame_index = self.debug_frame_counter
            self.debug_frame_counter += 1

        initial_error1 = None
        final_error1 = None
        initial_error2 = None
        final_error2 = None
        num_iter1 = 0
        num_iter2 = 0

        if self.use_ik_match_table1:
            # Solve the IK problem
            curr_error = self.error1()
            initial_error1 = float(curr_error)
            dt = self.configuration.model.opt.timestep
            vel1 = mink.solve_ik(
                self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits
            )
            self.configuration.integrate_inplace(vel1, dt)
            next_error = self.error1()
            num_iter1 = 0
            while curr_error - next_error > 0.001 and num_iter1 < self.max_iter:
                curr_error = next_error
                dt = self.configuration.model.opt.timestep
                vel1 = mink.solve_ik(
                    self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits
                )
                self.configuration.integrate_inplace(vel1, dt)
                next_error = self.error1()
                num_iter1 += 1
            final_error1 = float(next_error)

        if self.use_ik_match_table2:
            curr_error = self.error2()
            initial_error2 = float(curr_error)
            dt = self.configuration.model.opt.timestep
            vel2 = mink.solve_ik(
                self.configuration, self.tasks2, dt, self.solver, self.damping, self.ik_limits
            )
            self.configuration.integrate_inplace(vel2, dt)
            next_error = self.error2()
            num_iter2 = 0
            while curr_error - next_error > 0.001 and num_iter2 < self.max_iter:
                curr_error = next_error
                # Solve the IK problem with the second task
                dt = self.configuration.model.opt.timestep
                vel2 = mink.solve_ik(
                    self.configuration, self.tasks2, dt, self.solver, self.damping, self.ik_limits
                )
                self.configuration.integrate_inplace(vel2, dt)
                
                next_error = self.error2()
                num_iter2 += 1
            final_error2 = float(next_error)

        if self.debug_log_path is not None and frame_index % self.debug_log_every_n == 0:
            self.write_debug_log(
                frame_index=frame_index,
                initial_error1=initial_error1,
                final_error1=final_error1,
                initial_error2=initial_error2,
                final_error2=final_error2,
                num_iter1=num_iter1,
                num_iter2=num_iter2,
            )
                
            
        return self.configuration.data.qpos.copy()


    def error1(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks1]
            )
        )
    
    def error2(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks2]
            )
        )


    def to_numpy(self, human_data):
        for body_name in human_data.keys():
            human_data[body_name] = [np.asarray(human_data[body_name][0]), np.asarray(human_data[body_name][1])]
        return human_data


    def scale_human_data(self, human_data, human_root_name, human_scale_table):
        
        human_data_local = {}
        root_pos, root_quat = human_data[human_root_name]
        
        # scale root
        scaled_root_pos = human_scale_table[human_root_name] * root_pos
        
        # scale other body parts in local frame
        for body_name in human_data.keys():
            if body_name not in human_scale_table:
                continue
            if body_name == human_root_name:
                continue
            else:
                # transform to local frame (only position)
                human_data_local[body_name] = (human_data[body_name][0] - root_pos) * human_scale_table[body_name]
            
        # transform the human data back to the global frame
        human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data_local.keys():
            human_data_global[body_name] = (human_data_local[body_name] + scaled_root_pos, human_data[body_name][1])

        return human_data_global
    
    def offset_human_data(self, human_data, pos_offsets, rot_offsets):
        """the pos offsets are applied in the local frame"""
        offset_human_data = {}
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            # apply rotation offset first
            updated_quat = (R.from_quat(quat, scalar_first=True) * rot_offsets[body_name]).as_quat(scalar_first=True)
            offset_human_data[body_name][1] = updated_quat
            
            local_offset = pos_offsets[body_name]
            # compute the global position offset using the updated rotation
            global_pos_offset = R.from_quat(updated_quat, scalar_first=True).apply(local_offset)
            
            offset_human_data[body_name][0] = pos + global_pos_offset
           
        return offset_human_data
            
    def offset_human_data_to_ground(self, human_data):
        """find the lowest point of the human data and offset the human data to the ground"""
        offset_human_data = {}
        ground_offset = 0.1
        lowest_pos = np.inf

        for body_name in human_data.keys():
            # only consider the foot/Foot
            if "Foot" not in body_name and "foot" not in body_name:
                continue
            pos, quat = human_data[body_name]
            if pos[2] < lowest_pos:
                lowest_pos = pos[2]
                lowest_body_name = body_name
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            offset_human_data[body_name][0] = pos - np.array([0, 0, lowest_pos]) + np.array([0, 0, ground_offset])
        return offset_human_data

    def set_ground_offset(self, ground_offset):
        self.ground_offset = ground_offset

    def apply_ground_offset(self, human_data):
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            human_data[body_name][0] = pos - np.array([0, 0, self.ground_offset])
        return human_data

    def get_robot_body_pose(self, frame_name):
        """读取当前机器人某个 body 在世界坐标系下的位姿。

        这里直接从 Mujoco 当前配置里拿 `xpos/xquat`，
        这样记录下来的就是 IK 求解后机器人真实落到的目标 body 位姿。
        """

        if frame_name not in self.robot_body_names:
            return None

        body_id = self.robot_body_names[frame_name]
        body_pos = np.asarray(self.configuration.data.xpos[body_id]).copy()
        body_quat = np.asarray(self.configuration.data.xquat[body_id]).copy()
        return body_pos, body_quat

    def compute_quaternion_angle_error_deg(self, target_quat, current_quat):
        """计算两个四元数之间的最小旋转角度，单位为度。"""

        target_rotation = R.from_quat(target_quat, scalar_first=True)
        current_rotation = R.from_quat(current_quat, scalar_first=True)
        delta_rotation = target_rotation.inv() * current_rotation
        return float(np.degrees(delta_rotation.magnitude()))

    def collect_task_debug_info(self, body_to_task, body_to_frame, latest_targets):
        """收集当前 task 集合的逐项误差明细。

        输出里同时保留三类信息：
        1. Mink `task.compute_error()` 的原始误差向量；
        2. target frame 与当前 robot frame 的位置差；
        3. target frame 与当前 robot frame 的姿态角差（度）。

        这样后续排查时就能区分：
        - 是目标位姿本身就设置错了；
        - 还是 IK 在当前约束下追不到这个目标。
        """

        task_debug_info = {}

        for body_name, task in body_to_task.items():
            frame_name = body_to_frame[body_name]
            target_info = latest_targets.get(body_name)
            current_pose = self.get_robot_body_pose(frame_name)
            task_error_vector = np.asarray(task.compute_error(self.configuration)).reshape(-1)

            task_entry = {
                "frame_name": frame_name,
                "task_error_vector": task_error_vector.tolist(),
                "task_error_norm": float(np.linalg.norm(task_error_vector)),
            }

            if task_error_vector.shape[0] >= 3:
                task_entry["task_position_error_norm"] = float(np.linalg.norm(task_error_vector[:3]))
            if task_error_vector.shape[0] >= 6:
                task_entry["task_orientation_error_norm"] = float(np.linalg.norm(task_error_vector[3:6]))

            if target_info is not None:
                task_entry["target_pos"] = target_info["target_pos"].tolist()
                task_entry["target_quat_wxyz"] = target_info["target_quat"].tolist()

            if current_pose is not None:
                current_pos, current_quat = current_pose
                task_entry["current_pos"] = current_pos.tolist()
                task_entry["current_quat_wxyz"] = current_quat.tolist()

                if target_info is not None:
                    target_pos = target_info["target_pos"]
                    target_quat = target_info["target_quat"]
                    task_entry["position_error_norm"] = float(np.linalg.norm(current_pos - target_pos))
                    task_entry["orientation_error_deg"] = self.compute_quaternion_angle_error_deg(
                        target_quat,
                        current_quat,
                    )

            task_debug_info[body_name] = task_entry

        return task_debug_info

    def write_debug_log(
        self,
        frame_index,
        initial_error1,
        final_error1,
        initial_error2,
        final_error2,
        num_iter1,
        num_iter2,
    ):
        """把当前帧的数值化调试信息写入 jsonl 文件。"""

        qpos = np.asarray(self.configuration.data.qpos).copy()
        debug_record = {
            "frame_index": int(frame_index),
            "src_human": self.src_human,
            "tgt_robot": self.tgt_robot,
            "ik_config_path": str(self.ik_config_path),
            "initial_error1": initial_error1,
            "final_error1": final_error1,
            "initial_error2": initial_error2,
            "final_error2": final_error2,
            "num_iter1": int(num_iter1),
            "num_iter2": int(num_iter2),
            "root_pos": qpos[:3].tolist(),
            "root_rot_wxyz": qpos[3:7].tolist(),
            "dof_pos": qpos[7:].tolist(),
            "dof_names_in_order": self.robot_dof_names_in_order,
            "task_table1": self.collect_task_debug_info(
                self.human_body_to_task1,
                self.task1_body_to_frame,
                self.latest_task_targets1,
            ),
            "task_table2": self.collect_task_debug_info(
                self.human_body_to_task2,
                self.task2_body_to_frame,
                self.latest_task_targets2,
            ),
        }

        with open(self.debug_log_path, "a", encoding="utf-8") as debug_file:
            debug_file.write(json.dumps(debug_record, ensure_ascii=False) + "\n")
