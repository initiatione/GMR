import mujoco
import numpy as np
import xml.etree.ElementTree as ETree
import torch
import copy
from io import BytesIO
from lxml.etree import XMLParser, parse
from collections import OrderedDict


class MuJoCoFK:
    """
    通过 MuJoCo 解析机器人 XML，获取所有链接信息，用于前向运动学计算各 link 的姿态
    """
    def __init__(self, asset_file: str, device=torch.device("cpu")):
        self.mjcf_file = asset_file
        self.model = mujoco.MjModel.from_xml_path(self.mjcf_file)
        self.data  = mujoco.MjData(self.model)

        parser = XMLParser(remove_blank_text=True)
        tree = parse(
            BytesIO(open(self.mjcf_file, "rb").read()),
            parser=parser,
        )
        self.dof_axis = []
        joint_nodes = tree.getroot().find("worldbody").findall(".//joint")
        non_free_joint_nodes = [
            joint_node
            for joint_node in joint_nodes
            if joint_node.attrib.get("type") != "free"
        ]
        joints = {joint_node.attrib["name"] for joint_node in non_free_joint_nodes}

        actuator_root = tree.getroot().find("actuator")
        assert actuator_root is not None, "No actuator block found in the mjcf file"
        actuator_nodes = list(actuator_root.getchildren())
        assert len(actuator_nodes) > 0, "No motors found in the mjcf file"

        # 这里优先读取 actuator 里声明的 joint 名称。
        # 对 auto-ik 来说，真正需要的是“受驱动关节顺序”，而不是 actuator 自己的名字。
        self.motor_joint_names = []
        for actuator_node in actuator_nodes:
            joint_name = actuator_node.attrib.get("joint", actuator_node.attrib.get("name"))
            if joint_name is not None:
                self.motor_joint_names.append(joint_name)

        self.num_dof = len(self.motor_joint_names)
        self.num_extend_dof = self.num_dof

        self.mjcf_data = mjcf_data = self.from_mjcf(self.mjcf_file)
        self.body_names = copy.deepcopy(mjcf_data["node_names"])
        self._parents = mjcf_data["parent_indices"]
        self._proper_kinematic_structure = copy.deepcopy(mjcf_data["node_names"])
        self._offsets = mjcf_data["local_translation"][None,].to(device)
        self._local_rotation = mjcf_data["local_rotation"][None,].to(device)
        self.joint_to_body = {
            joint_name: body_name
            for body_name, joint_name in mjcf_data["body_to_joint"].items()
        }
        self.actuated_joints_idx = np.array(
            [
                self.body_names.index(self.joint_to_body[joint_name])
                for joint_name in self.motor_joint_names
                if joint_name in self.joint_to_body
            ]
        )

        for motor_joint_name in self.motor_joint_names:
            if motor_joint_name not in joints:
                print(motor_joint_name)

        self.has_freejoint = any(joint_node.attrib.get("type") == "free" for joint_node in joint_nodes)
        joint_axis_map = {
            joint_node.attrib["name"]: [float(value) for value in joint_node.attrib["axis"].split(" ")]
            for joint_node in non_free_joint_nodes
            if "axis" in joint_node.attrib
        }
        self.dof_axis = torch.tensor(
            [
                joint_axis_map[joint_name]
                for joint_name in self.motor_joint_names
                if joint_name in joint_axis_map
            ],
            dtype=torch.float32,
        )
        self.num_bodies = len(self.body_names)
        # 只保留可见刚体：排除 root(id=0)，并保证顺序固定
        self.body_ids = [i for i in range(self.num_bodies) if i != 0]

        self.joints_range = mjcf_data["joints_range"].to(device)
        # 添加关节顺序列表
        self.joint_order = self.get_joint_order()
    
    def from_mjcf(self, path):
        # function from Poselib:
        tree = ETree.parse(path)
        xml_doc_root = tree.getroot()
        xml_world_body = xml_doc_root.find("worldbody")
        if xml_world_body is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")
        # assume this is the root
        xml_body_root = xml_world_body.find("body")
        if xml_body_root is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")

        xml_joint_root = xml_body_root.find("joint")

        node_names = []
        parent_indices = []
        local_translation = []
        local_rotation = []
        joints_range = []
        body_to_joint = OrderedDict()

        # recursively adding all nodes into the skel_tree
        def _add_xml_node(xml_node, parent_index, node_index):
            node_name = xml_node.attrib.get("name")
            # parse the local translation into float list
            pos = np.fromstring(
                xml_node.attrib.get("pos", "0 0 0"), dtype=float, sep=" "
            )
            quat = np.fromstring(
                xml_node.attrib.get("quat", "1 0 0 0"), dtype=float, sep=" "
            )
            node_names.append(node_name)
            parent_indices.append(parent_index)
            local_translation.append(pos)
            local_rotation.append(quat)
            curr_index = node_index
            node_index += 1
            all_joints = [
                joint_node
                for joint_node in xml_node.findall("joint")
                if joint_node.attrib.get("type") != "free"
            ]

            for joint in all_joints:
                if not joint.attrib.get("range") is None:
                    joints_range.append(
                        np.fromstring(joint.attrib.get("range"), dtype=float, sep=" ")
                    )
                else:
                    if not joint.attrib.get("type") == "free":
                        joints_range.append([-np.pi, np.pi])
            for joint_node in xml_node.findall("joint"):
                if joint_node.attrib.get("type") == "free":
                    continue
                body_to_joint[node_name] = joint_node.attrib.get("name")

            for next_node in xml_node.findall("body"):
                node_index = _add_xml_node(next_node, curr_index, node_index)

            return node_index

        _add_xml_node(xml_body_root, -1, 0)
        assert len(joints_range) == self.num_dof
        return {
            "node_names": node_names,
            "parent_indices": torch.from_numpy(
                np.array(parent_indices, dtype=np.int32)
            ),
            "local_translation": torch.from_numpy(
                np.array(local_translation, dtype=np.float32)
            ),
            "local_rotation": torch.from_numpy(
                np.array(local_rotation, dtype=np.float32)
            ),
            "joints_range": torch.from_numpy(np.array(joints_range)),
            "body_to_joint": body_to_joint,
        }
    
    def get_joint_order(self):
        """获取求解器的关节顺序列表"""
        # 这里直接返回 actuator 中声明的 joint 顺序。
        # 对 auto-ik 来说，这个顺序就是后续 `robot_qpos_init` 应该映射到的目标顺序。
        return list(self.motor_joint_names)
    
    def get_specific_body_positions(self, qpos_full: np.ndarray, body_names: list):
        """
        获取指定身体部位的位置
        
        Args:
        qpos_full: 完整的机器人姿态
        body_names: 需要获取位置的身体名称列表
        
        Return:
        positions: 指定身体部位的3D位置数组
        rotations: 指定身体部位的旋转矩阵数组
        """
        self.data.qpos[:] = qpos_full
        mujoco.mj_forward(self.model, self.data)
        
        positions = []
        rotations = []
        
        for body_name in body_names:
            try:
                body_id = self.model.body(body_name).id
                pos = np.array(self.data.xpos[body_id], dtype=np.float32)
                rot = np.array(self.data.xmat[body_id], dtype=np.float64).reshape(3, 3)
                positions.append(pos)
                rotations.append(rot)
            except:
                print(f"[Warning] Body '{body_name}' not found in the model")
                # 添加默认位置
                positions.append(np.array([0, 0, 0], dtype=np.float32))
                rotations.append(np.eye(3, dtype=np.float64))
        
        return np.array(positions), np.array(rotations)
