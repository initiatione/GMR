from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def script_string_constants(script_name: str) -> set[str]:
    tree = ast.parse((REPO_ROOT / "scripts" / script_name).read_text(encoding="utf-8"))
    return {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}


def script_has_camera_mode_step_wiring(script_name: str) -> bool:
    tree = ast.parse((REPO_ROOT / "scripts" / script_name).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "step":
            continue
        for keyword in node.keywords:
            if keyword.arg != "follow_camera":
                continue
            value = keyword.value
            return (
                isinstance(value, ast.Compare)
                and isinstance(value.left, ast.Attribute)
                and isinstance(value.left.value, ast.Name)
                and value.left.value.id == "args"
                and value.left.attr == "camera_mode"
                and len(value.ops) == 1
                and isinstance(value.ops[0], ast.Eq)
                and len(value.comparators) == 1
                and isinstance(value.comparators[0], ast.Constant)
                and value.comparators[0].value == "follow"
            )
    return False


def script_has_viewer_ui_wiring(script_name: str, keyword_name: str, arg_name: str) -> bool:
    tree = ast.parse((REPO_ROOT / "scripts" / script_name).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "RobotMotionViewer":
            continue
        for keyword in node.keywords:
            if keyword.arg != keyword_name:
                continue
            value = keyword.value
            return (
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id == "args"
                and value.attr == arg_name
            )
    return False


def test_vis_robot_motion_cli_defines_camera_mode_choices() -> None:
    constants = script_string_constants("vis_robot_motion.py")

    assert "--camera_mode" in constants
    assert "follow" in constants
    assert "free" in constants
    assert script_has_camera_mode_step_wiring("vis_robot_motion.py")
    assert "--show_left_ui" in constants
    assert "--show_right_ui" in constants
    assert "--highlight_support_geoms" in constants
    assert script_has_viewer_ui_wiring("vis_robot_motion.py", "show_left_ui", "show_left_ui")
    assert script_has_viewer_ui_wiring("vis_robot_motion.py", "show_right_ui", "show_right_ui")
    assert script_has_viewer_ui_wiring("vis_robot_motion.py", "highlight_support_geoms", "highlight_support_geoms")


def test_vis_robot_motion_dataset_cli_defines_camera_mode_choices() -> None:
    constants = script_string_constants("vis_robot_motion_dataset.py")

    assert "--camera_mode" in constants
    assert "follow" in constants
    assert "free" in constants
    assert script_has_camera_mode_step_wiring("vis_robot_motion_dataset.py")
    assert "--show_left_ui" in constants
    assert "--show_right_ui" in constants
    assert "--highlight_support_geoms" in constants
    assert script_has_viewer_ui_wiring("vis_robot_motion_dataset.py", "show_left_ui", "show_left_ui")
    assert script_has_viewer_ui_wiring("vis_robot_motion_dataset.py", "show_right_ui", "show_right_ui")
    assert script_has_viewer_ui_wiring("vis_robot_motion_dataset.py", "highlight_support_geoms", "highlight_support_geoms")
