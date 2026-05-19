from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


GMR_ROOT = Path(__file__).resolve().parents[1]
T800_XML = GMR_ROOT / "assets" / "t800" / "mujoco" / "t800_full_gmr.xml"
T800_TRANSPARENT_XML = GMR_ROOT / "assets" / "t800" / "mujoco" / "t800_full_gmr_transparent.xml"
T800_MESH_DIR = GMR_ROOT / "assets" / "t800" / "meshes"
T800_TEXTURE_DIR = GMR_ROOT / "assets" / "t800" / "texture"


def _geom_group(model: mujoco.MjModel, geom_id: int) -> int:
    return int(model.geom_group[geom_id])


def _update_bbox(
    bbox: tuple[list[float], list[float]] | None,
    vertex: tuple[float, float, float],
) -> tuple[list[float], list[float]]:
    if bbox is None:
        return ([vertex[0], vertex[1], vertex[2]], [vertex[0], vertex[1], vertex[2]])
    for axis in range(3):
        bbox[0][axis] = min(bbox[0][axis], vertex[axis])
        bbox[1][axis] = max(bbox[1][axis], vertex[axis])
    return bbox


def _binary_stl_bbox(path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    data = path.read_bytes()
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    assert len(data) == 84 + triangle_count * 50

    bbox: tuple[list[float], list[float]] | None = None
    offset = 84
    for _ in range(triangle_count):
        offset += 12
        for _vertex in range(3):
            vertex = struct.unpack_from("<fff", data, offset)
            bbox = _update_bbox(bbox, vertex)
            offset += 12
        offset += 2

    assert bbox is not None
    return tuple(bbox[0]), tuple(bbox[1])  # type: ignore[return-value]


def _binary_stl_signed_volume(path: Path) -> float:
    data = path.read_bytes()
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    assert len(data) == 84 + triangle_count * 50

    signed_volume = 0.0
    offset = 84
    for _ in range(triangle_count):
        offset += 12
        a = struct.unpack_from("<fff", data, offset)
        b = struct.unpack_from("<fff", data, offset + 12)
        c = struct.unpack_from("<fff", data, offset + 24)
        signed_volume += (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        ) / 6.0
        offset += 38

    return signed_volume


def _obj_bbox(path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    bbox: tuple[list[float], list[float]] | None = None
    with path.open(encoding="utf-8", errors="replace") as obj:
        for line in obj:
            if not line.startswith("v "):
                continue
            parts = line.split()
            vertex = (float(parts[1]), float(parts[2]), float(parts[3]))
            bbox = _update_bbox(bbox, vertex)

    assert bbox is not None
    return tuple(bbox[0]), tuple(bbox[1])  # type: ignore[return-value]


def _merge_bboxes(
    bboxes: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    bbox: tuple[list[float], list[float]] | None = None
    for minimum, maximum in bboxes:
        bbox = _update_bbox(bbox, minimum)
        bbox = _update_bbox(bbox, maximum)
    assert bbox is not None
    return tuple(bbox[0]), tuple(bbox[1])  # type: ignore[return-value]


def _max_bbox_abs_diff(
    left: tuple[tuple[float, float, float], tuple[float, float, float]],
    right: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> float:
    return max(abs(left[side][axis] - right[side][axis]) for side in range(2) for axis in range(3))


def _same_sign(left: float, right: float) -> bool:
    return (left > 0 and right > 0) or (left < 0 and right < 0)


def test_t800_visual_meshes_use_reference_obj_texture_materials() -> None:
    model = mujoco.MjModel.from_xml_path(str(T800_XML))
    tree = ET.parse(T800_XML)
    root = tree.getroot()

    visual_rgba = {
        tuple(round(float(channel), 3) for channel in model.geom_rgba[geom_id])
        for geom_id in range(model.ngeom)
        if _geom_group(model, geom_id) == 2
    }
    visual_geoms = [geom for geom in root.findall(".//geom") if geom.get("class") == "visual_mesh"]
    texture_files = [
        texture.attrib["file"]
        for texture in root.findall("./asset/texture")
        if "file" in texture.attrib
    ]

    assert len(visual_geoms) == 26
    assert visual_rgba == {
        (0.753, 0.753, 0.753, 1.0),
        (0.792, 0.82, 0.933, 1.0),
    }
    assert len(texture_files) == 15
    assert "LINK_BASE.png" in texture_files
    assert "LINK_TORSO_YAW.png" in texture_files
    assert "LINK_HEAD_YAW.png" in texture_files


def test_t800_visual_meshes_use_reference_obj_textures() -> None:
    model = mujoco.MjModel.from_xml_path(str(T800_XML))
    tree = ET.parse(T800_XML)
    root = tree.getroot()

    mesh_files = [mesh.attrib["file"] for mesh in root.findall("./asset/mesh")]
    texture_files = [
        texture.attrib["file"]
        for texture in root.findall("./asset/texture")
        if "file" in texture.attrib
    ]

    assert len([file for file in mesh_files if file.endswith(".obj")]) == 26
    assert len(texture_files) == 15
    assert all((T800_MESH_DIR / file).is_file() for file in mesh_files)
    assert all((T800_TEXTURE_DIR / Path(file).name).is_file() for file in texture_files)
    assert model.ntex >= 15


def test_t800_fallback_collision_proxies_are_invisible_and_quiet_at_initial_pose() -> None:
    tree = ET.parse(T800_XML)
    root = tree.getroot()

    fallback_defaults = root.findall(".//default[@class='collision_fallback']/geom")
    assert fallback_defaults
    for geom in fallback_defaults:
        assert geom.attrib["group"] == "4"
        assert geom.attrib["rgba"].split()[-1] == "0"

    model = mujoco.MjModel.from_xml_path(str(T800_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    fallback_contact_pairs = []
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        group1 = _geom_group(model, int(contact.geom1))
        group2 = _geom_group(model, int(contact.geom2))
        if group1 == 4 or group2 == 4:
            fallback_contact_pairs.append((group1, group2))

    assert fallback_contact_pairs == []


def test_t800_reference_visual_update_keeps_joint_and_collision_contract() -> None:
    tree = ET.parse(T800_XML)
    root = tree.getroot()

    joints = root.findall(".//joint")
    collision_geoms = [
        geom
        for geom in root.findall(".//geom")
        if (geom.get("class") or "").startswith("collision_")
    ]

    assert len(joints) == 26
    assert len(collision_geoms) == 32
    assert {geom.attrib["class"] for geom in collision_geoms} == {
        "collision_urdf",
        "collision_fallback",
    }
    assert all("mesh" not in geom.attrib for geom in collision_geoms)


def test_t800_transparent_model_keeps_reference_visuals_with_transparent_alpha() -> None:
    model = mujoco.MjModel.from_xml_path(str(T800_TRANSPARENT_XML))

    visual_rgba = {
        tuple(round(float(channel), 3) for channel in model.geom_rgba[geom_id])
        for geom_id in range(model.ngeom)
        if _geom_group(model, geom_id) == 2
    }

    assert len(visual_rgba) == 2
    assert {rgba[3] for rgba in visual_rgba} == {0.22}
    assert (0.792, 0.82, 0.933, 0.22) in visual_rgba


def test_t800_reference_obj_meshes_preserve_original_visual_bounding_boxes() -> None:
    tree = ET.parse(T800_XML)
    root = tree.getroot()

    for mesh in root.findall("./asset/mesh"):
        mesh_name = mesh.attrib["name"]
        mesh_file = mesh.attrib["file"]
        if not mesh_file.endswith(".obj"):
            continue

        original_bbox = _binary_stl_bbox(T800_MESH_DIR / f"{mesh_name}.stl")
        reference_bbox = _obj_bbox(T800_MESH_DIR / mesh_file)

        assert _max_bbox_abs_diff(original_bbox, reference_bbox) < 1e-5


def test_t800_reference_obj_meshes_have_material_bindings_for_textured_links() -> None:
    tree = ET.parse(T800_XML)
    root = tree.getroot()

    textured_materials = {
        material.attrib["name"]
        for material in root.findall("./asset/material")
        if "texture" in material.attrib and material.attrib["name"].startswith("LINK_")
    }
    visual_materials = {
        geom.attrib["material"]
        for geom in root.findall(".//geom")
        if geom.get("class") == "visual_mesh" and "material" in geom.attrib
    }

    assert len(textured_materials) == 15
    assert textured_materials.issubset(visual_materials)
