from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


GMR_ROOT = Path(__file__).resolve().parents[1]
T800_XML = GMR_ROOT / "assets" / "t800" / "mujoco" / "t800_full_gmr.xml"
T800_TRANSPARENT_XML = GMR_ROOT / "assets" / "t800" / "mujoco" / "t800_full_gmr_transparent.xml"
T800_MESH_DIR = GMR_ROOT / "assets" / "t800" / "meshes"


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


def test_t800_visual_meshes_keep_source_material_colors() -> None:
    model = mujoco.MjModel.from_xml_path(str(T800_XML))

    visual_rgba = {
        tuple(round(float(channel), 3) for channel in model.geom_rgba[geom_id])
        for geom_id in range(model.ngeom)
        if _geom_group(model, geom_id) == 2
    }

    assert len(visual_rgba) >= 5
    assert (1.0, 1.0, 1.0, 1.0) in visual_rgba
    assert (0.0, 0.0, 0.0, 1.0) in visual_rgba
    assert (1.0, 0.286, 0.007, 1.0) in visual_rgba


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


def test_t800_transparent_model_keeps_source_colors_with_transparent_alpha() -> None:
    model = mujoco.MjModel.from_xml_path(str(T800_TRANSPARENT_XML))

    visual_rgba = {
        tuple(round(float(channel), 3) for channel in model.geom_rgba[geom_id])
        for geom_id in range(model.ngeom)
        if _geom_group(model, geom_id) == 2
    }

    assert len(visual_rgba) >= 5
    assert {rgba[3] for rgba in visual_rgba} == {0.22}
    assert (1.0, 1.0, 1.0, 0.22) in visual_rgba
    assert (0.0, 0.0, 0.0, 0.22) in visual_rgba


def test_t800_colored_mesh_parts_preserve_original_visual_bounding_boxes() -> None:
    tree = ET.parse(T800_XML)
    root = tree.getroot()

    colored_mesh_files_by_link: dict[str, list[Path]] = {}
    for mesh in root.findall("./asset/mesh"):
        mesh_name = mesh.attrib["name"]
        if "__mat" not in mesh_name:
            continue
        link_name = mesh_name.split("__mat", 1)[0]
        colored_mesh_files_by_link.setdefault(link_name, []).append(T800_MESH_DIR / mesh.attrib["file"])

    assert len(colored_mesh_files_by_link) == 26
    for link_name, colored_mesh_files in colored_mesh_files_by_link.items():
        original_bbox = _binary_stl_bbox(T800_MESH_DIR / f"{link_name}.stl")
        colored_bbox = _merge_bboxes([_binary_stl_bbox(path) for path in colored_mesh_files])

        assert _max_bbox_abs_diff(original_bbox, colored_bbox) < 1e-6


def test_t800_colored_mesh_parts_preserve_original_visual_winding() -> None:
    tree = ET.parse(T800_XML)
    root = tree.getroot()

    colored_mesh_files_by_link: dict[str, list[Path]] = {}
    for mesh in root.findall("./asset/mesh"):
        mesh_name = mesh.attrib["name"]
        if "__mat" not in mesh_name:
            continue
        link_name = mesh_name.split("__mat", 1)[0]
        colored_mesh_files_by_link.setdefault(link_name, []).append(T800_MESH_DIR / mesh.attrib["file"])

    assert len(colored_mesh_files_by_link) == 26
    for link_name, colored_mesh_files in colored_mesh_files_by_link.items():
        original_volume = _binary_stl_signed_volume(T800_MESH_DIR / f"{link_name}.stl")
        colored_volume = sum(_binary_stl_signed_volume(path) for path in colored_mesh_files)

        assert _same_sign(original_volume, colored_volume), link_name
