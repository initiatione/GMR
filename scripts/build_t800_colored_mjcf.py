from __future__ import annotations

import argparse
import hashlib
import math
import re
import shutil
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


COLLADA_NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
T800_LINKS = [
    "LINK_BASE",
    "LINK_HIP_PITCH_L",
    "LINK_HIP_ROLL_L",
    "LINK_HIP_YAW_L",
    "LINK_KNEE_PITCH_L",
    "LINK_ANKLE_PITCH_L",
    "LINK_ANKLE_ROLL_L",
    "LINK_HIP_PITCH_R",
    "LINK_HIP_ROLL_R",
    "LINK_HIP_YAW_R",
    "LINK_KNEE_PITCH_R",
    "LINK_ANKLE_PITCH_R",
    "LINK_ANKLE_ROLL_R",
    "LINK_TORSO_YAW",
    "LINK_SHOULDER_PITCH_L",
    "LINK_SHOULDER_ROLL_L",
    "LINK_SHOULDER_YAW_L",
    "LINK_ELBOW_PITCH_L",
    "LINK_ELBOW_YAW_L",
    "LINK_SHOULDER_PITCH_R",
    "LINK_SHOULDER_ROLL_R",
    "LINK_SHOULDER_YAW_R",
    "LINK_ELBOW_PITCH_R",
    "LINK_ELBOW_YAW_R",
    "LINK_HEAD_PITCH",
    "LINK_HEAD_YAW",
]
FOOT_MATERIAL_NAME = "t800_color_urdf_foot"
FOOT_RGBA = (0.75294, 0.75294, 0.75294, 1.0)


@dataclass(frozen=True)
class ColoredMesh:
    link_name: str
    mesh_name: str
    mesh_file: str
    material_name: str
    rgba: tuple[float, float, float, float]


def parse_floats(text: str) -> list[float]:
    return [float(value) for value in text.split()]


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.split()]


def material_name_for_rgba(rgba: tuple[float, float, float, float]) -> str:
    rgb_hex = "".join(f"{max(0, min(255, round(channel * 255))):02x}" for channel in rgba[:3])
    alpha_hex = f"{max(0, min(255, round(rgba[3] * 255))):02x}"
    return f"t800_color_{rgb_hex}_{alpha_hex}"


def format_float(value: float) -> str:
    return f"{value:.8g}"


def format_rgba(rgba: tuple[float, float, float, float], *, alpha: float | None = None) -> str:
    values = list(rgba)
    if alpha is not None:
        values[3] = alpha
    return " ".join(format_float(value) for value in values)


def sanitize_xml_name(text: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")
    if sanitized:
        return sanitized
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"mat_{digest}"


def identity_matrix() -> list[list[float]]:
    return [[1.0 if row == col else 0.0 for col in range(4)] for row in range(4)]


def multiply_matrices(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[row][k] * b[k][col] for k in range(4)) for col in range(4)] for row in range(4)]


def parse_matrix(element: ET.Element) -> list[list[float]]:
    values = parse_floats(element.text or "")
    if len(values) != 16:
        return identity_matrix()
    return [values[index : index + 4] for index in range(0, 16, 4)]


def transform_vertex(matrix: list[list[float]], vertex: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = vertex
    transformed = [
        matrix[row][0] * x + matrix[row][1] * y + matrix[row][2] * z + matrix[row][3]
        for row in range(4)
    ]
    w = transformed[3] or 1.0
    return (transformed[0] / w, transformed[1] / w, transformed[2] / w)


def linear_determinant(matrix: list[list[float]]) -> float:
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def vector_subtract(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def cross_product(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def triangle_normal(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
) -> tuple[float, float, float]:
    normal = cross_product(vector_subtract(b, a), vector_subtract(c, a))
    length = math.sqrt(sum(component * component for component in normal))
    if length == 0:
        return (0.0, 0.0, 0.0)
    return tuple(component / length for component in normal)


def read_dae_positions(root: ET.Element) -> dict[str, list[tuple[float, float, float]]]:
    positions: dict[str, list[tuple[float, float, float]]] = {}
    for source in root.findall(".//c:library_geometries/c:geometry/c:mesh/c:source", COLLADA_NS):
        source_id = source.get("id")
        if source_id is None or "position" not in source_id.lower():
            continue
        array = source.find("c:float_array", COLLADA_NS)
        if array is None:
            continue
        values = parse_floats(array.text or "")
        positions[source_id] = [
            (values[index], values[index + 1], values[index + 2])
            for index in range(0, len(values), 3)
        ]
    return positions


def read_vertices(root: ET.Element) -> dict[str, str]:
    vertices: dict[str, str] = {}
    for vertices_node in root.findall(".//c:vertices", COLLADA_NS):
        vertices_id = vertices_node.get("id")
        if vertices_id is None:
            continue
        position_input = vertices_node.find("c:input[@semantic='POSITION']", COLLADA_NS)
        if position_input is None:
            continue
        source = position_input.get("source", "").lstrip("#")
        vertices[vertices_id] = source
    return vertices


def read_material_colors(root: ET.Element) -> dict[str, tuple[float, float, float, float]]:
    effect_colors: dict[str, tuple[float, float, float, float]] = {}
    for effect in root.findall(".//c:library_effects/c:effect", COLLADA_NS):
        effect_id = effect.get("id")
        diffuse = effect.find(".//c:diffuse/c:color", COLLADA_NS)
        if effect_id is None or diffuse is None:
            continue
        values = parse_floats(diffuse.text or "")
        if len(values) == 3:
            values.append(1.0)
        effect_colors[effect_id] = tuple(values[:4])  # type: ignore[assignment]

    material_colors: dict[str, tuple[float, float, float, float]] = {}
    for material in root.findall(".//c:library_materials/c:material", COLLADA_NS):
        material_id = material.get("id")
        if material_id is None:
            continue
        instance = material.find("c:instance_effect", COLLADA_NS)
        effect_id = instance.get("url", "").lstrip("#") if instance is not None else ""
        if effect_id in effect_colors:
            material_colors[material_id] = effect_colors[effect_id]
    return material_colors


def collect_instance_material_bindings(instance: ET.Element) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for material in instance.findall(".//c:instance_material", COLLADA_NS):
        symbol = material.get("symbol")
        target = material.get("target", "").lstrip("#")
        if symbol and target:
            bindings[symbol] = target
    return bindings


def collect_scene_instances(
    node: ET.Element,
    parent_matrix: list[list[float]],
    instances: dict[str, tuple[list[list[float]], dict[str, str]]],
) -> None:
    current_matrix = parent_matrix
    for child in list(node):
        tag = child.tag.split("}", 1)[-1]
        if tag == "matrix":
            current_matrix = multiply_matrices(current_matrix, parse_matrix(child))

    for instance in node.findall("c:instance_geometry", COLLADA_NS):
        geometry_id = instance.get("url", "").lstrip("#")
        if geometry_id:
            instances[geometry_id] = (current_matrix, collect_instance_material_bindings(instance))

    for child_node in node.findall("c:node", COLLADA_NS):
        collect_scene_instances(child_node, current_matrix, instances)


def read_scene_instances(root: ET.Element) -> dict[str, tuple[list[list[float]], dict[str, str]]]:
    instances: dict[str, tuple[list[list[float]], dict[str, str]]] = {}
    scene = root.find(".//c:library_visual_scenes/c:visual_scene", COLLADA_NS)
    if scene is None:
        return instances
    for node in scene.findall("c:node", COLLADA_NS):
        collect_scene_instances(node, identity_matrix(), instances)
    return instances


def triangles_for_materials(
    dae_path: Path,
) -> dict[str, tuple[tuple[float, float, float, float], list[tuple[tuple[float, float, float], ...]]]]:
    root = ET.parse(dae_path).getroot()
    positions = read_dae_positions(root)
    vertices = read_vertices(root)
    material_colors = read_material_colors(root)
    scene_instances = read_scene_instances(root)

    grouped: dict[str, tuple[tuple[float, float, float, float], list[tuple[tuple[float, float, float], ...]]]] = {}
    for geometry in root.findall(".//c:library_geometries/c:geometry", COLLADA_NS):
        geometry_id = geometry.get("id")
        if geometry_id is None:
            continue
        matrix, material_bindings = scene_instances.get(geometry_id, (identity_matrix(), {}))
        mesh = geometry.find("c:mesh", COLLADA_NS)
        if mesh is None:
            continue

        reverses_winding = linear_determinant(matrix) < 0
        for triangles_node in mesh.findall("c:triangles", COLLADA_NS):
            material_symbol = triangles_node.get("material", "unassigned")
            material_id = material_bindings.get(material_symbol, material_symbol)
            rgba = material_colors.get(material_id, (0.82, 0.82, 0.86, 1.0))

            inputs = triangles_node.findall("c:input", COLLADA_NS)
            stride = max(int(input_node.get("offset", "0")) for input_node in inputs) + 1
            vertex_offset = None
            position_source = None
            for input_node in inputs:
                if input_node.get("semantic") != "VERTEX":
                    continue
                vertex_offset = int(input_node.get("offset", "0"))
                vertex_source = input_node.get("source", "").lstrip("#")
                position_source = vertices[vertex_source]
                break
            if vertex_offset is None or position_source is None:
                raise ValueError(f"No VERTEX input found in {dae_path}")

            position_values = positions[position_source]
            indices = parse_ints(triangles_node.findtext("c:p", default="", namespaces=COLLADA_NS))
            material_triangles = grouped.setdefault(material_id, (rgba, []))[1]
            for index in range(0, len(indices), stride * 3):
                triangle_vertices = []
                for vertex_number in range(3):
                    vertex_index = indices[index + vertex_number * stride + vertex_offset]
                    triangle_vertices.append(transform_vertex(matrix, position_values[vertex_index]))
                if reverses_winding:
                    triangle_vertices[1], triangle_vertices[2] = triangle_vertices[2], triangle_vertices[1]
                material_triangles.append(tuple(triangle_vertices))  # type: ignore[arg-type]
    return grouped


def write_binary_stl(path: Path, triangles: list[tuple[tuple[float, float, float], ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = b"Generated from T800 Collada material groups"
    header = header[:80].ljust(80, b"\0")
    with path.open("wb") as stl:
        stl.write(header)
        stl.write(struct.pack("<I", len(triangles)))
        for triangle in triangles:
            normal = triangle_normal(triangle[0], triangle[1], triangle[2])
            stl.write(struct.pack("<fff", *normal))
            for vertex in triangle:
                stl.write(struct.pack("<fff", *vertex))
            stl.write(struct.pack("<H", 0))


def export_colored_meshes(source_dae_dir: Path, output_mesh_dir: Path) -> list[ColoredMesh]:
    if output_mesh_dir.exists():
        shutil.rmtree(output_mesh_dir)
    output_mesh_dir.mkdir(parents=True, exist_ok=True)

    colored_meshes: list[ColoredMesh] = []
    for link_name in T800_LINKS:
        dae_path = source_dae_dir / f"{link_name}.dae"
        material_groups = triangles_for_materials(dae_path)
        for material_index, (material_id, (rgba, triangles)) in enumerate(material_groups.items()):
            suffix = sanitize_xml_name(material_id)
            mesh_name = f"{link_name}__mat{material_index:02d}_{suffix}"
            mesh_file = f"colored/{mesh_name}.stl"
            write_binary_stl(output_mesh_dir / f"{mesh_name}.stl", triangles)
            colored_meshes.append(
                ColoredMesh(
                    link_name=link_name,
                    mesh_name=mesh_name,
                    mesh_file=mesh_file,
                    material_name=material_name_for_rgba(rgba),
                    rgba=rgba,
                )
            )
    return colored_meshes


def comment_preserving_tree(path: Path) -> ET.ElementTree:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.parse(path, parser=parser)


def insert_asset_materials_and_meshes(
    asset: ET.Element,
    colored_meshes: list[ColoredMesh],
    *,
    alpha: float,
) -> None:
    old_mesh_names = set(T800_LINKS)
    for child in list(asset):
        if child.tag == "mesh" and child.get("name") in old_mesh_names:
            asset.remove(child)
        elif child.tag == "mesh" and child.get("name", "").startswith(tuple(f"{link}__mat" for link in T800_LINKS)):
            asset.remove(child)
        elif child.tag == "material" and child.get("name", "").startswith("t800_color_"):
            asset.remove(child)

    insert_index = 0
    for colored_mesh in colored_meshes:
        mesh_element = ET.Element("mesh", {"name": colored_mesh.mesh_name, "file": colored_mesh.mesh_file})
        asset.insert(insert_index, mesh_element)
        insert_index += 1

    materials_by_name: dict[str, tuple[float, float, float, float]] = {}
    for colored_mesh in colored_meshes:
        materials_by_name.setdefault(colored_mesh.material_name, colored_mesh.rgba)
    materials_by_name[FOOT_MATERIAL_NAME] = FOOT_RGBA

    for material_name, rgba in sorted(materials_by_name.items()):
        material_element = ET.Element(
            "material",
            {
                "name": material_name,
                "rgba": format_rgba(rgba, alpha=alpha),
            },
        )
        asset.insert(insert_index, material_element)
        insert_index += 1


def strip_visual_default_rgba(root: ET.Element) -> None:
    for visual_default in root.findall(".//default[@class='visual_mesh']/geom"):
        visual_default.attrib.pop("rgba", None)


def replace_visual_mesh_geoms(root: ET.Element, colored_meshes: list[ColoredMesh], *, alpha: float) -> None:
    meshes_by_link: dict[str, list[ColoredMesh]] = {}
    for colored_mesh in colored_meshes:
        meshes_by_link.setdefault(colored_mesh.link_name, []).append(colored_mesh)

    for body in root.iter("body"):
        link_name = body.get("name")
        if link_name not in meshes_by_link:
            continue

        visual_geoms = [
            child
            for child in list(body)
            if child.tag == "geom"
            and child.get("class") == "visual_mesh"
            and (child.get("mesh") == link_name or (child.get("mesh") or "").startswith(f"{link_name}__mat"))
        ]
        if not visual_geoms:
            continue

        child_index = list(body).index(visual_geoms[0])
        for child in visual_geoms:
            body.remove(child)
        for offset, colored_mesh in enumerate(meshes_by_link[link_name]):
            body.insert(
                child_index + offset,
                ET.Element(
                    "geom",
                    {
                        "class": "visual_mesh",
                        "mesh": colored_mesh.mesh_name,
                        "material": colored_mesh.material_name,
                        "rgba": format_rgba(colored_mesh.rgba, alpha=alpha),
                    },
                ),
            )


def update_foot_materials(root: ET.Element, *, alpha: float) -> None:
    for body in root.iter("body"):
        if body.get("name") not in {"LINK_FOOT_L", "LINK_FOOT_R"}:
            continue
        for geom in body.findall("geom"):
            if geom.get("group") != "2":
                continue
            geom.set("material", FOOT_MATERIAL_NAME)
            geom.set("rgba", format_rgba(FOOT_RGBA, alpha=alpha))


def build_colored_xml(
    source_xml: Path,
    output_xml: Path,
    colored_meshes: list[ColoredMesh],
    *,
    model_name: str,
    alpha: float,
) -> None:
    tree = comment_preserving_tree(source_xml)
    root = tree.getroot()
    root.set("model", model_name)

    asset = root.find("asset")
    if asset is None:
        raise ValueError(f"No <asset> block found in {source_xml}")

    insert_asset_materials_and_meshes(asset, colored_meshes, alpha=alpha)
    strip_visual_default_rgba(root)
    replace_visual_mesh_geoms(root, colored_meshes, alpha=alpha)
    update_foot_materials(root, alpha=alpha)

    ET.indent(tree, space="  ")
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    human_robot_root = repo_root.parent
    parser = argparse.ArgumentParser(description="Build T800 MuJoCo visual meshes split by source DAE material.")
    parser.add_argument(
        "--source-dae-dir",
        type=Path,
        default=human_robot_root
        / "whole_body_tracking_engineai"
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking"
        / "assets"
        / "t800"
        / "meshes",
    )
    parser.add_argument(
        "--source-xml",
        type=Path,
        default=repo_root / "assets" / "t800" / "mujoco" / "t800_full_gmr.xml",
    )
    parser.add_argument(
        "--transparent-source-xml",
        type=Path,
        default=repo_root / "assets" / "t800" / "mujoco" / "t800_full_gmr_transparent.xml",
    )
    parser.add_argument(
        "--output-xml",
        type=Path,
        default=repo_root / "assets" / "t800" / "mujoco" / "t800_full_gmr.xml",
    )
    parser.add_argument(
        "--transparent-output-xml",
        type=Path,
        default=repo_root / "assets" / "t800" / "mujoco" / "t800_full_gmr_transparent.xml",
    )
    parser.add_argument(
        "--output-mesh-dir",
        type=Path,
        default=repo_root / "assets" / "t800" / "meshes" / "colored",
    )
    args = parser.parse_args()

    colored_meshes = export_colored_meshes(args.source_dae_dir, args.output_mesh_dir)
    build_colored_xml(
        args.source_xml,
        args.output_xml,
        colored_meshes,
        model_name="t800_full_gmr",
        alpha=1.0,
    )
    build_colored_xml(
        args.transparent_source_xml,
        args.transparent_output_xml,
        colored_meshes,
        model_name="t800_full_gmr_transparent",
        alpha=0.22,
    )
    print(f"Exported {len(colored_meshes)} colored mesh part(s) to {args.output_mesh_dir}")
    print(f"Updated {args.output_xml}")
    print(f"Updated {args.transparent_output_xml}")


if __name__ == "__main__":
    main()
