from __future__ import annotations

import argparse
import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


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
REFERENCE_BODY_ALIASES = {
    "LINK_WAIST_YAW": "LINK_TORSO_YAW",
}
REFERENCE_MESH_ALIASES = {
    "LINK_WAIST_YAW": "LINK_TORSO_YAW",
}
FOOT_MATERIAL_NAME = "t800_color_urdf_foot"
FOOT_RGBA = "0.75294 0.75294 0.75294"
REFERENCE_VISUAL_RGBA = "0.792157 0.819608 0.933333"


@dataclass(frozen=True)
class ReferenceVisual:
    mesh: str
    material: str | None


def comment_preserving_tree(path: Path) -> ET.ElementTree:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.parse(path, parser=parser)


def parse_xml_with_includes(path: Path) -> ET.Element:
    root = ET.parse(path).getroot()
    expand_includes(root, path.parent)
    return root


def expand_includes(element: ET.Element, base_dir: Path) -> None:
    for child in list(element):
        if child.tag == "include" and child.get("file"):
            include_path = base_dir / child.attrib["file"]
            included_root = parse_xml_with_includes(include_path)
            child_index = list(element).index(child)
            element.remove(child)

            if included_root.tag == "mujoco":
                replacement_children = list(included_root)
            else:
                replacement_children = [included_root]
            for offset, replacement in enumerate(replacement_children):
                element.insert(child_index + offset, copy.deepcopy(replacement))
        else:
            expand_includes(child, base_dir)


def load_reference_visuals(reference_xml: Path) -> tuple[list[ET.Element], list[ET.Element], dict[str, ReferenceVisual]]:
    root = parse_xml_with_includes(reference_xml)
    asset = root.find("asset")
    if asset is None:
        raise ValueError(f"No <asset> block found in {reference_xml}")

    textures = []
    materials = []
    for child in list(asset):
        if child.tag == "texture" and "file" in child.attrib:
            texture = copy.deepcopy(child)
            texture.set("file", Path(texture.attrib["file"]).name)
            textures.append(texture)
        elif child.tag == "material" and child.get("texture"):
            materials.append(copy.deepcopy(child))

    visuals: dict[str, ReferenceVisual] = {}
    for body in root.iter("body"):
        body_name = body.get("name")
        link_name = REFERENCE_BODY_ALIASES.get(body_name or "", body_name)
        if link_name not in T800_LINKS:
            continue
        for geom in body.findall("geom"):
            if geom.get("class") != "visual":
                continue
            mesh = geom.get("mesh")
            if mesh:
                mesh = REFERENCE_MESH_ALIASES.get(mesh, mesh)
                visuals[link_name] = ReferenceVisual(mesh=mesh, material=geom.get("material"))
                break

    missing = sorted(set(T800_LINKS) - set(visuals))
    if missing:
        raise ValueError(f"Missing reference visual geoms for: {missing}")

    return textures, materials, visuals


def is_t800_visual_asset(element: ET.Element, reference_materials: set[str]) -> bool:
    name = element.get("name", "")
    if element.tag == "mesh":
        return name in T800_LINKS or any(name.startswith(f"{link}__mat") for link in T800_LINKS)
    if element.tag == "texture":
        return name in reference_materials
    if element.tag == "material":
        return name in reference_materials or name.startswith("t800_color_")
    return False


def install_reference_assets(
    asset: ET.Element,
    textures: list[ET.Element],
    materials: list[ET.Element],
    *,
    alpha: float,
) -> None:
    reference_materials = {material.attrib["name"] for material in materials}
    for child in list(asset):
        if is_t800_visual_asset(child, reference_materials):
            asset.remove(child)

    insert_index = 0
    for texture in textures:
        asset.insert(insert_index, texture)
        insert_index += 1
    for material in materials:
        material = copy.deepcopy(material)
        material.set("rgba", f"1 1 1 {alpha:.8g}")
        asset.insert(insert_index, material)
        insert_index += 1
    foot_material = ET.Element(
        "material",
        {"name": FOOT_MATERIAL_NAME, "rgba": f"{FOOT_RGBA} {alpha:.8g}"},
    )
    asset.insert(insert_index, foot_material)
    insert_index += 1

    for link_name in T800_LINKS:
        asset.insert(insert_index, ET.Element("mesh", {"name": link_name, "file": f"{link_name}.obj"}))
        insert_index += 1


def strip_visual_default_rgba(root: ET.Element) -> None:
    for visual_default in root.findall(".//default[@class='visual_mesh']/geom"):
        visual_default.attrib.pop("rgba", None)


def replace_visual_geoms(
    root: ET.Element,
    visuals: dict[str, ReferenceVisual],
    *,
    alpha: float,
) -> None:
    rgba = f"{REFERENCE_VISUAL_RGBA} {alpha:.8g}"
    for body in root.iter("body"):
        link_name = body.get("name")
        if link_name not in visuals:
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

        visual = visuals[link_name]
        attributes = {
            "class": "visual_mesh",
            "mesh": visual.mesh,
            "rgba": rgba,
        }
        if visual.material:
            attributes["material"] = visual.material
        body.insert(child_index, ET.Element("geom", attributes))


def update_foot_materials(root: ET.Element, *, alpha: float) -> None:
    for body in root.iter("body"):
        if body.get("name") not in {"LINK_FOOT_L", "LINK_FOOT_R"}:
            continue
        for geom in body.findall("geom"):
            if geom.get("group") != "2":
                continue
            geom.set("material", FOOT_MATERIAL_NAME)
            geom.set("rgba", f"{FOOT_RGBA} {alpha:.8g}")


def apply_reference_visuals(
    xml_path: Path,
    output_path: Path,
    reference_xml: Path,
    *,
    model_name: str,
    alpha: float,
) -> None:
    textures, materials, visuals = load_reference_visuals(reference_xml)
    tree = comment_preserving_tree(xml_path)
    root = tree.getroot()
    root.set("model", model_name)

    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", "../meshes")
        compiler.set("texturedir", "../texture")

    asset = root.find("asset")
    if asset is None:
        raise ValueError(f"No <asset> block found in {xml_path}")

    install_reference_assets(asset, textures, materials, alpha=alpha)
    strip_visual_default_rgba(root)
    replace_visual_geoms(root, visuals, alpha=alpha)
    update_foot_materials(root, alpha=alpha)

    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    human_robot_root = repo_root.parent
    reference_xml = (
        human_robot_root
        / "unitree_rl_lab"
        / "deploy"
        / "robots"
        / "t800_sim2sim"
        / "assets"
        / "resource"
        / "robot"
        / "t800"
        / "xml"
        / "serial_t800_sdk.xml"
    )

    parser = argparse.ArgumentParser(description="Apply reference OBJ/PNG T800 visual assets to GMR MJCF files.")
    parser.add_argument("--reference-xml", type=Path, default=reference_xml)
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
    args = parser.parse_args()

    apply_reference_visuals(
        args.source_xml,
        args.output_xml,
        args.reference_xml,
        model_name="t800_full_gmr",
        alpha=1.0,
    )
    apply_reference_visuals(
        args.transparent_source_xml,
        args.transparent_output_xml,
        args.reference_xml,
        model_name="t800_full_gmr_transparent",
        alpha=0.22,
    )
    print(f"Updated {args.output_xml}")
    print(f"Updated {args.transparent_output_xml}")


if __name__ == "__main__":
    main()
