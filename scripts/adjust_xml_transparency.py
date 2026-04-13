import argparse
import pathlib
import xml.etree.ElementTree as ET


def strip_namespace(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def parse_rgba(rgba_text: str):
    parts = [float(x) for x in rgba_text.split()]
    if len(parts) == 3:
        parts.append(1.0)
    if len(parts) != 4:
        raise ValueError(f"Invalid rgba value: {rgba_text!r}")
    return parts


def format_rgba(rgba_values):
    return " ".join(f"{value:.6g}" for value in rgba_values)


def matches_filters(element, args) -> bool:
    if args.tag_names and strip_namespace(element.tag) not in args.tag_names:
        return False

    class_name = element.attrib.get("class", "")
    name = element.attrib.get("name", "")

    if args.class_contains and args.class_contains not in class_name:
        return False
    if args.name_contains and args.name_contains not in name:
        return False

    return True


def build_default_rgba(args):
    if args.default_rgb is None:
        return None
    rgb = [float(x) for x in args.default_rgb.split()]
    if len(rgb) != 3:
        raise ValueError("--default-rgb must contain exactly 3 numbers")
    return [rgb[0], rgb[1], rgb[2], args.alpha]


def adjust_alpha(input_path: pathlib.Path, output_path: pathlib.Path, args):
    tree = ET.parse(input_path)
    root = tree.getroot()

    default_rgba = build_default_rgba(args)
    changed = 0

    for element in root.iter():
        if not matches_filters(element, args):
            continue

        rgba_text = element.attrib.get("rgba")
        if rgba_text is None:
            if not args.inject_missing_rgba or default_rgba is None:
                continue
            element.set("rgba", format_rgba(default_rgba))
            changed += 1
            continue

        rgba = parse_rgba(rgba_text)
        if rgba[3] == 0.0 and not args.include_zero_alpha:
            continue

        rgba[3] = args.alpha
        element.set("rgba", format_rgba(rgba))
        changed += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(str(output_path), encoding="utf-8", xml_declaration=True)
    return changed


def build_output_path(input_path: pathlib.Path, alpha: float):
    alpha_suffix = str(alpha).replace(".", "_")
    return input_path.with_name(f"{input_path.stem}_alpha_{alpha_suffix}{input_path.suffix}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Adjust transparency in an XML model by rewriting rgba alpha values. "
            "Designed for MuJoCo/MJCF XML, but works for generic XML tags that use rgba."
        )
    )
    parser.add_argument("--input", required=True, help="Path to the source XML file.")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save the modified XML. Defaults to a sibling file with an alpha suffix.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file instead of writing a sibling output file.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="Target alpha value in [0, 1]. Example: 0.2",
    )
    parser.add_argument(
        "--tag-names",
        nargs="*",
        default=["geom", "material"],
        help="Tag names to process. Defaults to geom and material.",
    )
    parser.add_argument(
        "--class-contains",
        default=None,
        help='Only modify elements whose class contains this substring, e.g. "visual".',
    )
    parser.add_argument(
        "--name-contains",
        default=None,
        help="Only modify elements whose name contains this substring.",
    )
    parser.add_argument(
        "--include-zero-alpha",
        action="store_true",
        help="Also modify elements that are already invisible with alpha 0.",
    )
    parser.add_argument(
        "--inject-missing-rgba",
        action="store_true",
        help="Add rgba to matching elements that do not already have it.",
    )
    parser.add_argument(
        "--default-rgb",
        default="1 1 1",
        help="RGB to use when injecting rgba onto elements that do not already define it.",
    )

    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")

    input_path = pathlib.Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input XML not found: {input_path}")

    if args.in_place and args.output is not None:
        raise ValueError("--output and --in-place cannot be used together")

    if args.in_place:
        output_path = input_path
    elif args.output is not None:
        output_path = pathlib.Path(args.output).resolve()
    else:
        output_path = build_output_path(input_path, args.alpha).resolve()

    changed = adjust_alpha(input_path, output_path, args)
    print(f"Updated {changed} element(s).")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
