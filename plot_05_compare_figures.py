# compare_figures.py
from __future__ import annotations

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from utils_plot import set_paper_style
set_paper_style()


def load_image(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[-1])
        return bg
    return img.convert("RGB")

def make_placeholder(size: tuple[int, int], text: str) -> Image.Image:
    w, h = size
    img = Image.new("RGB", (w, h), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, w - 1, h - 1], outline=(180, 180, 180), width=2)
    draw.text((10, 10), text, fill=(80, 80, 80))
    return img

def side_by_side(img1: Image.Image, img2: Image.Image, label1: str, label2: str) -> Image.Image:
    # normalize height
    h = max(img1.height, img2.height)
    w1 = int(img1.width * (h / img1.height))
    w2 = int(img2.width * (h / img2.height))
    img1 = img1.resize((w1, h), Image.LANCZOS)
    img2 = img2.resize((w2, h), Image.LANCZOS)

    label_h = 28
    out = Image.new("RGB", (w1 + w2, h + label_h), (255, 255, 255))
    out.paste(img1, (0, label_h))
    out.paste(img2, (w1, label_h))

    # labels in a top bar (avoid covering the figure)
    draw = ImageDraw.Draw(out)
    text_color = (20, 20, 20)
    draw.rectangle([0, 0, w1, label_h], fill=(255, 255, 255))
    draw.rectangle([w1, 0, w1 + w2, label_h], fill=(255, 255, 255))
    draw.text((8, 6), label1, fill=text_color)
    draw.text((w1 + 8, 6), label2, fill=text_color)
    return out

def _experiment_root(output_root: str, exp: dict, subdir: str) -> Path:
    if exp.get("figure_dir"):
        return Path(output_root) / str(exp["figure_dir"]) / subdir
    return Path(output_root) / f"Figures_{exp['method_a']}_vs_{exp['method_b']}" / subdir


def compare_experiments(
    exp1: dict,
    exp2: dict,
    figure_dirs: list[str],
    image_names: dict[str, list[str]],
    output_root: str = "outputs",
    output_dir: str | None = None,
) -> None:
    """
    exp = {"method_a": "...", "method_b": "..."}
    figure_dirs: ["kg_stable_figures", "kg_edge_interaction_effects", ...]
    image_names: {dir_name: ["a.png", "b.png", ...]}
    """
    for d in figure_dirs:
        names = image_names.get(d, [])
        if not names:
            continue

        exp1_root = _experiment_root(output_root, exp1, d)
        exp2_root = _experiment_root(output_root, exp2, d)

        if output_dir is None:
            out_dir = Path(output_root) / f"Compare_{exp1['method_a']}_vs_{exp2['method_a']}" / d
        else:
            out_dir = Path(output_dir) / d
        out_dir.mkdir(parents=True, exist_ok=True)

        for name in names:
            p1 = exp1_root / name
            p2 = exp2_root / name
            img1 = load_image(p1) if p1.exists() else None
            img2 = load_image(p2) if p2.exists() else None

            if img1 is None and img2 is None:
                print(f"[SKIP] missing both: {p1} and {p2}")
                continue
            if img1 is None:
                img1 = make_placeholder(img2.size, f"Missing: {p1.name}")
            if img2 is None:
                img2 = make_placeholder(img1.size, f"Missing: {p2.name}")
            merged = side_by_side(
                img1,
                img2,
                label1=f"{exp1['method_a']} vs {exp1['method_b']}",
                label2=f"{exp2['method_a']} vs {exp2['method_b']}",
            )

            out_path = out_dir / name
            merged.save(out_path)
            print(f"[OK] {out_path}")

if __name__ == "__main__":
    from plot_all_from_config import main
    main(default_sections=["comparisons"])
