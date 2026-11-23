"""Compose a Zhihu article cover from an AI-generated textless background.

The image model renders Chinese poorly, so the pipeline is:
    AI background (no text)  +  Pillow overlay (system CJK fonts)

Usage:
    python make_cover.py --bg <path> --out <path> \
        --no 01 --total 06 \
        --kicker "从零训练一个大模型 · 硬核工程复盘" \
        --line1 "我想自己训一个大模型" \
        --line2-seg "前三周:accent" "都在吵它该干什么:white" \
        --sub "立项与定位的三次漂移 · 弯路复盘"

Each --lineN-seg is "text:color" where color is white|accent|dim|cyan.
Segments let one phrase be highlighted (e.g. the hook word) without
hand-placing coordinates. All coordinates are proportional to image size,
so the same layout works for any landscape background.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_MED = "/System/Library/Fonts/STHeiti Medium.ttc"
FONT_LIGHT = "/System/Library/Fonts/STHeiti Light.ttc"

# Palette, matched to the deep-navy / cyan / amber background.
WHITE = (242, 247, 255)
DIM = (150, 170, 200)
CYAN = (57, 208, 255)
AMBER = (255, 184, 96)
SCRIM = (7, 14, 36)

COLORS = {"white": WHITE, "dim": DIM, "cyan": CYAN, "accent": AMBER}


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size, index=0)


def _apply_left_scrim(img: Image.Image) -> Image.Image:
    """Darken the left ~65% so a title stays legible over the glow."""
    w, h = img.size
    xs = np.linspace(0.0, 1.0, w)
    # Full-strength scrim through 50% width, smooth falloff to 80%.
    plateau = 0.50
    end = 0.80
    alpha = np.where(
        xs <= plateau,
        0.90,
        np.clip((end - xs) / (end - plateau), 0.0, 1.0) ** 1.3 * 0.90,
    )
    alpha = (alpha * 255).astype(np.uint8)
    layer = np.zeros((h, w, 4), dtype=np.uint8)
    layer[..., 0], layer[..., 1], layer[..., 2] = SCRIM
    layer[..., 3] = alpha[None, :].repeat(h, axis=0)
    overlay = Image.fromarray(layer, "RGBA")
    base = img.convert("RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


def _draw_tracking(draw: ImageDraw.ImageDraw, xy, text, font, fill, spacing=4.0):
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += font.getlength(ch) + spacing
    return x


def _draw_segmented(draw, xy, segments, font, stroke=1):
    """Draw a line made of (text, color) segments, left to right."""
    x, y = xy
    for text, color in segments:
        draw.text(
            (x, y), text, font=font, fill=color,
            stroke_width=stroke, stroke_fill=(10, 20, 45),
        )
        x += font.getlength(text)
    return x


def compose(args: argparse.Namespace) -> Path:
    bg = Image.open(args.bg).convert("RGB")
    W, H = bg.size
    img = _apply_left_scrim(bg)
    d = ImageDraw.Draw(img)

    mx = int(W * 0.080)          # left margin
    # Font sizes scale with height so the layout is resolution-independent.
    f_kicker = _font(FONT_LIGHT, max(18, int(H * 0.028)))
    f_title = _font(FONT_MED, max(40, int(H * 0.070)))
    f_sub = _font(FONT_LIGHT, max(20, int(H * 0.032)))
    f_no = _font(FONT_MED, max(22, int(H * 0.034)))
    f_foot = _font(FONT_LIGHT, max(16, int(H * 0.023)))

    # 1) accent bar + kicker
    ky = int(H * 0.215)
    d.rounded_rectangle(
        [mx, ky + 6, mx + int(W * 0.030), ky + 11],
        radius=3, fill=CYAN,
    )
    kx = mx + int(W * 0.045)
    _draw_tracking(d, (kx, ky), args.kicker, f_kicker, CYAN, spacing=3.0)

    # 2) two-line main title (line2 supports highlighted segments)
    t1y = int(H * 0.335)
    t2y = int(H * 0.335) + int(H * 0.098)
    d.text(
        (mx, t1y), args.line1, font=f_title, fill=WHITE,
        stroke_width=1, stroke_fill=(10, 20, 45),
    )

    line2: list[tuple[str, tuple[int, int, int]]] = []
    for seg in args.line2_seg:
        text, _, key = seg.rpartition(":")
        line2.append((text, COLORS.get(key.strip().lower(), WHITE)))
    _draw_segmented(d, (mx, t2y), line2, f_title)

    # 3) divider + subtitle
    dy = t2y + int(H * 0.125)
    d.line([mx, dy, mx + int(W * 0.20), dy], fill=(*CYAN,), width=2)
    d.text((mx, dy + int(H * 0.022)), args.sub, font=f_sub, fill=DIM)

    # 4) footer: issue number + series tag
    fy = int(H * 0.865)
    d.text((mx, fy), f"{args.no} / {args.total}", font=f_no, fill=AMBER)
    no_w = f_no.getlength(f"{args.no} / {args.total}")
    d.line(
        [mx + no_w + int(W * 0.020), fy + 6, mx + no_w + int(W * 0.020), fy + int(H * 0.028)],
        fill=(70, 95, 135), width=2,
    )
    d.text(
        (mx + no_w + int(W * 0.032), fy + 2),
        args.footer, font=f_foot, fill=DIM,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "PNG", optimize=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no", default="01")
    ap.add_argument("--total", default="06")
    ap.add_argument("--kicker", default="从零训练一个大模型 · 硬核工程复盘")
    ap.add_argument("--line1", required=True)
    ap.add_argument("--line2-seg", nargs="+", required=True,
                    help='segments like "前三周:accent" "后半句:white"')
    ap.add_argument("--sub", default="")
    ap.add_argument("--footer", default="WANXIANG · 自研大模型工程复盘连载")
    args = ap.parse_args()
    out = compose(args)
    print(f"[cover] {out}  ({Image.open(out).size[0]}x{Image.open(out).size[1]})")


if __name__ == "__main__":
    main()
