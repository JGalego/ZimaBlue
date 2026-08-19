#!/usr/bin/env python3
"""Trace a pool out of a photograph, then clean it.

    python examples/pool_from_photo.py
    python examples/pool_from_photo.py --photo backyard.jpg --width 8.4 --sample 640,410
    python examples/pool_from_photo.py --minutes 5

With no ``--photo`` this synthesises one, so the example runs anywhere and the
answer can be checked: the fake is a render of the ``kidney`` preset, and the
script prints how close the trace came to the pool it was drawn from.

What a photograph does not tell you, and how each gap is handled:

* **Scale.** Nothing in an image separates a small pool nearby from a large one
  far away, so a real measurement is required. ``--width`` is the easy one.
* **Which blue is the pool.** Skies and parasols are blue too. ``--sample``
  points at a pixel inside the water and settles it.
* **Depth.** The surface says nothing about the floor. ``--depth`` is a guess
  you are making explicitly rather than one the library makes for you.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import zimablue as zb
from zimablue.imaging import trace_pool


def synthesise(path: Path) -> tuple[Path, tuple[int, int], float, zb.Pool]:
    """Draw a plausible backyard photo of a known pool."""
    from PIL import Image, ImageDraw, ImageFilter

    truth = zb.make_pool("kidney")
    width, height = 900, 620
    image = Image.new("RGB", (width, height), (120, 128, 110))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, width, int(height * 0.20)], fill=(126, 176, 214))  # sky
    draw.rectangle([0, int(height * 0.20), width, int(height * 0.40)], fill=(96, 122, 74))
    draw.rectangle([0, int(height * 0.40), width, height], fill=(196, 186, 168))  # decking

    ring = np.asarray(truth.boundary.exterior.coords)
    minx, miny, maxx, maxy = truth.boundary.bounds
    pad, top, bottom = 0.09 * width, 0.44 * height, 0.96 * height
    scale = min((width - 2 * pad) / (maxx - minx), (bottom - top) / (maxy - miny))
    xs = pad + (ring[:, 0] - minx) * scale
    ys = bottom - (ring[:, 1] - miny) * scale
    draw.polygon(list(zip(xs.tolist(), ys.tolist(), strict=True)), fill=(38, 122, 176))

    # Decoys: a parasol and a towel, both blue, both bigger than they look.
    draw.ellipse([27, 136, 198, 223], fill=(46, 110, 168))
    draw.rectangle([774, 148, 873, 210], fill=(60, 132, 190))

    # Sun on the water.
    array = np.asarray(image).astype(np.float64)
    gy, gx = np.mgrid[0:height, 0:width]
    array += (
        np.exp(-(((gx - width * 0.58) ** 2) / 4200 + ((gy - height * 0.66) ** 2) / 900)) * 150
    )[..., None]
    array += np.random.default_rng(0).normal(0, 7, array.shape)
    Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(0.6)
    ).save(path)

    inside = truth.boundary.representative_point()
    seed = (
        int(pad + (inside.x - minx) * scale),
        int(bottom - (inside.y - miny) * scale),
    )
    return path, seed, maxx - minx, truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photo", type=Path, help="A photo of a pool. Synthesised if omitted.")
    parser.add_argument("--width", type=float, help="The pool's real width in metres.")
    parser.add_argument("--sample", help="'x,y' pixel inside the water.")
    parser.add_argument("--depth", type=float, default=1.6)
    parser.add_argument("--minutes", type=float, default=20.0)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument(
        "--sam",
        metavar="ENCODER,DECODER",
        help="Two SAM .onnx files to segment with instead of the colour rules.",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    truth = None
    if args.photo is None:
        photo, seed, width, truth = synthesise(args.out / "synthetic_pool.jpg")
        print(f"no --photo given, so here is one: {photo}")
    else:
        photo, width = args.photo, args.width
        if width is None:
            raise SystemExit("--width is required with --photo: a photo carries no scale")
        seed = None
        if args.sample:
            sx, sy = (int(v) for v in args.sample.split(","))
            seed = (sx, sy)

    # ------------------------------------------------------------------
    segmenter = None
    if args.sam:
        from zimablue.segment import SamSegmenter

        encoder, decoder = args.sam.split(",")
        segmenter = SamSegmenter.load(encoder.strip(), decoder.strip())
        if seed is None:
            raise SystemExit("--sam needs --sample: SAM has to be prompted with a point")

    traced = trace_pool(photo, sample=seed, width=width, segmenter=segmenter)
    print()
    print(traced.summary())

    if segmenter is not None:
        print(f"\nSAM offered {len(segmenter.candidates)} masks, ranked by {segmenter.ranked_by}:")
        for candidate in segmenter.candidates:
            mark = "  <-- taken" if candidate.index == segmenter.chosen else ""
            print(f"  {candidate}{mark}")

    overlay = args.out / "trace_overlay.png"
    traced.overlay(overlay)
    print(f"\nlook at {overlay} before believing any of it -- segmenting a")
    print("photograph is a guess, and that picture is how you check the guess")

    if truth is not None:
        error = traced.area / truth.floor_area - 1
        print(f"\nthe pool it was drawn from is {truth.floor_area:.1f} m2")
        print(f"the trace came within {abs(error) * 100:.1f}%")

    # ------------------------------------------------------------------
    pool = traced.pool(args.depth, name="from_photo")
    print(f"\n{pool}")
    print("  depth is not measurable from a photo -- that came from --depth")

    result = zb.Simulation(pool=pool, dirt="autumn", seed=42).run(minutes=args.minutes)
    print()
    print(result.metrics.summary())

    path = result.save(args.out / "from_photo.zbr")
    print(f"\n  watch it:  zimablue replay {path}")


if __name__ == "__main__":
    main()
