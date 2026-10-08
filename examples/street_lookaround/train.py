"""Train the Look Around street example with the street preset and save photos.

    python train.py --data data/street --out runs/street --iterations 5000

Saves the trained splat (PLY), the learned sky, renders of training views next to
the photos with rendered depth, novel views between and around the two capture
points, and a drive-through animation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
from PIL import Image, ImageDraw

import msplat


def u8(img: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))


def colorize_depth(depth: np.ndarray, alpha: np.ndarray, near: float = 2.0, far: float = 60.0) -> Image.Image:
    """Inverse depth on a fixed near/far range (warm = near), black where empty."""
    valid = alpha > 0.5
    x = np.zeros_like(depth)
    x[valid] = (1 / np.maximum(depth[valid], near) - 1 / far) / (1 / near - 1 / far)
    x = np.clip(x, 0, 1)
    anchors = np.array([[0.05, 0.03, 0.25], [0.10, 0.35, 0.75], [0.15, 0.75, 0.60],
                        [0.95, 0.85, 0.20], [0.85, 0.20, 0.10]])
    pos = x * (len(anchors) - 1)
    i0 = np.clip(np.floor(pos).astype(int), 0, len(anchors) - 2)
    f = (pos - i0)[..., None]
    rgb = anchors[i0] * (1 - f) + anchors[i0 + 1] * f
    rgb[~valid] = 0
    return u8(rgb)


def label(img: Image.Image, text: str) -> Image.Image:
    img = img.copy()
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 10 + 6 * len(text), 16], fill=(0, 0, 0))
    d.text((5, 2), text, fill=(255, 255, 255))
    return img


def grid(rows: list[list[Image.Image]], pad: int = 4) -> Image.Image:
    w, h = rows[0][0].size
    cols = max(len(r) for r in rows)
    sheet = Image.new("RGB", (cols * w + (cols - 1) * pad, len(rows) * h + (len(rows) - 1) * pad), (255, 255, 255))
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            sheet.paste(img, (c * (w + pad), r * (h + pad)))
    return sheet


def look_pose(position: np.ndarray, forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """OpenGL camera-to-world (x right, y up, z back) looking along forward."""
    f = forward / np.linalg.norm(forward)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.eye(4, dtype=np.float32)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, u, -f, position
    return m


def rotate(v: np.ndarray, axis: np.ndarray, deg: float) -> np.ndarray:
    a = math.radians(deg)
    axis = axis / np.linalg.norm(axis)
    return v * math.cos(a) + np.cross(axis, v) * math.sin(a) + axis * np.dot(axis, v) * (1 - math.cos(a))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iterations", type=int, default=5000)
    ap.add_argument("--render-only", action="store_true", help="load out/model.ckpt instead of training")
    ap.add_argument("--preview-every", type=int, default=1000, help="save preview renders every N steps (0: off)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(args.data, "transforms.json")) as f:
        frames = sorted(json.load(f)["frames"], key=lambda fr: fr["file_path"])  # msplat's camera order
    names = [os.path.splitext(os.path.basename(fr["file_path"]))[0] for fr in frames]

    ds = msplat.load_dataset(args.data)
    print("priors attached:", ds.prior_counts(), flush=True)
    cfg = msplat.street_config(iterations=args.iterations, num_downscales=1, resolution_schedule=750,
                               output=os.path.join(args.out, "street.ply"))
    t = msplat.GaussianTrainer(ds, cfg)
    ckpt = os.path.join(args.out, "model.ckpt")
    if args.render_only:
        t.load_checkpoint(ckpt)
    else:
        preview_names = ["pano_a_y-015_p+00", "pano_b_y+045_p+00"]
        preview_idx = [names.index(n) for n in preview_names]

        def progress(s):
            print(f"  step {s.iteration}/{args.iterations}  splats {s.splat_count}  {s.ms_per_step:.0f} ms/step",
                  flush=True)
            if args.preview_every > 0 and s.iteration % args.preview_every == 0 and s.iteration < args.iterations:
                rows = [[label(u8(ds.image(i)), f"photo: {names[i]}"),
                         label(u8(t.render(i)), f"render at step {s.iteration}")] for i in preview_idx]
                size = rows[0][0].size
                rows = [[im.resize(size) for im in row] for row in rows]
                grid(rows).save(os.path.join(args.out, f"preview_step{s.iteration:05d}.jpg"), quality=90)

        t0 = time.time()
        every = max(1, args.iterations // 20)
        if args.preview_every > 0:
            every = math.gcd(every, args.preview_every)  # the callback must land on preview steps
        t.train(progress, callback_every=every)
        t.save_checkpoint(ckpt)
        t.export_ply(os.path.join(args.out, "street.ply"))
        t.export_sky(os.path.join(args.out, "sky.png"))
        print(f"trained {args.iterations} steps in {time.time() - t0:.0f}s, {t.splat_count} gaussians", flush=True)

    # ── Training views: photo | render | depth ──────────────────────────────
    show = ["pano_a_y-015_p+00", "pano_a_y+045_p+00", "pano_b_y-075_p+00", "pano_b_y+015_p+00", "pano_a_y+020_p+35"]
    rows, psnrs = [], {}
    for i, name in enumerate(names):
        gt = ds.image(i)
        rgb = t.render(i)
        mse = float(np.mean((rgb - gt) ** 2))
        psnrs[name] = 10 * math.log10(1 / max(mse, 1e-12))
        if name in show:
            depth, alpha = t.render_depth(i)
            rows.append([label(u8(gt), f"photo: {name}"), label(u8(rgb), f"gaussian render  {psnrs[name]:.1f} dB"),
                         label(colorize_depth(depth, alpha), "rendered depth")])
            u8(rgb).save(os.path.join(args.out, f"render_{name}.png"))
    grid(rows).save(os.path.join(args.out, "training_views.jpg"), quality=92)
    with open(os.path.join(args.out, "training_psnr.json"), "w") as f:
        json.dump({"mean": float(np.mean(list(psnrs.values()))), "views": psnrs}, f, indent=1)
    print(f"training-view PSNR: mean {np.mean(list(psnrs.values())):.2f} dB", flush=True)

    # ── Novel views ──────────────────────────────────────────────────────────
    idx = {n: i for i, n in enumerate(names)}
    pa = ds.camera_pose(idx["pano_a_y-015_p+00"])[:3, 3]
    pb = ds.camera_pose(idx["pano_b_y-015_p+00"])[:3, 3]
    travel = (pb - pa) / np.linalg.norm(pb - pa)
    # up: mean camera y axis of the level views
    up = np.mean([ds.camera_pose(i)[:3, 1] for i, n in enumerate(names) if n.endswith("p+00")], axis=0)
    up /= np.linalg.norm(up)
    travel -= up * np.dot(travel, up)
    travel /= np.linalg.norm(travel)
    side = np.cross(travel, up)
    ref = idx["pano_a_y-015_p+00"]
    gap = np.linalg.norm(pb - pa)

    novel = []
    for label_text, pos, fwd in [
        ("novel: halfway between captures, forward", (pa + pb) / 2, travel),
        ("novel: 1.5 m left of capture A, forward", pa - 1.5 * side * (gap / 5.54), travel),
        ("novel: under the arch, looking up 30 deg", pb, rotate(travel, side, 30)),
        ("novel: halfway, looking right 60 deg", (pa + pb) / 2, rotate(travel, up, -60)),
    ]:
        img = t.render_from_pose(look_pose(pos, fwd, up), ref)
        novel.append(label(u8(img), label_text))
    grid([novel[:2], novel[2:]]).save(os.path.join(args.out, "novel_views.jpg"), quality=92)

    # ── Drive-through: from 2 m behind capture A to 2 m past capture B ───────
    frames_out = []
    n = 48
    for k in range(n):
        s = k / (n - 1)
        pos = pa + travel * (-0.35 * gap + s * 1.7 * gap)
        fwd = rotate(travel, up, 12 * math.sin(2 * math.pi * s))
        img = t.render_from_pose_rgba8(look_pose(pos, fwd, up), ref)
        frames_out.append(Image.fromarray(img[..., :3]))
    frames_out[0].save(os.path.join(args.out, "drive_through.gif"), save_all=True, append_images=frames_out[1:],
                       duration=80, loop=0)
    print("saved photos to", args.out, flush=True)


if __name__ == "__main__":
    main()
