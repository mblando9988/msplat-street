"""Build an msplat dataset from Look Around panoramas.

For each panorama: pinhole views (lookaround.py), metric depth (Depth Anything V2,
metric outdoor), sky and vehicle masks (SegFormer, ADE20K), then one point cloud fused
from all views' depth for initialization. Writes a Nerfstudio-style transforms.json
whose frames carry the prior paths msplat reads (prior_depth_path, ...), and a sizing
log that states every size from the original faces to the training views.

    python prepare.py --out data/street --views-per-pano 10 --size 640x480
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
from PIL import Image

from lookaround import FACE_NAMES, Panorama, make_view, render_view

HERE = os.path.dirname(os.path.abspath(__file__))

# The two captures of this example: face index -> file. The back face (2) was not
# downloaded, so views stay within +-120 deg of the heading.
PANORAMAS = [
    ("pano_a", "meta/d1177723-meta.json", {0: "faces/a_front.webp", 1: "faces/a_right.webp", 3: "faces/a_left.webp"}),
    ("pano_b", "meta/1d0e2c66-meta.json", {0: "faces/b_front.webp", 1: "faces/b_right.webp", 3: "faces/b_left.webp"}),
]

# (yaw, pitch) in degrees relative to the panorama heading
VIEW_GRID = [(-75, 0), (-45, 0), (-15, 0), (15, 0), (45, 0), (75, 0), (-60, 35), (-20, 35), (20, 35), (60, 35)]

DYNAMIC_CLASSES = {"car", "truck", "bus", "van", "person", "bicycle", "motorbike", "minibike", "boat", "airplane"}


def log(msg: str) -> None:
    print(f"[prepare] {msg}", flush=True)


def depth_model(name: str):
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    proc = AutoImageProcessor.from_pretrained(name)
    model = AutoModelForDepthEstimation.from_pretrained(name).eval()

    def run(img: Image.Image) -> np.ndarray:
        with torch.no_grad():
            out = model(**proc(images=img, return_tensors="pt")).predicted_depth
            out = torch.nn.functional.interpolate(out[:, None], size=(img.height, img.width), mode="bicubic",
                                                  align_corners=False)
        return out[0, 0].clamp(min=0).numpy().astype(np.float32)

    return run


def segmentation_model(name: str):
    import torch
    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(name).eval()
    labels = {i: n.split(",")[0].strip().lower() for i, n in model.config.id2label.items()}
    sky_ids = [i for i, n in labels.items() if n == "sky"]
    dyn_ids = [i for i, n in labels.items() if n in DYNAMIC_CLASSES]
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)

    def run(img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        # SegformerImageProcessor: 512x512 bilinear, ImageNet normalization
        x = (np.asarray(img.resize((512, 512), Image.BILINEAR), np.float32) / 255.0 - mean) / std
        with torch.no_grad():
            logits = model(pixel_values=torch.from_numpy(x.transpose(2, 0, 1))[None]).logits
            logits = torch.nn.functional.interpolate(logits, size=(img.height, img.width), mode="bilinear",
                                                     align_corners=False)
        cls = logits[0].argmax(0).numpy()
        return np.isin(cls, sky_ids), np.isin(cls, dyn_ids)

    return run


def confidence(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Heuristic confidence: monocular depth is least reliable far away and at depth edges."""
    logd = np.log(np.maximum(depth, 1e-3))
    gy, gx = np.gradient(logd)
    edge = np.exp(-6.0 * np.hypot(gx, gy))
    far = np.clip(40.0 / np.maximum(depth, 1e-3), 0.15, 1.0)
    return np.where(valid, edge * far, 0.0).astype(np.float32)


def backproject(view, depth: np.ndarray, rgb: np.ndarray, valid: np.ndarray, stride: int):
    j, i = np.mgrid[0:view.height:stride, 0:view.width:stride]
    sel = valid[j, i]
    j, i = j[sel], i[sel]
    d = depth[j, i]
    x = (i + 0.5 - view.cx) / view.fx * d
    y = (j + 0.5 - view.cy) / view.fx * d
    pts_cam = np.stack([x, y, d], -1)
    pts = pts_cam @ view.cam_to_world[:3, :3].T + view.cam_to_world[:3, 3]
    return pts, rgb[j, i], d


def fuse(points: np.ndarray, colors: np.ndarray, dist: np.ndarray, rel_voxel: float = 0.02,
         min_voxel: float = 0.06) -> tuple[np.ndarray, np.ndarray]:
    """Voxel merge with voxels growing with distance from the camera (log-spaced bands)."""
    band = np.floor(np.log(np.maximum(dist, 1.0)) / math.log(1.5)).astype(np.int64)
    size = np.maximum(min_voxel, rel_voxel * np.power(1.5, band + 0.5))
    q = np.floor(points / size[:, None]).astype(np.int64)
    key = np.stack([band, q[:, 0], q[:, 1], q[:, 2]], -1)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    out_p = np.zeros((counts.size, 3))
    out_c = np.zeros((counts.size, 3))
    np.add.at(out_p, inv, points)
    np.add.at(out_c, inv, colors)
    return (out_p / counts[:, None]).astype(np.float32), (out_c / counts[:, None]).astype(np.float32)


def write_ply(path: str, pts: np.ndarray, cols: np.ndarray) -> None:
    c8 = np.clip(cols * 255 + 0.5, 0, 255).astype(np.uint8)
    rec = np.zeros(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                    ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rec["red"], rec["green"], rec["blue"] = c8[:, 0], c8[:, 1], c8[:, 2]
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        f.write(rec.tobytes())


def rebuild_points(args) -> None:
    """Initial point cloud from the saved views and priors (no model inference)."""
    from types import SimpleNamespace

    with open(os.path.join(args.out, "transforms.json")) as f:
        tf = json.load(f)
    all_pts, all_cols, all_dist = [], [], []
    for fr in tf["frames"]:
        m = np.array(fr["transform_matrix"], dtype=np.float64)
        m[:3, 1:3] *= -1  # OpenGL -> OpenCV axes
        view = SimpleNamespace(width=tf["w"], height=tf["h"], fx=tf["fl_x"], cx=tf["cx"], cy=tf["cy"], cam_to_world=m)
        rgb = np.asarray(Image.open(os.path.join(args.out, fr["file_path"])).convert("RGB"), np.float32) / 255.0
        depth = np.load(os.path.join(args.out, fr["prior_depth_path"]))
        pts, cols, dist = backproject(view, depth, rgb, (depth > 0) & (depth < args.max_depth), stride=2)
        all_pts.append(pts); all_cols.append(cols); all_dist.append(dist)
    pts, cols = fuse(np.concatenate(all_pts), np.concatenate(all_cols), np.concatenate(all_dist), args.voxel)
    write_ply(os.path.join(args.out, tf["ply_file_path"]), pts, cols)
    log(f"initial point cloud: {len(pts)} points from {sum(len(a) for a in all_pts)} depth samples")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(HERE, "data"), help="faces/ and meta/ live here")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="640x480", help="view size WxH")
    ap.add_argument("--hfov", type=float, default=90.0)
    ap.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf")
    ap.add_argument("--seg-model", default="nvidia/segformer-b2-finetuned-ade-512-512")
    ap.add_argument("--max-depth", type=float, default=60.0, help="no initial points beyond this (m)")
    ap.add_argument("--voxel", type=float, default=0.02, help="initial point spacing, fraction of the distance")
    ap.add_argument("--points-only", action="store_true",
                    help="rebuild points3D.ply from the views and priors already in --out")
    args = ap.parse_args()
    width, height = (int(v) for v in args.size.lower().split("x"))
    if args.points_only:
        rebuild_points(args)
        return

    for sub in ("images", "priors/depth", "priors/confidence", "priors/sky", "priors/mask"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    t0 = time.time()
    panos = [Panorama.load(name, os.path.join(args.data, meta),
                           {k: os.path.join(args.data, v) for k, v in faces.items()})
             for name, meta, faces in PANORAMAS]
    origin = np.mean([p.position for p in panos], axis=0)

    # Sizing log: every size from the metadata's originals to the training views
    sizing = {"view_size": [width, height], "view_hfov_deg": args.hfov, "faces": []}
    for p in panos:
        for idx, img in sorted(p.faces.items()):
            meta_w, meta_h = p.meta["face_sizes"][idx]
            cam = p.meta["camera_metadata"][idx]
            px_per_deg = img.shape[1] / math.degrees(cam["fov_s"])
            sizing["faces"].append({
                "panorama": p.name, "face": FACE_NAMES[idx], "file": os.path.relpath(p.files[idx], args.data),
                "metadata_size": [meta_w, meta_h], "file_size": [img.shape[1], img.shape[0]],
                "file_scale": round(img.shape[1] / meta_w, 4), "pixels_per_degree": round(px_per_deg, 2),
            })
            log(f"{p.name} {FACE_NAMES[idx]:6s} metadata {meta_w}x{meta_h} -> file {img.shape[1]}x{img.shape[0]} "
                f"(x{img.shape[1] / meta_w:.3f}), {px_per_deg:.1f} px/deg")
    fx = (width / 2) / math.tan(math.radians(args.hfov) / 2)
    log(f"views {width}x{height}, hfov {args.hfov} deg, fx {fx:.1f}, {fx * math.pi / 180:.1f} px/deg at the centre "
        f"(rendered at 2x and box-filtered)")
    log(f"capture points {np.linalg.norm(panos[0].position - panos[1].position):.2f} m apart")

    log("loading models")
    depth_fn = depth_model(args.depth_model)
    seg_fn = segmentation_model(args.seg_model)

    frames, all_pts, all_cols, all_dist = [], [], [], []
    for p in panos:
        for yaw, pitch in VIEW_GRID:
            name = f"{p.name}_y{yaw:+04d}_p{pitch:+03d}"
            view = make_view(p, origin, yaw, pitch, width, height, args.hfov, name)
            img, cov = render_view(p, view)
            rgb = np.asarray(img, dtype=np.float32) / 255.0
            depth = depth_fn(img)
            sky, dynamic = seg_fn(img)
            covered = cov > 0.99
            keep = covered & ~dynamic
            sky &= covered
            depth_valid = keep & ~sky & (depth > 0.5)
            conf = confidence(depth, depth_valid)

            img.save(os.path.join(args.out, "images", name + ".png"))
            np.save(os.path.join(args.out, "priors/depth", name + ".npy"), np.where(depth_valid, depth, 0).astype(np.float32))
            np.save(os.path.join(args.out, "priors/confidence", name + ".npy"), conf)
            Image.fromarray(sky.astype(np.uint8) * 255).save(os.path.join(args.out, "priors/sky", name + ".png"))
            Image.fromarray(keep.astype(np.uint8) * 255).save(os.path.join(args.out, "priors/mask", name + ".png"))

            pts, cols, dist = backproject(view, depth, rgb, depth_valid & (depth < args.max_depth), stride=2)
            all_pts.append(pts); all_cols.append(cols); all_dist.append(dist)
            frames.append({
                "file_path": f"images/{name}.png",
                "transform_matrix": view.gl_cam_to_world().tolist(),
                "prior_depth_path": f"priors/depth/{name}.npy",
                "prior_confidence_path": f"priors/confidence/{name}.npy",
                "prior_sky_mask_path": f"priors/sky/{name}.png",
                "mask_path": f"priors/mask/{name}.png",
            })
            log(f"{name}: sky {sky.mean():.0%}, masked {1 - keep.mean():.0%}, "
                f"median depth {np.median(depth[depth_valid]) if depth_valid.any() else 0:.1f} m "
                f"({time.time() - t0:.0f}s)")

    pts, cols = fuse(np.concatenate(all_pts), np.concatenate(all_cols), np.concatenate(all_dist), args.voxel)
    write_ply(os.path.join(args.out, "points3D.ply"), pts, cols)
    log(f"initial point cloud: {len(pts)} points from {sum(len(a) for a in all_pts)} depth samples")

    transforms = {"camera_model": "OPENCV", "w": width, "h": height, "fl_x": fx, "fl_y": fx,
                  "cx": width / 2, "cy": height / 2, "ply_file_path": "points3D.ply", "frames": frames}
    with open(os.path.join(args.out, "transforms.json"), "w") as f:
        json.dump(transforms, f, indent=1)
    with open(os.path.join(args.out, "sizing.json"), "w") as f:
        json.dump(sizing, f, indent=1)
    log(f"done in {time.time() - t0:.0f}s: {len(frames)} views -> {args.out}")


if __name__ == "__main__":
    main()
