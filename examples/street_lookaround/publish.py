"""Export the trained street scene for the antimatter15 web splat viewer.

    python publish.py --data data/street_aligned --run runs/street --out street.splat \\
        --url https://raw.githubusercontent.com/<owner>/<repo>/main/docs/street/street.splat

Writes a .splat and prints a viewer link. A plain export of this scene opens badly:

- The viewer starts at its own default camera, which for an arbitrary scene frame lands
  inside the street. The scene is written in metres and placed so that default camera is
  the first capture point, level, looking down the road, so even a link without a camera
  opens there. The printed link also carries that view in its #hash, which stops the
  viewer's idle camera swing.
- The learned sky is a texture behind the gaussians, not gaussians, so viewers show black
  above the horizon. It is added as a dome of large, distant splats coloured from the
  texture (same frame and mapping as the trainer's sky).
- The trainer draws each splat with a small screen-space blur that viewers don't apply,
  so splats thinner than a pixel show up as needles. The blur is baked into their size.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import urllib.parse

import numpy as np
from PIL import Image

import msplat

C0 = 0.28209479177387814  # degree-0 spherical harmonic
SPLAT = np.dtype([("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
# antimatter15 main.js defaultViewMatrix: world-to-camera, OpenCV axes, column-major
VIEWER_DEFAULT_VIEW = [0.47, 0.04, 0.88, 0, -0.11, 0.99, 0.02, 0, -0.88, -0.11, 0.47, 0, 0.07, 0.03, 6.55, 1]
CAPTURE_SPACING_M = 5.535  # distance between the two Look Around captures


def read_ply(path: str) -> np.ndarray:
    """Gaussians from msplat's export_ply (binary little-endian, float properties)."""
    with open(path, "rb") as f:
        names, count = [], 0
        while (line := f.readline().decode().strip()) != "end_header":
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            elif line.startswith("property float"):
                names.append(line.split()[-1])
        return np.fromfile(f, dtype=[(n, "<f4") for n in names], count=count)


def quat_from_matrix(m: np.ndarray) -> np.ndarray:
    """(w, x, y, z) of a rotation matrix."""
    t = np.trace(m)
    if t > 0:
        s = 2 * math.sqrt(1 + t)
        return np.array([s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2 * math.sqrt(1 + m[i, i] - m[j, j] - m[k, k])
    q = np.zeros(4)
    q[0], q[1 + i], q[1 + j], q[1 + k] = (m[k, j] - m[j, k]) / s, s / 4, (m[j, i] + m[i, j]) / s, (m[k, i] + m[i, k]) / s
    return q


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a * b of (w, x, y, z) quaternions along the last axis."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], -1)


def splat_records(pos, scale, rgb, alpha, quat) -> np.ndarray:
    rec = np.zeros(len(pos), SPLAT)
    rec["pos"] = pos
    rec["scale"] = scale
    rec["rgba"][:, :3] = np.clip(rgb * 255 + 0.5, 0, 255)
    rec["rgba"][:, 3] = np.clip(alpha * 255 + 0.5, 0, 255)
    # unit quaternion in bytes: viewers do not renormalize it
    q = quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-12)
    rec["rot"] = np.clip(q * 128 + 128.5, 0, 255)
    return rec


def sky_frame(poses: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Model.cpp computeSkyFrame: up = mean camera up, e1 = mean heading, e2 = up x e1."""
    up = np.mean([p[:3, 1] for p in poses], axis=0)
    fwd = np.mean([-p[:3, 2] for p in poses], axis=0)
    up /= np.linalg.norm(up)
    fwd -= np.dot(fwd, up) * up
    fwd /= np.linalg.norm(fwd)
    return up, fwd, np.cross(up, fwd)


def fill_unseen(tex: np.ndarray) -> np.ndarray:
    """Sky texels the trainer drove to black (only ever seen through gaussians) take the
    mean colour of their neighbours, growing in from the seen ones."""
    tex = tex.copy()
    bad = tex.max(-1) < 3 / 255
    h = tex.shape[0]
    while bad.any():
        good = np.pad(~bad, ((1, 1), (0, 0)))  # no neighbours past the poles; wraps in azimuth
        col = np.pad(tex * ~bad[..., None], ((1, 1), (0, 0), (0, 0)))
        acc, cnt = np.zeros_like(tex), np.zeros(bad.shape)
        for dy in range(3):
            for dx in (-1, 0, 1):
                acc += np.roll(col[dy:dy + h], dx, 1)
                cnt += np.roll(good[dy:dy + h], dx, 1)
        fill = bad & (cnt > 0)
        if not fill.any():
            break
        tex[fill] = acc[fill] / cnt[fill][:, None]
        bad &= ~fill
    return tex


def sky_dome(tex: np.ndarray, frame, to_scene: np.ndarray, center: np.ndarray, up: np.ndarray,
             radius: float, n: int = 40000, min_elevation_deg: float = -10.0) -> np.ndarray:
    """Splats on a sphere around center, down to min_elevation_deg below the true horizon,
    coloured from the equirect sky texture looked up in the trainer's sky frame."""
    th, tw = tex.shape[:2]
    a = np.cross(up, [1.0, 0, 0] if abs(up[0]) < 0.9 else [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(up, a)
    k = np.arange(n) + 0.5  # Fibonacci sphere
    z = 1 - 2 * k / n
    phi = math.pi * (3 - math.sqrt(5)) * k
    z = z[z > math.sin(math.radians(min_elevation_deg))]
    phi = phi[: len(z)]
    r = np.sqrt(1 - z * z)
    dirs = (r * np.cos(phi))[:, None] * a + (r * np.sin(phi))[:, None] * b + z[:, None] * up
    d = dirs @ to_scene.T
    sky_up, e1, e2 = frame
    el = np.arcsin(np.clip(d @ sky_up, -1, 1))
    az = np.arctan2(d @ e2, d @ e1)
    u = np.clip(((az / (2 * math.pi) + 0.5) * tw - 0.5).round().astype(int), 0, tw - 1)
    v = np.clip(((0.5 - el / math.pi) * th - 0.5).round().astype(int), 0, th - 1)
    size = radius * math.sqrt(4 * math.pi / n) * 0.7  # mean angle between neighbours
    return splat_records(center + radius * dirs, np.full((len(d), 3), size), tex[v, u],
                         np.ones(len(d)), np.tile([1.0, 0, 0, 0], (len(d), 1)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True, help="train.py output (model.ckpt, sky.png)")
    ap.add_argument("--out", required=True, help=".splat to write")
    ap.add_argument("--url", default="", help="public URL the .splat will have, for the viewer link")
    args = ap.parse_args()

    with open(os.path.join(args.data, "transforms.json")) as f:
        meta = json.load(f)
    frames = sorted(meta["frames"], key=lambda fr: fr["file_path"])
    names = [os.path.splitext(os.path.basename(fr["file_path"]))[0] for fr in frames]

    ds = msplat.load_dataset(args.data)
    t = msplat.GaussianTrainer(ds, msplat.street_config(iterations=1))
    t.load_checkpoint(os.path.join(args.run, "model.ckpt"))
    with tempfile.TemporaryDirectory() as tmp:
        t.export_ply(os.path.join(tmp, "scene.ply"))
        g = read_ply(os.path.join(tmp, "scene.ply"))

    # Start view in the scene frame: 1 m behind the first capture, level, looking down the road
    poses = [ds.camera_pose(i) for i in range(ds.num_train)]  # camera-to-world, OpenGL axes
    up = np.mean([p[:3, 1] for p, n in zip(poses, names) if n.endswith("_p+00")], axis=0)
    up /= np.linalg.norm(up)  # level views' up is the true vertical; the mean of all views' is not
    pa = poses[names.index("pano_a_y-015_p+00")][:3, 3]
    pb = poses[names.index("pano_b_y-015_p+00")][:3, 3]
    units_per_m = np.linalg.norm(pb - pa) / CAPTURE_SPACING_M
    road = (pb - pa) - np.dot(pb - pa, up) * up
    road /= np.linalg.norm(road)
    start = pa - units_per_m * road
    start_axes = np.stack([np.cross(road, up), -up, road], 1)  # OpenCV right, down, forward

    # Rigid map + scale to metres that puts the start view at the viewer's default camera
    view = np.array(VIEWER_DEFAULT_VIEW, np.float64).reshape(4, 4).T
    u_, _, vt = np.linalg.svd(view[:3, :3])
    w2c_rot = u_ @ vt  # the preset is rounded to 2 decimals
    default_pos = -w2c_rot.T @ view[:3, 3]
    rot = w2c_rot.T @ start_axes.T  # scene directions -> viewer world

    def to_viewer(x: np.ndarray) -> np.ndarray:
        return (x - start) @ rot.T / units_per_m + default_pos

    means = np.stack([g["x"], g["y"], g["z"]], -1).astype(np.float64)
    quats = np.stack([g[f"rot_{k}"] for k in range(4)], -1).astype(np.float64)
    quats /= np.maximum(np.linalg.norm(quats, axis=-1, keepdims=True), 1e-12)
    # The trainer draws every splat with 0.3 px^2 added to its screen-space covariance
    # (opacity unchanged), so it leaves many splats thinner than a pixel; viewers don't add
    # it and show those as hard needles. Bake the same blur into the 3D size, at the
    # splat's distance from the nearest capture and the trained focal length.
    fx = float(np.median([fr.get("fl_x", meta.get("fl_x")) for fr in frames]))
    fx *= ds.image(0).shape[1] / (frames[0].get("w") or meta["w"])
    centers = np.unique(np.round([p[:3, 3] for p in poses], 6), axis=0)
    dist = np.min([np.linalg.norm(means - c, axis=1) for c in centers], axis=0)
    blur = math.sqrt(0.3) * dist / fx
    scales = np.sqrt(np.exp(2 * np.stack([g[f"scale_{k}"] for k in range(3)], -1)) + blur[:, None] ** 2)
    scene = splat_records(to_viewer(means),
                          scales / units_per_m,
                          np.stack([g[f"f_dc_{k}"] for k in range(3)], -1) * C0 + 0.5,
                          1 / (1 + np.exp(-g["opacity"])),
                          quat_mul(quat_from_matrix(rot), quats))

    tex = fill_unseen(np.asarray(Image.open(os.path.join(args.run, "sky.png")).convert("RGB"), np.float32) / 255)
    centers = to_viewer(np.array([p[:3, 3] for p in poses]))
    dome = sky_dome(tex, sky_frame(poses), rot.T, centers.mean(0), rot @ up, radius=400.0)

    rec = np.concatenate([scene, dome])
    order = np.argsort(-(rec["scale"].sum(1) * rec["rgba"][:, 3]), kind="stable")  # large first: streams in coarse-to-fine
    rec[order].tofile(args.out)
    print(f"{args.out}: {len(scene)} scene gaussians + {len(dome)} sky dome splats, in metres")

    w2c = np.eye(4)
    w2c[:3, :3], w2c[:3, 3] = w2c_rot, view[:3, 3]
    cam = "#" + urllib.parse.quote(json.dumps([round(float(x), 4) for x in w2c.T.reshape(-1)], separators=(",", ":")))
    print("https://antimatter15.com/splat/?url=" + (args.url or os.path.basename(args.out)) + cam)


if __name__ == "__main__":
    main()
