"""Make the depth priors agree with the multi-view geometry.

Monocular depth is right about shape but wrong about scale, and wrong differently in
every view: on the Look Around example the priors of level views put surfaces 1.4-2.4x
farther than triangulation does, and those of tilted views 0.7-1.3x. Fed to training,
each view pulls the same surface to its own depth, and the gaussians end up spread
along the rays: shards on thin structures, gravel on the road.

This step measures the geometry and bends each prior onto it:

1. SIFT matches between every overlapping pair of views from different capture points,
   kept when they satisfy the epipolar constraint, then triangulated: metric 3D points
   that both captures agree on. The relative rotation of the captures is refined on
   the same matches first (a few tenths of a degree is enough to break thin structures).
2. Per view, the log ratio between triangulated and prior depth at the matched pixels
   is spread over the image with a joint bilateral kernel (image distance and prior
   depth), so a correction measured on the truss does not leak onto the sky-side
   buildings behind it. Pixels far from any anchor fall back to the view's median.
3. The road is a plane: fitted (RANSAC) to the triangulated points on road pixels and
   checked against the camera height, then road pixels get the exact ray-plane depth.
4. Confidence follows the evidence: high at anchors and on the plane, decaying with
   distance from the nearest anchor.

Writes a new dataset next to the input (images and masks are linked), with aligned
priors, the rotation fix applied to the second capture, and a rebuilt point cloud.

    python align_depth.py --data data/street --out data/street_aligned
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil

import cv2
import numpy as np
from PIL import Image

import prepare


def log(msg: str) -> None:
    print(f"[align] {msg}", flush=True)


class Views:
    def __init__(self, root: str):
        self.root = root
        with open(os.path.join(root, "transforms.json")) as f:
            self.tf = json.load(f)
        self.w, self.h, self.f = self.tf["w"], self.tf["h"], self.tf["fl_x"]
        self.cx, self.cy = self.tf["cx"], self.tf["cy"]
        self.frames = self.tf["frames"]
        self.names = [os.path.splitext(os.path.basename(fr["file_path"]))[0] for fr in self.frames]

    def c2w(self, i: int) -> np.ndarray:
        m = np.array(self.frames[i]["transform_matrix"], dtype=np.float64)
        m[:3, 1:3] *= -1  # OpenGL -> OpenCV
        return m

    def pano(self, i: int) -> str:
        return self.names[i].split("_y")[0]

    def load(self, i: int, key: str) -> np.ndarray:
        fr = self.frames[i]
        path = os.path.join(self.root, fr[key])
        if path.endswith(".npy"):
            return np.load(path)
        return np.asarray(Image.open(path).convert("L")) > 127

    def rays(self, i: int, px: np.ndarray, rot: np.ndarray | None = None) -> np.ndarray:
        m = self.c2w(i)
        R = m[:3, :3] if rot is None else rot @ m[:3, :3]
        d = np.stack([(px[:, 0] + 0.5 - self.cx) / self.f, (px[:, 1] + 0.5 - self.cy) / self.f,
                      np.ones(len(px))], -1) @ R.T
        return d / np.linalg.norm(d, axis=1, keepdims=True)


def rotvec(w: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def match_all(v: Views, n_features: int = 8000, max_axis_angle: float = 75.0):
    """SIFT matches between every pair of views from different panoramas whose axes are close."""
    sift = cv2.SIFT_create(nfeatures=n_features)
    feats = []
    for i in range(len(v.names)):
        img = np.asarray(Image.open(os.path.join(v.root, v.frames[i]["file_path"])).convert("L"))
        mask = (v.load(i, "mask_path") & ~v.load(i, "prior_sky_mask_path")).astype(np.uint8) * 255
        kp, desc = sift.detectAndCompute(img, mask)
        feats.append((np.array([k.pt for k in kp], np.float64), desc))
    bf = cv2.BFMatcher()
    pairs = []
    for i in range(len(v.names)):
        for j in range(len(v.names)):
            if v.pano(i) >= v.pano(j):
                continue
            ai, aj = v.c2w(i)[:3, 2], v.c2w(j)[:3, 2]
            if math.degrees(math.acos(np.clip(ai @ aj, -1, 1))) > max_axis_angle:
                continue
            di, dj = feats[i][1], feats[j][1]
            if di is None or dj is None or len(di) < 2 or len(dj) < 2:
                continue
            good = [m for m, n in bf.knnMatch(di, dj, k=2) if m.distance < 0.75 * n.distance]
            if len(good) < 8:
                continue
            pi = feats[i][0][[m.queryIdx for m in good]]
            pj = feats[j][0][[m.trainIdx for m in good]]
            pairs.append((i, j, pi, pj))
    return pairs


def coplanarity(ra, rb, t):
    n = np.cross(ra, t)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return np.arcsin(np.clip(np.sum(n * rb, axis=1), -1, 1))


def refine_rotation(ra, rb, t, iters: int = 25) -> np.ndarray:
    """Small rotation of the second capture minimizing the robust epipolar error."""
    w = np.zeros(3)
    for _ in range(iters):
        r = coplanarity(ra, rb @ rotvec(w).T, t)
        J = np.zeros((len(r), 3))
        for k in range(3):
            d = np.zeros(3); d[k] = 1e-6
            J[:, k] = (coplanarity(ra, rb @ rotvec(w + d).T, t) - r) / 1e-6
        wt = 1.0 / np.maximum(1.0, np.abs(r) / math.radians(0.3))
        w -= np.linalg.solve((J * wt[:, None]).T @ J + 1e-8 * np.eye(3), (J * wt[:, None]).T @ r)
    return w


def triangulate(ca, ra, cb, rb):
    """Midpoints of the closest points of two ray bundles; distances along each ray."""
    w0 = ca - cb
    aa, bb, ab = np.sum(ra * ra, 1), np.sum(rb * rb, 1), np.sum(ra * rb, 1)
    ad, bd = np.sum(ra * w0, 1), np.sum(rb * w0, 1)
    den = aa * bb - ab * ab
    s = (ab * bd - bb * ad) / np.maximum(den, 1e-12)
    u = (aa * bd - ab * ad) / np.maximum(den, 1e-12)
    X = 0.5 * ((ca + s[:, None] * ra) + (cb + u[:, None] * rb))
    return X, s, u


def bilateral_field(h, w, px, logr, logd_anchor, logd_img, sigma_px=110.0, sigma_d=0.35):
    """Joint bilateral spread of anchor log ratios over the image (at 1/4 resolution)."""
    s = 4
    hs, ws = h // s, w // s
    yy, xx = np.mgrid[0:hs, 0:ws]
    gx, gy = (xx + 0.5) * s, (yy + 0.5) * s
    ld = logd_img[::s, ::s][:hs, :ws]
    num = np.zeros((hs, ws)); den = np.zeros((hs, ws)); near = np.full((hs, ws), np.inf)
    for (x, y), r, la in zip(px, logr, logd_anchor):
        d2 = (gx - x) ** 2 + (gy - y) ** 2
        wgt = np.exp(-d2 / (2 * sigma_px ** 2)) * np.exp(-((ld - la) ** 2) / (2 * sigma_d ** 2))
        num += wgt * r
        den += wgt
        near = np.minimum(near, np.sqrt(d2) + 400.0 * np.abs(ld - la))
    up = lambda a: np.asarray(Image.fromarray(a.astype(np.float32)).resize((w, h), Image.BILINEAR))
    return up(num), up(den), up(near)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset written by prepare.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seg-model", default="nvidia/segformer-b2-finetuned-ade-512-512")
    args = ap.parse_args()

    v = Views(args.data)
    panos = sorted({v.pano(i) for i in range(len(v.names))})
    if len(panos) != 2:
        raise SystemExit("expects two capture points")
    pa, pb = panos
    centers = {p: v.c2w(next(i for i in range(len(v.names)) if v.pano(i) == p))[:3, 3] for p in panos}
    t = centers[pb] - centers[pa]

    pairs = match_all(v)
    ra = np.concatenate([v.rays(i, pi) for i, j, pi, pj in pairs])
    rb = np.concatenate([v.rays(j, pj) for i, j, pi, pj in pairs])
    r0 = coplanarity(ra, rb, t / np.linalg.norm(t))
    log(f"{len(ra)} matches over {len(pairs)} view pairs; epipolar error with the metadata poses: "
        f"median {np.degrees(np.median(np.abs(r0))):.3f} deg")
    inl = np.abs(r0) < math.radians(0.5)
    w = refine_rotation(ra[inl], rb[inl], t / np.linalg.norm(t))
    R = rotvec(w)
    r1 = coplanarity(ra, rb @ R.T, t / np.linalg.norm(t))
    log(f"rotation of {pb} refined by {np.degrees(w).round(3)} deg; median error now "
        f"{np.degrees(np.median(np.abs(r1[np.abs(r1) < math.radians(0.5)]))):.3f} deg")

    # Triangulate inliers with the refined rotation; collect anchors per view
    anchors = {i: [] for i in range(len(v.names))}  # (px, py, depth along the view axis, X)
    all_X = []
    k0 = 0
    for i, j, pi, pj in pairs:
        n = len(pi)
        sel = slice(k0, k0 + n)
        k0 += n
        rai, rbj = ra[sel], rb[sel] @ R.T
        ok = np.abs(coplanarity(rai, rbj, t / np.linalg.norm(t))) < math.radians(0.15)
        X, s, u = triangulate(centers[pa], rai, centers[pb], rbj)
        parallax = np.degrees(np.arccos(np.clip(np.sum(rai * rbj, 1), -1, 1)))
        ok &= (s > 1.0) & (u > 1.0) & (s < 80) & (parallax > 2.0)
        mi, mj = v.c2w(i), v.c2w(j)
        Rj = R @ mj[:3, :3]
        for k in np.nonzero(ok)[0]:
            anchors[i].append((pi[k, 0], pi[k, 1], (X[k] - mi[:3, 3]) @ mi[:3, 2], X[k]))
            anchors[j].append((pj[k, 0], pj[k, 1], (X[k] - mj[:3, 3]) @ Rj[:, 2], X[k]))
            all_X.append(X[k])
    all_X = np.array(all_X)
    log(f"{len(all_X)} triangulated anchors (parallax > 2 deg)")

    # Road plane: SegFormer road class, RANSAC over anchors on road pixels
    road_masks = {}
    for i in range(len(v.names)):
        img = Image.open(os.path.join(v.root, v.frames[i]["file_path"])).convert("RGB")
        road_masks[i] = road_mask(img, args.seg_model)
    road_pts = []
    for i, an in anchors.items():
        for x, y, d, X in an:
            if road_masks[i][min(int(y), v.h - 1), min(int(x), v.w - 1)]:
                road_pts.append(X)
    road_pts = np.array(road_pts)
    plane = None
    if len(road_pts) >= 20:
        rng = np.random.default_rng(0)
        best = (0, None)
        for _ in range(2000):
            p3 = road_pts[rng.choice(len(road_pts), 3, replace=False)]
            n = np.cross(p3[1] - p3[0], p3[2] - p3[0])
            if np.linalg.norm(n) < 1e-9:
                continue
            n /= np.linalg.norm(n)
            if abs(n[2]) < 0.9:  # roughly horizontal (z is up)
                continue
            dist = np.abs((road_pts - p3[0]) @ n)
            cnt = int((dist < 0.08).sum())
            if cnt > best[0]:
                best = (cnt, (n, p3[0]))
        if best[1] is not None:
            n, p0 = best[1]
            inl_pts = road_pts[np.abs((road_pts - p0) @ n) < 0.08]
            c = inl_pts.mean(0)
            n = np.linalg.svd(inl_pts - c)[2][2]
            n = n if n[2] > 0 else -n
            plane = (n, c)
            heights = [float((centers[p] - c) @ n) for p in panos]
            log(f"road plane from {len(inl_pts)}/{len(road_pts)} road anchors; camera heights "
                f"{heights[0]:.2f} m and {heights[1]:.2f} m; tilt {math.degrees(math.acos(n[2])):.2f} deg")
            if not all(1.5 < hgt < 4.5 for hgt in heights):
                log("camera heights implausible: road plane not used")
                plane = None
    else:
        log(f"only {len(road_pts)} road anchors: no road plane")

    # Write the aligned dataset
    os.makedirs(args.out, exist_ok=True)
    for sub in ("images", "priors/sky", "priors/mask"):
        dst = os.path.join(args.out, sub)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(os.path.join(args.data, sub)), dst)
    for sub in ("priors/depth", "priors/confidence", "priors/road"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    report = {}
    frames = []
    for i, name in enumerate(v.names):
        prior = v.load(i, "prior_depth_path")
        valid = prior > 0
        an = anchors[i]
        m = v.c2w(i)
        if v.pano(i) == pb:
            m[:3, :3] = R @ m[:3, :3]
        aligned = prior.copy()
        conf = np.zeros_like(prior)
        if len(an) >= 5:
            px = np.array([(a[0], a[1]) for a in an])
            z = np.array([a[2] for a in an])
            dprior = prior[np.clip(px[:, 1].astype(int), 0, v.h - 1), np.clip(px[:, 0].astype(int), 0, v.w - 1)]
            ok = (dprior > 0) & (z > 0.5)
            px, z, dprior = px[ok], z[ok], dprior[ok]
            logr = np.log(z / dprior)
            med = float(np.median(logr))
            keep = np.abs(logr - med) < 0.6
            px, logr, la = px[keep], logr[keep], np.log(dprior[keep])
            logd = np.log(np.where(valid, prior, 1.0))
            num, den, near = bilateral_field(v.h, v.w, px, logr, la, logd)
            blend = den / (den + 0.15)
            field = blend * (num / np.maximum(den, 1e-9)) + (1 - blend) * med
            aligned = np.where(valid, prior * np.exp(field), 0.0)
            conf = np.where(valid, np.clip(np.exp(-near / 120.0), 0.15, 1.0), 0.0)
            report[name] = {"anchors": int(len(px)), "median_ratio_before": round(float(np.exp(med)), 3)}
        else:
            med = 0.0
            conf = np.where(valid, 0.15, 0.0)
            report[name] = {"anchors": int(len(an)), "median_ratio_before": None}

        if plane is not None:
            n, c = plane
            jj, ii = np.mgrid[0:v.h, 0:v.w]
            d = np.stack([(ii + 0.5 - v.cx) / v.f, (jj + 0.5 - v.cy) / v.f, np.ones_like(ii, float)], -1) @ m[:3, :3].T
            denom = d @ n
            lam = ((c - m[:3, 3]) @ n) / np.where(np.abs(denom) > 1e-6, denom, np.nan)
            road = road_masks[i] & np.isfinite(lam) & (lam > 0) & valid
            zplane = lam * (d @ m[:3, :3][:, 2])  # depth along the view axis
            aligned = np.where(road, zplane, aligned)
            conf = np.where(road, 1.0, conf)
            report[name]["road_pixels"] = int(road.sum())
        np.save(os.path.join(args.out, "priors/depth", name + ".npy"), aligned.astype(np.float32))
        np.save(os.path.join(args.out, "priors/confidence", name + ".npy"), (conf * prepare.confidence(aligned, aligned > 0) ** 0.5).astype(np.float32))
        Image.fromarray(road_masks[i].astype(np.uint8) * 255).save(os.path.join(args.out, "priors/road", name + ".png"))
        fr = dict(v.frames[i])
        mg = m.copy()
        mg[:3, 1:3] *= -1
        fr["transform_matrix"] = mg.tolist()
        frames.append(fr)

    tf = dict(v.tf)
    tf["frames"] = frames
    with open(os.path.join(args.out, "transforms.json"), "w") as f:
        json.dump(tf, f, indent=1)
    with open(os.path.join(args.out, "alignment.json"), "w") as f:
        json.dump({"rotation_fix_deg": np.degrees(w).tolist(), "anchors": int(len(all_X)),
                   "road_plane": None if plane is None else {"normal": plane[0].tolist(), "point": plane[1].tolist()},
                   "views": report}, f, indent=1)
    if os.path.exists(os.path.join(args.data, "sizing.json")):
        shutil.copy(os.path.join(args.data, "sizing.json"), args.out)
    log(f"wrote {args.out}")


_SEG = {}


def road_mask(img: Image.Image, name: str) -> np.ndarray:
    """ADE20K road / sidewalk / path pixels (SegFormer)."""
    import torch
    from transformers import SegformerForSemanticSegmentation

    if name not in _SEG:
        model = SegformerForSemanticSegmentation.from_pretrained(name).eval()
        labels = {i: n.split(",")[0].strip().lower() for i, n in model.config.id2label.items()}
        ids = [i for i, n in labels.items() if n in {"road", "sidewalk", "path", "dirt track"}]
        _SEG[name] = (model, ids)
    model, ids = _SEG[name]
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    x = (np.asarray(img.resize((512, 512), Image.BILINEAR), np.float32) / 255.0 - mean) / std
    with torch.no_grad():
        logits = model(pixel_values=torch.from_numpy(x.transpose(2, 0, 1))[None]).logits
        logits = torch.nn.functional.interpolate(logits, size=(img.height, img.width), mode="bilinear",
                                                 align_corners=False)
    return np.isin(logits[0].argmax(0).numpy(), ids)


if __name__ == "__main__":
    main()
