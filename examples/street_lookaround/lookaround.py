"""Apple Look Around panoramas to pinhole views.

A Look Around panorama is six face images plus metadata (the *-meta.json files): four
side faces (0 front, 1 right, 2 back, 3 left) and two caps (4 top, 5 bottom). Each side
face is an equirectangular segment of the sphere around the capture point:

    longitude = yaw + ((u + 0.5) / W - 0.5) * fov_s     (clockwise from the heading)
    latitude  = cy  + (0.5 - (v + 0.5) / H) * fov_h     (up)

so the formulas hold at any file size; the metadata's face_sizes only describe the
full-resolution originals. Panoramas are placed in a local east-north-up frame from
their east/north offsets and elevation; the heading points the panorama's forward
direction along the bearing -heading_rad (clockwise from north), which matches the
direction of travel between consecutive captures.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

FACE_NAMES = {0: "front", 1: "right", 2: "back", 3: "left", 4: "top", 5: "bottom"}
SIDE_FACES = (0, 1, 2, 3)
CAP_FACES = (4, 5)


@dataclass
class Panorama:
    name: str
    meta: dict
    faces: dict[int, np.ndarray] = field(default_factory=dict)  # face index -> (H, W, 3) float32
    files: dict[int, str] = field(default_factory=dict)
    # Caps (top/bottom) are equirect patches in a frame whose forward axis is the zenith
    # or nadir; the metadata's yaw/pitch/roll leave their in-plane orientation ambiguous,
    # so it is calibrated against the side faces where they overlap: (axis, mirror).
    cap_orient: dict[int, tuple[int, bool]] = field(default_factory=dict)

    @classmethod
    def load(cls, name: str, meta_path: str, face_files: dict[int, str]) -> "Panorama":
        with open(meta_path) as f:
            meta = json.load(f)
        pano = cls(name, meta)
        for idx, path in face_files.items():
            pano.faces[idx] = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
            pano.files[idx] = path
        for idx in CAP_FACES:
            if idx in pano.faces:
                pano.cap_orient[idx] = pano.calibrate_cap(idx)
        return pano

    # ── Placement ────────────────────────────────────────────────────────────

    @property
    def position(self) -> np.ndarray:
        """Capture point (east, north, up) in metres, in the local frame of the metadata."""
        m = self.meta
        return np.array([m["east"], m["north"], m["elevation"]], dtype=np.float64)

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward, right and up unit vectors of the panorama in east-north-up."""
        bearing = -self.meta["heading_rad"]
        forward = np.array([math.sin(bearing), math.cos(bearing), 0.0])
        right = np.array([math.cos(bearing), -math.sin(bearing), 0.0])
        return forward, right, np.array([0.0, 0.0, 1.0])

    # ── Sampling ─────────────────────────────────────────────────────────────

    def _cap_axes(self, idx: int, orient: tuple[int, bool]):
        f, r, u = self.basis()
        forward = u if idx == 4 else -u
        right = [f, r, -f, -r][orient[0]]
        down = np.cross(forward, right)  # x forward, y right, z down
        return forward, right, down

    def _sample_face(self, idx: int, dirs: np.ndarray, orient=None):
        """(rgb, weight) of one face along directions; weight feathers to 0 at its border."""
        cam = self.meta["camera_metadata"][idx]
        img = self.faces[idx]
        h, w = img.shape[:2]
        if idx in SIDE_FACES:
            f, r, u = self.basis()
            x, y, z = dirs @ f, dirs @ r, dirs @ u
            lon = np.arctan2(y, x)
            lat = np.arcsin(np.clip(z / np.linalg.norm(dirs, axis=-1), -1.0, 1.0))
            dlon = (lon - cam["yaw"] + math.pi) % (2 * math.pi) - math.pi
            dlat = lat - cam["cy"]
        else:
            forward, right, down = self._cap_axes(idx, orient or self.cap_orient[idx])
            x, y, z = dirs @ forward, dirs @ right, dirs @ down
            dlon = np.arctan2(y, x)
            if (orient or self.cap_orient[idx])[1]:
                dlon = -dlon
            dlat = np.arcsin(np.clip(-z / np.linalg.norm(dirs, axis=-1), -1.0, 1.0))
            dlon = np.where(x > 0, dlon, math.pi)  # behind the cap: outside
        half_s, half_h = cam["fov_s"] / 2, cam["fov_h"] / 2
        weight = np.clip(np.minimum(half_s - np.abs(dlon), half_h - np.abs(dlat)) / math.radians(3), 0, 1)
        px = (dlon / cam["fov_s"] + 0.5) * w - 0.5
        py = (0.5 - dlat / cam["fov_h"]) * h - 0.5
        return bilinear(img, px, py), weight

    def calibrate_cap(self, idx: int) -> tuple[int, bool]:
        """Cap orientation that best matches the side faces in their overlap band."""
        f, r, u = self.basis()
        # directions in the band the cap shares with the side faces
        lat0 = math.radians(64) if idx == 4 else math.radians(-32)
        lons = np.radians(np.arange(-120, 121, 0.5))
        lats = lat0 + np.radians(np.linspace(-3, 3, 13))
        L, A = np.meshgrid(lons, lats)
        dirs = (np.cos(A)[..., None] * (np.cos(L)[..., None] * f + np.sin(L)[..., None] * r)
                + np.sin(A)[..., None] * u)
        side = np.zeros(dirs.shape[:-1] + (3,))
        sw = np.zeros(dirs.shape[:-1])
        for k in SIDE_FACES:
            if k in self.faces:
                c, wk = self._sample_face(k, dirs)
                side += wk[..., None] * c
                sw += wk
        best, best_err = (0, False), np.inf
        for axis in range(4):
            for mirror in (False, True):
                c, wc = self._sample_face(idx, dirs, (axis, mirror))
                ok = (wc > 0.5) & (sw > 0.5)
                if ok.sum() < 200:
                    continue
                err = float(np.mean(np.abs(c[ok] - side[ok] / sw[ok][:, None])))
                if err < best_err:
                    best, best_err = (axis, mirror), err
        return best

    def sample(self, dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Colour along world directions (..., 3), feather-blended across face seams.

        Returns (rgb (..., 3), coverage (...) in [0, 1]); coverage 0 where no loaded
        face sees the direction (the back face is often missing).
        """
        rgb = np.zeros(dirs.shape[:-1] + (3,), np.float64)
        wsum = np.zeros(dirs.shape[:-1], np.float64)
        for idx in SIDE_FACES + CAP_FACES:
            if idx not in self.faces:
                continue
            # feather weight: angular distance to the face border (side seams overlap ~5.6 deg)
            c, weight = self._sample_face(idx, dirs)
            if not weight.any():
                continue
            rgb += weight[..., None] * c
            wsum += weight
        covered = wsum > 1e-6
        rgb[covered] /= wsum[covered][..., None]
        return rgb.astype(np.float32), np.clip(wsum, 0, 1).astype(np.float32)


def bilinear(img: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    px = np.clip(px, 0, w - 1)
    py = np.clip(py, 0, h - 1)
    x0 = np.floor(px).astype(np.int64)
    y0 = np.floor(py).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (px - x0)[..., None]
    fy = (py - y0)[..., None]
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bottom = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


# ── Pinhole views ────────────────────────────────────────────────────────────


@dataclass
class View:
    name: str
    pano: str
    yaw_deg: float
    pitch_deg: float
    width: int
    height: int
    fx: float
    cam_to_world: np.ndarray  # 4x4, OpenCV axes (x right, y down, z forward)

    @property
    def cx(self) -> float:
        return self.width / 2

    @property
    def cy(self) -> float:
        return self.height / 2

    def gl_cam_to_world(self) -> np.ndarray:
        """Nerfstudio / OpenGL axes (x right, y up, z back)."""
        m = self.cam_to_world.copy()
        m[:3, 1:3] *= -1
        return m

    def rays(self, supersample: int = 1) -> np.ndarray:
        """World directions through the pixel centres, (H*s, W*s, 3)."""
        s = supersample
        j, i = np.meshgrid(np.arange(self.height * s), np.arange(self.width * s), indexing="ij")
        x = ((i + 0.5) / s - self.cx) / self.fx
        y = ((j + 0.5) / s - self.cy) / self.fx
        d = np.stack([x, y, np.ones_like(x)], -1) @ self.cam_to_world[:3, :3].T
        return d / np.linalg.norm(d, axis=-1, keepdims=True)


def make_view(pano: Panorama, origin: np.ndarray, yaw_deg: float, pitch_deg: float,
              width: int, height: int, hfov_deg: float, name: str) -> View:
    f, r, u = pano.basis()
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    heading = math.cos(yaw) * f + math.sin(yaw) * r
    forward = math.cos(pitch) * heading + math.sin(pitch) * u
    right = -math.sin(yaw) * f + math.cos(yaw) * r
    down = np.cross(forward, right)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, down, forward, pano.position - origin
    fx = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    return View(name, pano.name, yaw_deg, pitch_deg, width, height, fx, m)


def render_view(pano: Panorama, view: View, supersample: int = 2) -> tuple[Image.Image, np.ndarray]:
    """Pinhole image of the panorama (antialiased by supersampling) and its coverage."""
    rgb, cov = pano.sample(view.rays(supersample))
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8))
    img = img.resize((view.width, view.height), Image.BOX)
    cov = np.asarray(Image.fromarray((cov * 255).astype(np.uint8)).resize((view.width, view.height), Image.BOX))
    return img, cov.astype(np.float32) / 255.0
