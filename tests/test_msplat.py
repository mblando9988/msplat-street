"""msplat test suite."""

import pytest
import numpy as np
import tempfile
import os

GARDEN = os.path.join(os.path.dirname(__file__), "..", "datasets", "mipnerf360", "garden")
HAS_GARDEN = os.path.isdir(GARDEN)


# ── Import tests ─────────────────────────────────────────────────────────────


def test_import():
    import msplat
    assert hasattr(msplat, "GaussianTrainer")
    assert hasattr(msplat, "TrainingConfig")
    assert hasattr(msplat, "Dataset")
    assert hasattr(msplat, "load_dataset")


def test_training_config_defaults():
    from msplat import TrainingConfig

    cfg = TrainingConfig()
    assert cfg.iterations == 30000
    assert cfg.sh_degree == 3
    assert cfg.ssim_weight == pytest.approx(0.2)
    assert cfg.refine_every == 100
    assert cfg.warmup_length == 500


def test_training_config_custom():
    from msplat import TrainingConfig

    cfg = TrainingConfig(iterations=100, sh_degree=1, ssim_weight=0.0)
    assert cfg.iterations == 100
    assert cfg.sh_degree == 1
    assert cfg.ssim_weight == 0.0


def test_training_config_mutable():
    from msplat import TrainingConfig

    cfg = TrainingConfig()
    cfg.iterations = 500
    assert cfg.iterations == 500


# ── Dataset tests ────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_load_dataset():
    from msplat import Dataset

    ds = Dataset(GARDEN, downscale_factor=4.0, eval_mode=True, test_every=8)
    assert ds.num_train > 0
    assert ds.num_test > 0
    assert ds.num_train + ds.num_test > 100


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_load_dataset_no_eval():
    from msplat import Dataset

    ds = Dataset(GARDEN, downscale_factor=4.0, eval_mode=False)
    assert ds.num_train > 0
    assert ds.num_test == 0


# ── Training tests ───────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_train_short():
    """Train 50 steps at 4x downscale — verify it runs without error."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=50, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    steps_seen = []
    trainer.train(lambda s: steps_seen.append(s.iteration), callback_every=10)

    assert trainer.iteration == 50
    assert trainer.splat_count > 100000
    assert steps_seen == [10, 20, 30, 40, 50]


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_step_by_step():
    """Manual step loop works."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=10, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    for _ in range(10):
        stats = trainer.step()

    assert stats.iteration == 10
    assert stats.splat_count > 0
    assert stats.ms_per_step > 0


# ── Render tests ─────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_render():
    """Render produces valid image array."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer, sync

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=10, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    for _ in range(10):
        trainer.step()

    img = trainer.render(0)
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.float32
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.shape[0] > 0 and img.shape[1] > 0
    # Values should be in [0, 1] range (approximately)
    assert img.min() >= -0.1
    assert img.max() <= 1.5


# ── Export tests ─────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_export_ply():
    """PLY export creates a valid file."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=10, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    for _ in range(10):
        trainer.step()

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name

    try:
        trainer.export_ply(path)
        assert os.path.exists(path)
        size = os.path.getsize(path)
        assert size > 1000  # non-trivial file
    finally:
        os.unlink(path)


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_export_splat():
    """Splat export creates a valid file."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=10, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    for _ in range(10):
        trainer.step()

    with tempfile.NamedTemporaryFile(suffix=".splat", delete=False) as f:
        path = f.name

    try:
        trainer.export_splat(path)
        assert os.path.exists(path)
        size = os.path.getsize(path)
        assert size > 1000
    finally:
        os.unlink(path)


# ── Eval tests ───────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_evaluate():
    """Evaluation returns valid metrics dict."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0, eval_mode=True, test_every=8)
    cfg = TrainingConfig(iterations=50, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    trainer.train(lambda s: None, callback_every=50)
    metrics = trainer.evaluate()

    assert "psnr" in metrics
    assert "ssim" in metrics
    assert "l1" in metrics
    assert "num_test" in metrics
    assert metrics["num_test"] > 0
    assert metrics["psnr"] > 10  # sanity — should be at least somewhat trained
    assert 0 < metrics["ssim"] < 1
    assert metrics["l1"] > 0


# ── Checkpoint tests ────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_checkpoint_save_load():
    """Save checkpoint, load it, verify state is preserved."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=100, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    for _ in range(50):
        trainer.step()

    splats_at_50 = trainer.splat_count

    with tempfile.NamedTemporaryFile(suffix=".msplat", delete=False) as f:
        ckpt_path = f.name

    try:
        trainer.save_checkpoint(ckpt_path)
        assert os.path.exists(ckpt_path)
        assert os.path.getsize(ckpt_path) > 1000

        # Load into a fresh trainer
        ds2 = Dataset(GARDEN, downscale_factor=4.0)
        cfg2 = TrainingConfig(iterations=100, num_downscales=0)
        trainer2 = GaussianTrainer(ds2, cfg2)
        trainer2.load_checkpoint(ckpt_path)

        assert trainer2.iteration == 50
        assert trainer2.splat_count == splats_at_50
    finally:
        os.unlink(ckpt_path)


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_checkpoint_resume_training():
    """Train 50 → save → load → train 50 more. Verify it completes."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=100, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    for _ in range(50):
        trainer.step()

    with tempfile.NamedTemporaryFile(suffix=".msplat", delete=False) as f:
        ckpt_path = f.name

    try:
        trainer.save_checkpoint(ckpt_path)

        # Resume in a new trainer
        ds2 = Dataset(GARDEN, downscale_factor=4.0)
        cfg2 = TrainingConfig(iterations=100, num_downscales=0)
        trainer2 = GaussianTrainer(ds2, cfg2)
        trainer2.load_checkpoint(ckpt_path)

        for _ in range(50):
            stats = trainer2.step()

        assert trainer2.iteration == 100
        assert stats.splat_count > 0
        assert stats.ms_per_step > 0
    finally:
        os.unlink(ckpt_path)


# ── Prior-guided training ───────────────────────────────────────────────────


def _write_png_gray(path, img):
    """Minimal 8-bit grayscale PNG writer (stdlib only; CI has no Pillow)."""
    import struct
    import zlib

    img = np.ascontiguousarray(img, dtype=np.uint8)
    h, w = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw)))
        f.write(chunk(b"IEND", b""))


def _make_priors(tmpdir, n_steps=30):
    """Depth priors from a short warm-up render, plus synthetic sky (top rows) and keep
    masks, keyed by image name like msplat-prior writes them."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    trainer = GaussianTrainer(ds, TrainingConfig(iterations=n_steps, num_downscales=0))
    for _ in range(n_steps):
        trainer.step()

    names = sorted(os.listdir(os.path.join(GARDEN, "images")))
    assert len(names) == ds.num_train
    for sub in ("depth", "confidence", "sky", "mask"):
        os.makedirs(os.path.join(tmpdir, sub), exist_ok=True)
    for i, name in enumerate(names):
        stem = os.path.splitext(name)[0]
        depth, alpha = trainer.render_depth(i)
        assert depth.ndim == 2 and depth.shape == alpha.shape
        # Half resolution: priors are sampled by normalized position, not pixel index
        np.save(os.path.join(tmpdir, "depth", stem + ".npy"), depth[::2, ::2].astype(np.float32))
        np.save(os.path.join(tmpdir, "confidence", stem + ".npy"),
                np.clip(alpha[::2, ::2], 0, 1).astype(np.float16))
        h, w = depth.shape
        sky = np.zeros((h, w), np.uint8)
        sky[: h // 10] = 255
        _write_png_gray(os.path.join(tmpdir, "sky", stem + ".png"), sky)
        keep = np.full((h, w), 255, np.uint8)
        keep[h // 2: h // 2 + 8, w // 2: w // 2 + 8] = 0
        _write_png_gray(os.path.join(tmpdir, "mask", stem + ".png"), keep)
    return names


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_render_depth():
    """render_depth returns finite expected depth where alpha is high."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    trainer = GaussianTrainer(ds, TrainingConfig(iterations=20, num_downscales=0))
    for _ in range(20):
        trainer.step()
    depth, alpha = trainer.render_depth(0)
    img = trainer.render(0)
    assert depth.shape == alpha.shape == img.shape[:2]
    assert np.isfinite(depth).all() and np.isfinite(alpha).all()
    assert alpha.min() >= -1e-4 and alpha.max() <= 1 + 1e-4
    covered = alpha > 0.5
    assert covered.mean() > 0.5
    assert (depth[covered] > 0).all()


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_street_preset_trains():
    """Prior-guided training (depth, sky, masks, exposure, scale cap) runs end to end."""
    import msplat
    from msplat import Dataset, GaussianTrainer

    with tempfile.TemporaryDirectory() as tmp:
        _make_priors(tmp)
        ds = Dataset(GARDEN, downscale_factor=4.0, prior_dir=tmp)
        counts = ds.prior_counts()
        assert counts["depth"] == counts["sky"] == counts["mask"] == ds.num_train

        cfg = msplat.street_config(iterations=120, num_downscales=1, resolution_schedule=60,
                                   refine_every=50, warmup_length=40, sky_width=64, sky_height=16)
        assert cfg.learn_sky and cfg.exposure_compensation and cfg.sh_degree == 1
        trainer = GaussianTrainer(ds, cfg)
        losses = []
        for _ in range(120):
            stats = trainer.step()
            if stats.iteration % 20 == 0:
                losses.append(trainer.prior_losses())
        assert trainer.splat_count > 0
        for l in losses:
            assert all(np.isfinite(v) and v >= 0 for v in l.values()), l
        assert losses[-1]["depth"] > 0  # depth prior active on every camera

        img = trainer.render(0)
        assert np.isfinite(img).all()
        depth, alpha = trainer.render_depth(0)
        assert np.isfinite(depth).all()

        sky_png = os.path.join(tmp, "sky_out.png")
        trainer.export_sky(sky_png)
        assert os.path.getsize(sky_png) > 0

        ckpt = os.path.join(tmp, "street.msplat")
        trainer.save_checkpoint(ckpt)
        trainer2 = GaussianTrainer(ds, cfg)
        trainer2.load_checkpoint(ckpt)
        assert trainer2.iteration == 120
        np.testing.assert_allclose(trainer2.render(0), img, atol=1e-5)


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_scale_ratio_cap_and_exposure_only():
    """Options that do not need the aux rasterizer work on their own."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=60, num_downscales=0, max_scale_ratio=4.0,
                         exposure_compensation=True)
    trainer = GaussianTrainer(ds, cfg)
    for _ in range(60):
        trainer.step()
    assert trainer.splat_count > 0
    assert np.isfinite(trainer.render(0)).all()


def test_street_config_overrides():
    import msplat

    cfg = msplat.street_config(depth_weight=0.5, sh_degree=2)
    assert cfg.depth_weight == pytest.approx(0.5)
    assert cfg.sh_degree == 2
    assert cfg.learn_sky
    assert msplat.TrainingConfig().depth_weight == 0.0
    assert not msplat.TrainingConfig().learn_sky


# ── GPU primitives (no dataset needed) ──────────────────────────────────────


@pytest.mark.parametrize("n,bits,hi", [
    (1, 32, 10), (5, 32, 4), (1023, 32, 2**30), (1024, 32, 2**30), (1025, 32, 7),
    (70_000, 32, 2**30), (5_000, 64, 2**63), (2_049, 64, 3),
])
def test_gpu_radix_sort_matches_numpy(n, bits, hi):
    from msplat import _core

    rng = np.random.default_rng(n + bits)
    keys = rng.integers(0, hi, n, dtype=np.uint64)
    sorted_keys, order = _core._gpu_radix_sort(keys, bits)
    ref = np.argsort(keys, kind="stable")
    np.testing.assert_array_equal(sorted_keys, keys[ref])
    np.testing.assert_array_equal(order, ref)  # stable


def _knn3_bruteforce(pts):
    d2 = ((pts[:, None, :].astype(np.float64) - pts[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    return np.sqrt(np.sort(d2, axis=1)[:, :3]).mean(axis=1)


@pytest.mark.parametrize("kind", ["uniform", "street", "duplicates"])
def test_gpu_knn_matches_bruteforce(kind):
    from msplat import _core

    rng = np.random.default_rng(7)
    if kind == "uniform":
        pts = rng.uniform(-1, 1, (3000, 3))
    elif kind == "street":  # thin road plane + facade + sparse far points
        road = np.c_[rng.uniform(-20, 20, 2000), rng.normal(0, 0.02, 2000), rng.uniform(0, 80, 2000)]
        facade = np.c_[8 + rng.normal(0, 0.05, 800), rng.uniform(0, 10, 800), rng.uniform(0, 80, 800)]
        pts = np.vstack([road, facade, rng.normal(0, 300, (200, 3))])
    else:
        pts = np.vstack([rng.uniform(-1, 1, (500, 3)), np.repeat(rng.uniform(-1, 1, (8, 3)), 4, axis=0)])
    pts = pts.astype(np.float32)
    got = _core._gpu_knn3_mean_dist(pts)
    np.testing.assert_allclose(got, _knn3_bruteforce(pts), rtol=1e-4, atol=1e-6)


# ── GPU image pipeline (no dataset needed) ───────────────────────────────────


def _area_weights(src, dst):
    """(dst, src) overlap weights of the area (box) resample, rows normalized."""
    s = src / dst
    w = np.zeros((dst, src))
    for d in range(dst):
        a, b = d * s, (d + 1) * s
        for i in range(int(np.floor(a)), min(int(np.ceil(b)), src)):
            w[d, i] = min(i + 1, b) - max(i, a)
    return w / w.sum(1, keepdims=True)


def _area_resize(img, out_w, out_h):
    wy = _area_weights(img.shape[0], out_h)
    wx = _area_weights(img.shape[1], out_w)
    return np.einsum("yh,hwc,xw->yxc", wy, np.asarray(img, np.float64), wx)


@pytest.mark.parametrize("src,dst", [
    ((48, 64), (24, 32)), ((50, 75), (17, 29)), ((40, 60), (40, 60)), ((10, 12), (25, 31)),
])
def test_gpu_resize_area_matches_numpy(src, dst):
    from msplat import _core

    rng = np.random.default_rng(src[0] * 1000 + dst[1])
    rgba = rng.integers(0, 256, (*src, 4), dtype=np.uint8)
    got = _core._gpu_resize_area(rgba, dst[1], dst[0])
    assert got.shape == (*dst, 3) and got.dtype == np.float32
    np.testing.assert_allclose(got, _area_resize(rgba[..., :3] / 255.0, dst[1], dst[0]), atol=1e-5)

    img = rng.random((*src, 3), dtype=np.float32)
    got = _core._gpu_resize_area(img, dst[1], dst[0])
    np.testing.assert_allclose(got, _area_resize(img, dst[1], dst[0]), atol=1e-5)


def _undistort_ref(img, k, d, rx, ry, out_w, out_h):
    """Forward-distort each output pixel of the crop, bilinear sample clamped to the border."""
    fx, fy, cx, cy = k
    k1, k2, p1, p2, k3 = d
    ys, xs = np.mgrid[0:out_h, 0:out_w].astype(np.float64)
    x, y = (xs + rx - cx) / fx, (ys + ry - cy) / fy
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    sx = (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)) * fx + cx
    sy = (y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y) * fy + cy
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    ax, ay = (sx - x0)[..., None], (sy - y0)[..., None]
    h, w = img.shape[:2]
    xa, xb = np.clip(x0, 0, w - 1), np.clip(x0 + 1, 0, w - 1)
    ya, yb = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    img = np.asarray(img, np.float64)
    top = img[ya, xa] * (1 - ax) + img[ya, xb] * ax
    bottom = img[yb, xa] * (1 - ax) + img[yb, xb] * ax
    return top * (1 - ay) + bottom * ay, sx, sy


def test_gpu_undistort_matches_numpy():
    from msplat import _core

    h, w = 90, 120
    ys, xs = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    img = np.stack([np.sin(6 * xs) * 0.5 + 0.5, np.cos(5 * ys) * 0.5 + 0.5, xs * ys], -1).astype(np.float32)
    k = [100.0, 102.0, 61.0, 44.0]
    d = [0.25, 0.05, 0.001, -0.002, 0.01]  # pincushion: the undistorted frame needs a crop
    got, rx, ry = _core._gpu_undistort(img, k, d)
    assert rx > 0 and ry > 0, "pincushion distortion must crop the border"
    oh, ow = got.shape[:2]
    assert rx + ow <= w and ry + oh <= h
    ref, sx, sy = _undistort_ref(img, k, d, rx, ry, ow, oh)
    np.testing.assert_allclose(got, ref, atol=1e-4)
    # alpha = 0 crop: every output pixel has a source pixel
    assert sx.min() > -1 and sx.max() < w and sy.min() > -1 and sy.max() < h


def _gauss_1d(n=11, sigma=1.5):
    x = np.arange(n) - n // 2
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def _blur_clamp(img):
    """Separable 11-tap Gaussian with clamp-to-edge borders."""
    g = _gauss_1d()
    r = len(g) // 2
    p = np.pad(img, ((0, 0), (r, r)), mode="edge")
    out = sum(g[k] * p[:, k:k + img.shape[1]] for k in range(len(g)))
    p = np.pad(out, ((r, r), (0, 0)), mode="edge")
    return sum(g[k] * p[k:k + img.shape[0], :] for k in range(len(g)))


def _metrics_ref(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = []
    for c in range(3):
        x, y = a[..., c], b[..., c]
        mx, my = _blur_clamp(x), _blur_clamp(y)
        sxx, syy = _blur_clamp(x * x) - mx * mx, _blur_clamp(y * y) - my * my
        sxy = _blur_clamp(x * y) - mx * my
        ssim.append((2 * mx * my + c1) * (2 * sxy + c2) / ((mx * mx + my * my + c1) * (sxx + syy + c2)))
    mse = ((a - b) ** 2).mean()
    return 10 * np.log10(1 / mse), float(np.mean(ssim)), float(np.abs(a - b).mean())


@pytest.mark.parametrize("shape", [(37, 53), (5, 40), (64, 64), (100, 17)])
def test_gpu_image_metrics_match_numpy(shape):
    from msplat import _core

    rng = np.random.default_rng(shape[0] + shape[1])
    a = rng.random((*shape, 3), dtype=np.float32)
    b = np.clip(a + rng.normal(0, 0.05, a.shape), 0, 1).astype(np.float32)
    psnr, ssim, l1 = _core._gpu_image_metrics(a, b)
    ref = _metrics_ref(a, b)
    assert psnr == pytest.approx(ref[0], rel=1e-5)
    assert ssim == pytest.approx(ref[1], abs=1e-5)
    assert l1 == pytest.approx(ref[2], abs=1e-6)


def test_gpu_pack_rgba8_matches_numpy():
    from msplat import _core

    img = np.random.default_rng(3).uniform(-0.2, 1.2, (37, 53, 3)).astype(np.float32)
    got = _core._gpu_pack_rgba8(img)
    assert got.shape == (37, 53, 4) and got.dtype == np.uint8
    ref = (np.clip(img, 0, 1) * np.float32(255)).astype(np.uint8)
    assert np.abs(got[..., :3].astype(int) - ref).max() <= 1
    assert (got[..., 3] == 255).all()


def _write_png_rgb(path, img):
    """Minimal 8-bit RGB PNG writer (stdlib only)."""
    import struct
    import zlib

    img = np.ascontiguousarray(img, dtype=np.uint8)
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw)))
        f.write(chunk(b"IEND", b""))


def _make_synthetic_dataset(root, n=4, w=67, h=45, meta=None, intrinsics=None, distortion=None):
    """Nerfstudio dataset of n random RGB PNGs (w x h). meta: (w, h) the cameras claim
    (default: the file size); intrinsics: (fx, fy, cx, cy) at the meta size."""
    import json

    os.makedirs(os.path.join(root, "images"), exist_ok=True)
    mw, mh = meta or (w, h)
    fx, fy, cx, cy = intrinsics or (0.8 * mw, 0.8 * mw, mw / 2, mh / 2)
    rng = np.random.default_rng(n * w + h)
    imgs, frames = [], []
    for i in range(n):
        img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        _write_png_rgb(os.path.join(root, "images", f"img_{i:02d}.png"), img)
        imgs.append(img)
        c2w = np.eye(4)
        c2w[:3, 3] = [0.1 * i, 0.0, 0.0]
        frames.append({"file_path": f"images/img_{i:02d}.png", "transform_matrix": c2w.tolist()})
    meta_json = {"w": mw, "h": mh, "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy, "frames": frames}
    if distortion:
        meta_json.update(dict(zip(["k1", "k2", "p1", "p2", "k3"], distortion)))
    with open(os.path.join(root, "transforms.json"), "w") as f:
        json.dump(meta_json, f)
    return imgs


def test_dataset_images_load_on_gpu(tmp_path):
    from msplat import Dataset

    imgs = _make_synthetic_dataset(str(tmp_path), n=5, w=67, h=45)
    full = Dataset(str(tmp_path), downscale_factor=1.0)
    half = Dataset(str(tmp_path), downscale_factor=2.0)
    assert full.num_train == 5
    for i in range(5):
        a = full.image(i)
        assert a.shape == (45, 67, 3) and a.dtype == np.float32
        ref = imgs[i] / 255.0
        # Decoded straight into GPU memory: same pixels, no flips or channel swaps
        # (correlation, so a color-managed decode cannot fail it spuriously)
        for c in range(3):
            assert np.corrcoef(a[..., c].ravel(), ref[..., c].ravel())[0, 1] > 0.98
        assert np.abs(a - ref).mean() < 0.05
        b = half.image(i)
        assert b.shape == (22, 33, 3)
        np.testing.assert_allclose(b, _area_resize(a, 33, 22), atol=1e-5)
