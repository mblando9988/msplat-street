# Street view: Apple Look Around → Gaussian splat

Two Look Around panoramas captured 5.5 m apart on a highway under an arched truss
footbridge, turned into a Gaussian splat with msplat's prior-guided street preset.
Street captures are what plain 3DGS handles worst: every camera looks the same way,
so depth along the road is barely constrained (streaks and needles), the sky has no
geometry (black or smeared sky), and other cars move between captures. This example
runs the whole pipeline on a CPU (Linux works), no Apple GPU needed.

## Pipeline

1. **Faces → pinhole views** (`lookaround.py`). Each side face of a panorama is an
   equirectangular segment of the sphere (see the module docstring for the mapping).
   Ten pinhole views per panorama (90° horizontal field of view, yaw −75°…75°, level
   and tilted up 35°) are resampled from the front, right and left faces with
   feathered seams, rendered at 2× and box-filtered. Panoramas are placed from the
   metadata's east/north offsets and elevation; the heading orients them.
2. **Priors** (`prepare.py`). Depth Anything V2 (metric, outdoor) gives depth in
   metres, the same unit as the camera positions. SegFormer (ADE20K) gives sky masks
   and masks for cars, trucks and people, which are dropped from the photometric loss
   because they move between captures. Pixels the panorama does not cover are masked
   too. Confidence falls off with distance and at depth edges.
3. **Initialization**. All views' depth is back-projected and merged in voxels that
   grow with distance, giving the initial point cloud.
4. **Training** (`train.py`). `msplat.street_config()`: log-depth Huber loss to the
   priors, a learned sky behind the gaussians with sky pixels pushed transparent,
   per-image exposure compensation, masks, and a cap on needle-shaped gaussians.
5. **Photos**: training views next to the photos with rendered depth, novel views
   between and around the capture points, and a drive-through animation.

Every size is stated on the way: `prepare.py` logs each face's metadata size, file
size and pixels per degree, and writes them to `sizing.json` next to the dataset.

## Run

Put the face images in `data/faces/` (`a_front.webp`, `a_right.webp`, `a_left.webp`,
and the same for `b_`) and the two `*-meta.json` files in `data/meta/`, then:

```bash
pip install torch transformers pillow numpy   # depth and segmentation models (CPU is fine)
python prepare.py --out data/street
python train.py --data data/street --out runs/street --iterations 3000
```

`prepare.py --points-only --voxel 0.03` rebuilds just the initial point cloud from
the saved priors (no model inference).

The Look Around imagery is Apple's; it is not included in this repository.
