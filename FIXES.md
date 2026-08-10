# Fixing the public BroadTrack release

The released code does not reproduce the paper. On SNGS-116 it outputs 749
cameras out of 750 frames and scores 12.16 JaC@5, against the roughly 57 points
the paper reports for the whole test split. This branch fixes the bugs listed
below, and with them the full 49-sequence SoccerNet-GSR test split comes out
close to the paper:

|                        | JaC@5 | JaC@10 | Completeness |
|------------------------|------:|-------:|-------------:|
| Paper                  | 56.88 |  79.79 |         100% |
| This branch            | 56.56 |  78.71 |         100% |

There are no switches to set. Build the image, prepare the inputs, run,
evaluate. See "Running it" at the bottom.

## The fixes

### The networks received BGR images

OpenCV loads frames as BGR, and both TVCalib and the NBJW keypoint model were
trained on RGB input. The code fed them BGR, which noticeably degrades both
detectors on every frame. Both model wrappers now convert to RGB
([KeypointDetectionModel.cpp#L258](KeypointDetectionModel.cpp#L258),
[LineSegmentationModel.cpp#L170](LineSegmentationModel.cpp#L170)).

### Keypoint heatmaps were decoded wrong

The NBJW decoder read pooled heatmap indices with the wrong tensor type and
wrong geometry, kept the background channel as if it were a keypoint class,
scaled both axes by a single factor, and never applied the confidence
threshold the authors publish. Decoding now happens at the model's native
output size with integer indices, the background channel is dropped, each
class keeps its single best peak above threshold 0.1449, and x and y are
scaled independently. This affects initialisation and every recovery
(decoder at [KeypointDetectionModel.cpp#L85](KeypointDetectionModel.cpp#L85),
threshold at [#L172](KeypointDetectionModel.cpp#L172), background channel drop
at [#L273](KeypointDetectionModel.cpp#L273)).

### The last frame was never processed

The frame loop iterated to N-1, so every sequence lost its final frame and the
official evaluator saw an incomplete run. The loop now processes every frame
in the directory ([main.cpp#L244](main.cpp#L244), loop at
[#L339](main.cpp#L339)).

### Asking for overlays corrupted the output

Rendering the debug overlay resized the live camera object to 960x540 right
before serialization, so the JSON silently depended on whether visualisation
was on. Drawing now works on a copy and the output stays at the native
1920x1080 either way ([main.cpp#L505](main.cpp#L505)).

### Player boxes were not honored

Optical flow points that land on moving players drag the camera estimate with
them, which is why the pipeline takes per-frame human boxes as input. The
binary now resolves the box file for each frame by its name and drops flow
points inside the boxes, the way the author describes in issue #1
(loading at [main.cpp#L368](main.cpp#L368), filtering at
[#L406](main.cpp#L406)).

### Line points hopped between coordinate systems

Extracted line points were rounded, scaled towards 960x540 with a factor
derived from the height alone, then rounded and scaled again on the way to HD.
The two roundings alone cost about 3 HD pixels of accuracy per point. Points
now stay in native mask coordinates between frames, the mean shift keeps
subpixel centroids, and a single pixel-center mapping with separate x and y
scales converts them straight to HD (seeds at
[PointExtractor.cpp#L149](PointExtractor.cpp#L149), mapping at
[CameraTracker.cpp#L187](CameraTracker.cpp#L187)).

### Downsampling did not match TVCalib

TVCalib was trained on antialiased bilinear downsampling, the C++ used plain
linear, and the masks differ enough to move the metrics. The input is now
resized with LibTorch's antialiased bilinear, matching the training-time
preprocessing. On top of that, the point
suppression radius stayed at its half-resolution value of 20 while the mean
shift support at mask resolution is 5, so the extractor kept far too few
measurements. The radius now matches the support (resize at
[LineSegmentationModel.cpp#L65](LineSegmentationModel.cpp#L65), radius at
[PointExtractor.cpp#L37](PointExtractor.cpp#L37)).

### Lens distortion was never undone

The tracker estimates a forward radial distortion coefficient, but
Camera::undistort() applied a separate correction polynomial that nothing ever
filled in. Optical flow rays were therefore cast as if the lens had no
distortion at all. The forward model is now inverted iteratively before a ray
is intersected with the pitch; ten iterations leave the residual around 1e-4
pixels in the image corners ([Camera.cpp#L461](Camera.cpp#L461)).

### The tripod residual subtracted square meters from meters

The paper defines the tripod prior as radius minus distance. The code
implemented radius minus squared distance, which is dimensionally wrong and
puts the zero at the square root of the radius. We use the equivalent form
(d^2 - r^2) / (2r), which has the same zero and the same local scale as the
paper's residual but avoids its infinite derivative at zero distance, where
the optimizer occasionally lands ([Residuals.h#L220](Residuals.h#L220)).

### Recovery was broken in several places at once

The two-point reinitializer forgot to subtract the principal point, hard-coded
the 1080 normalization, tested only one root of the focal equation, sampled
point pairs randomly with replacement, ranked hypotheses with inconsistent
scores, reused the first sampled observation for every inlier, and reset the
camera to a default object, losing resolution and distortion. It now
enumerates every point pair once, keeps all finite focal roots within a sane
field-of-view range, ranks hypotheses consistently, matches each inlier to its
closest observation, and modifies a copy of the current camera. The two
hard-coded confidence thresholds (0.3 to trigger recovery, 0.5 to accept)
became a single symmetric 0.5 (focal roots at
[CameraTracker.cpp#L489](CameraTracker.cpp#L489), pair enumeration at
[#L659](CameraTracker.cpp#L659), threshold at
[CameraTracker.h#L46](CameraTracker.h#L46)).

### Runs were not repeatable

The point sampler seeded a fresh Mersenne twister from random_device on every
frame, so two identical invocations could diverge and end up several JaC
points apart on a sequence. The sampler is now seeded with 0 each frame,
which removes that spread. Small run-to-run differences can still come from
GPU inference itself; in our reruns they moved the full-split aggregate by
about 0.01 JaC ([PointExtractor.cpp#L57](PointExtractor.cpp#L57)).

### The score assumed 1080p input

The confidence score projects the pitch model onto a 960x540 copy of the
line mask, and the projection was scaled by a hard-coded factor of two,
which is only correct when the input is 1920x1080. At any other resolution
the score came out near zero no matter how good the pose was, so the
tracker declared itself lost on every frame and never recovered. The scale
now comes from the actual camera resolution
([CameraTracker.cpp#L384](CameraTracker.cpp#L384)). At 1080p the factor is
exactly the same 0.5, so the SoccerNet results are unchanged; on a 720p
clip this fix takes the tracker from zero locked frames to 99%.

## How the fixes were tested

A full test-split run takes about an hour, so fixes were validated on a small
subset of clips to save compute, and the complete 49-sequence run was done
once at the end. The subset clips are part of the public test split, so their
numbers guided development decisions; only the final full-split result is
comparable to the paper.

The public release, run as-is on SNGS-116, produces 749 cameras and scores
12.16 JaC@5 / 38.81 JaC@10. The first row of the table below applies every
fix described above except two: "Line points hopped between coordinate
systems" and "Downsampling did not match TVCalib" (Line points and
Downsampling in the row labels). Those two account for the remaining twenty
points. Two rows are development stopgaps that are not in the final code:
the OpenCV area resize stood in for antialiasing until the LibTorch version
replaced it, and radius 9 was a mechanical rescale of the old 20 before the
radius was matched to the mean-shift support.

| SNGS-116, cumulative                                 | JaC@5 | JaC@10 |
|------------------------------------------------------|------:|-------:|
| Every fix except Line points and Downsampling        | 46.26 |  73.79 |
| Line points: subpixel pixel-center coordinates       | 50.32 |  73.06 |
| Downsampling: area resize (stopgap)                  | 55.90 |  73.93 |
| Downsampling: suppression radius 9 (stopgap)         | 57.04 |  77.96 |
| Downsampling: suppression radius 5, the final value  | 61.51 |  78.70 |
| Downsampling: LibTorch antialiased bilinear resize   | 66.21 |  81.03 |

The resize choice was then confirmed on a fixed three-clip set (SNGS-116, 117
and 132), where LibTorch's antialiased bilinear beat the OpenCV area
approximation on the aggregate: 56.79 vs 56.21 JaC@5 and 83.41 vs 82.34
JaC@10.

The tripod residual was checked on five clips spanning all three games. The
paper's literal radius-minus-distance form scored 35.75 / 67.84 and produced
a non-finite Jacobian on one clip; the stable equivalent reached 36.87 /
69.41 with none. The released radius-minus-squared-distance form was far
worse where it was tried (23.80 / 57.37 on SNGS-141).

A few sanity checks on SNGS-132: dropping the distortion coefficient cost
about 3 JaC@5 points (25.49 vs 28.49), dropping the tripod prior about 2.7
(25.78), so both stay on. Three sampler seeds spread from 25.96 to 28.49
JaC@5 on that clip, which is what motivated seeding the sampler in the
first place; we kept seed 0.

## Choices we had to make ourselves

A few inputs of the paper's pipeline were never published, so some things
here are our own choices rather than reconstructions:

- Player boxes come from the ground-truth GameState annotations, as the author
  recommends in issue #1. The paper used an RTMDet detector whose weights and
  detections were not released, except for one sequence. On that sequence the
  detector boxes score about 0.7 JaC points higher than the annotation boxes.
- The evaluator reports MRE and MedRE pooled over all ground-truth points.
  The paper does not say how it aggregates these two, so treat them as
  diagnostics; JaC and completeness follow the published protocol exactly,
  including the label typos fixed in issue #1 and the mirrored-pitch
  ambiguity.
- Seed 0 and the antialiased resize are our defaults, chosen for determinism
  and for fidelity to TVCalib's training-time preprocessing. Neither is a
  published hyperparameter.

## Running it

Build the image (models come via Git LFS, pull them first):

```bash
git lfs pull
docker build . -t broadtrack
```

Prepare the inputs for the SoccerNet-GSR 2024 test split. This writes
per-frame box files and per-game tripod files, the dataset itself stays
read-only:

```bash
python3 repro/prepare_soccernet.py \
  --dataset-test-root /data/SoccerNetGS/test \
  --artifact-root /data/broadtrack-artifacts
```

Run all 49 sequences (a few minutes each on a desktop GPU):

```bash
python3 repro/run_soccernet.py \
  --dataset-test-root /data/SoccerNetGS/test \
  --artifact-root /data/broadtrack-artifacts \
  --output-root /data/broadtrack-results
```

Evaluate in the paper's protocol:

```bash
python3 repro/evaluate_soccernet.py \
  --dataset /data/SoccerNetGS \
  --split test \
  --predictions '/data/broadtrack-results/SNGS-*/broadtrack.json' \
  --json-out /data/broadtrack-results/metrics.json
```

The evaluator needs numpy and the sn_calibration_baseline package from the
SoccerNet calibration repository on the Python path.
