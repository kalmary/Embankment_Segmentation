# Ground Segmentation

## Overview

This repository contains a ground-profile segmentation workflow for railway LiDAR point clouds. The documented workflow uses `src/SegmentGround.py` as the segmentation entry point.

`GroundSegmenter` uses point coordinates, existing LAS classification labels, and rail geometry. It finds rail-adjacent points, builds a local rail centerline, cuts the point cloud into cross-sections, and classifies nearby terrain into ground, rail, embankment, and ditch labels.

The segmentation is configured with `src/ground_segm_config.json`.

## Repository Structure

```text
.
├── src
│   ├── SegmentGround.py              # Ground, embankment, and ditch segmentation
│   ├── ground_segm_config.json       # Ground segmentation parameters
│   └── utils
│       ├── pcd_tools.py              # Point cloud preprocessing helpers
│       ├── plot_cloud.py             # Point cloud visualization
│       └── plot_sections.py          # Section visualization helpers
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/kalmary/Embankment_Segmentation.git
cd Embankment_Segmentation

python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

```python
import laspy
import numpy as np

from src.SegmentGround import GroundSegmenter

las = laspy.read("input.laz")
points = np.column_stack((las.x, las.y, las.z))
labels = np.asarray(las.classification)

segmenter = GroundSegmenter.from_config(
    cfg_path="src/ground_segm_config.json",
    verbose=True,
)

result_labels = segmenter.segment(points, labels)
```

`segment()` returns a copy of the input labels with terrain classes updated. It only processes points whose labels match `ground_label` or `rail_label` from the config. Other input labels are preserved.

Default output labels:

| Label | Meaning |
|---|---|
| `1` | Ground |
| `0` | Rail |
| `10` | Embankment |
| `11` | Ditch |

## Configuration

Default `src/ground_segm_config.json`:

```json
{
  "distance_limit": 12.0,
  "ground_label": 1,
  "rail_label": 0,
  "rail_radius": 1.0,
  "embankment_label": 10,
  "ditch_label": 11,
  "length_min": 1.0,
  "length_max": 10.0,
  "width_margin": 0.0,
  "max_curve_ratio": 1.05,
  "curve_resolution": 0.25,
  "graph_x_bin": 0.25,
  "graph_uphill_slope": 0.15,
  "graph_min_uphill_points": 3,
  "graph_min_embankment_points": 8,
  "graph_noise_points": 2,
  "graph_smooth_window": 3,
  "graph_max_gap_bins": 1.0,
  "graph_ditch_min_downhill_points": 2,
  "graph_ditch_min_uphill_points": 3,
  "graph_ditch_immediate_points": 3,
  "graph_ditch_max_flat_points": 5,
  "smoothing": true,
  "smooth_level": 5
}
```

| Parameter | Description |
|---|---|
| `distance_limit` | Maximum lateral distance from the rail considered during nearest-point selection. |
| `ground_label` | Input and output label for ground points. |
| `rail_label` | Input and output label for rail points. |
| `rail_radius` | Distance from rail geometry used to detect rail-adjacent points. |
| `embankment_label` | Output label assigned to embankment points. |
| `ditch_label` | Output label assigned to ditch points. |
| `length_min` | Minimum centerline section length used for local cross-section analysis. |
| `length_max` | Maximum centerline section length used for local cross-section analysis. |
| `width_margin` | Extra cross-section width retained after rotating a local section. |
| `max_curve_ratio` | Maximum allowed arc-to-chord ratio before a section is shortened around a curve. |
| `curve_resolution` | Sampling resolution for rail centerline construction and curvature checks. |
| `graph_x_bin` | Bin size for building the X-Z terrain profile in each cross-section. |
| `graph_uphill_slope` | Minimum positive gradient treated as uphill terrain. |
| `graph_min_uphill_points` | Minimum consecutive uphill bins required for slope detection. |
| `graph_min_embankment_points` | Minimum bins required before accepting an embankment interval. |
| `graph_noise_points` | Number of noisy bins tolerated while splitting profile sections. |
| `graph_smooth_window` | Window size used to smooth the X-Z profile before gradient checks. |
| `graph_max_gap_bins` | Maximum gap allowed between related profile regions. |
| `graph_ditch_min_downhill_points` | Minimum consecutive downhill bins for ditch detection. |
| `graph_ditch_min_uphill_points` | Minimum consecutive uphill bins for ditch detection. |
| `graph_ditch_immediate_points` | Number of bins after embankment where an immediate ditch may begin. |
| `graph_ditch_max_flat_points` | Maximum flat/noisy bins allowed between ditch sides. |
| `smoothing` | Present in the default config. Boundary smoothing is enabled by default in `GroundSegmenter`; use `smooth` if this needs to be controlled from config. |
| `smooth_level` | Boundary smoothing level in meters along the centerline. |
