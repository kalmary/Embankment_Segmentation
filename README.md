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
  "distance_limit": 25.0,
  "ground_label": 1,
  "rail_label": 0,
  "rail_radius": 1.0,
  "embankment_label": 10,
  "ditch_label": 11,

  "length_min": 2.0,
  "length_max": 10.0,
  "width_margin": 0.0,
  "max_curve_ratio": 1.1,
  "curve_resolution": 1.5,

  "graph_x_bin": 0.25,
  "graph_uphill_slope": 0.1,

  "graph_embankment_min_stop_m": 0.6,
  "graph_min_embankment_m": 1.6,

  "graph_noise_points": 2,
  "graph_smooth_window": 5,
  "graph_max_gap_bins": 1.0,

  "graph_ditch_min_downhill_m": 0.4,
  "graph_ditch_min_uphill_m": 0.4,
  "graph_ditch_immediate_points_m": 0.7,
  "graph_ditch_max_flat_m": 1.0,
  "graph_ditch_max_uphill_m": 2.0,

  "graph_ditch_search_min_m": 6.0,
  "graph_ditch_search_max_m": 16.0,

  "smooth": true,
  "smooth_level": 20.0
}
```

| Parameter | Description |
|---|---|
| `distance_limit` | Maximum distance from a rail point used when selecting terrain points, in metres. |
| `ground_label` | Input and output label for ground points. |
| `rail_label` | Input and output label for rail points. |
| `rail_radius` | Distance from rail geometry used to detect rail-adjacent points. |
| `embankment_label` | Output label assigned to embankment points. |
| `ditch_label` | Output label assigned to ditch points. |
| `length_min` | Minimum centerline section length, in metres. |
| `length_max` | Maximum centerline section length, in metres. |
| `width_margin` | Extra cross-section width retained after rotating a local section. |
| `max_curve_ratio` | Maximum allowed arc-to-chord ratio before a section is shortened around a curve. |
| `curve_resolution` | Centerline and curvature sampling interval, in metres. |
| `graph_x_bin` | X-Z profile bin width, in metres. |
| `graph_uphill_slope` | Minimum positive gradient treated as uphill terrain. |
| `graph_embankment_min_stop_m` | Flat or uphill run that ends embankment detection, in metres. |
| `graph_min_embankment_m` | Minimum embankment length before its stop condition is considered, in metres. |
| `graph_noise_points` | Number of noisy bins tolerated while splitting profile sections. |
| `graph_smooth_window` | Window size used to smooth the X-Z profile before gradient checks. |
| `graph_max_gap_bins` | Retained legacy setting; rail-based graph splitting does not use it. |
| `graph_ditch_min_downhill_m` | Minimum downhill run for a ditch, in metres. |
| `graph_ditch_min_uphill_m` | Minimum uphill run for a ditch, in metres. |
| `graph_ditch_immediate_points_m` | Maximum distance from embankment to an immediate ditch uphill, in metres. |
| `graph_ditch_max_flat_m` | Maximum flat bottom width between ditch sides, in metres. |
| `graph_ditch_max_uphill_m` | Maximum retained uphill wall length for a ditch, in metres. |
| `graph_ditch_search_min_m` | Ditch search start, measured outward from the rail, in metres. |
| `graph_ditch_search_max_m` | Ditch search end, measured outward from the rail, in metres. |
| `smooth` | Enables boundary smoothing. |
| `smooth_level` | Gaussian smoothing distance along the centerline, in metres. |
