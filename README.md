![Banner](/src/img/Banner.png)

# Table of contents
1. [Overview](#overview)
2. [Repository Structure](#fstructure)
3. [Installation](#installation)
4. [Usage](#usage)

# 1. Overview <a name="overview"></a>
This repository provides a pipeline for segmentation of railway embankments from LiDAR point cloud data (.laz files). It covers the full workflow from raw data ingestion and rail geometry retrieval to embankment mask generation and parameter optimization.

The pipeline is built around the SegmentEmbankment class, which performs 2D raster-based region growing from rail centerlines outward onto surrounding terrain. Rail geometry is fetched automatically from a PostGIS database and used to seed the segmentation. The algorithm operates on ground and rail point classes only, ignoring vegetation, buildings, and other objects. Segmentation parameters are controlled through a JSON configuration file and can be tuned to match varying terrain conditions such as high embankments, shallow slopes, or dense urban surroundings.

# 2. Repository structure: <a name="fstructure"></a>

```
.
├── src
│   ├── utils
│   │   ├── pcd_tools.py                        #Point cloud preprocessing utilities: statistical outlier removal and vectorized voxel subsampling
│   │   └── plot_cloud.py                       #3D point cloud visualization
|   ├── embankment_config.json                  #Segmentation hyperparameters controlling rail detection radius, terrain slope thresholds, morphological
|   ├── db_params.txt                           #PostGIS connection parameters (dbname, host, port, user, password) for fetching rail geometry
│   └── Segment_embankment.py                   #Railway embankment segmentation from LiDAR point clouds using 2D raster-based region growing seeded from PostGIS rail geometry


```
---
# 3. Installation: <a name="installation"></a>

Clone the repository to your local machine:
```bash

git clone https://github.com/kalmary/Embankment_Segmentation.git

cd Embankment_segmentation
```

Create and activate a Virtual Environment and install requirements:
```bash
uv venv
source .venv/bin/activate

uv pip install -r requirements.txt
```

# 4. Usage <a name="usage"></a>

Script for segmentation of railway embankments from LiDAR point clouds. The `SegmentEmbankment` 
class performs 2D raster-based region growing seeded from rail geometry fetched automatically 
from a PostGIS database. To do so run:

```bash
python src/Segment_embankment.py
```

---

## Configuration

Segmentation behaviour is controlled through `embankment_config.json`. We recommend starting 
with the default values below, which were tuned on our dataset. If results are unsatisfactory 
for your data, refer to the parameter descriptions and adjust accordingly.

```json
{
    "voxel_size": 0.10,
    "rail_radius": 0.50,
    "grid_cell_size": 0.50,
    "max_dist_m": 10.0,
    "crown_width_m": 3.0,
    "min_slope": 0.05,
    "max_slope": 5.5,
    "min_global_slope": 0.05,
    "max_embankment_height": 6.0,
    "max_elev_diff": 0.20,
    "closing_radius": 1,
    "min_cluster_size": 50,
    "overlap": 10.0,
    "tile_size": 500.0,
    "ground_label": 2,
    "rail_label": 11
}
```

| Parameter | Description |
|---|---|
| `voxel_size` | Voxel size for input point cloud subsampling. Smaller values preserve more detail but increase processing time. |
| `rail_radius` | Points within this distance from the rail centerline are labelled as rail and used to seed the embankment growth. |
| `grid_cell_size` | Cell size of the 2D raster grid used for segmentation. Smaller values give finer boundaries but use more memory. |
| `max_dist_m` | Maximum growth radius from the rail. Embankment cannot extend beyond this distance regardless of terrain shape. |
| `crown_width_m` | Width of the flat track crown included unconditionally — slope checks are skipped in this zone. |
| `min_slope` | Minimum local terrain slope to qualify as an embankment slope. Increase to exclude flat surrounding ground. |
| `max_slope` | Maximum local terrain slope. Points steeper than this (walls, dense vegetation) are excluded. |
| `min_global_slope` | Minimum global descent rate from the rail. Prevents the mask from spreading onto flat terrain that is reachable through a slope. |
| `max_embankment_height` | Maximum depth below the rail level. Points deeper than this are excluded — limits downward growth into cuttings. |
| `max_elev_diff` | Maximum height above the rail level. Points higher than this are excluded. |
| `closing_radius` | Radius of the morphological closing operation applied to the mask. Larger values fill bigger gaps and smooth edges. |
| `min_cluster_size` | Connected components smaller than this are removed as noise after morphological refinement. |
| `overlap` | Tile overlap used in tiled processing mode. Prevents boundary artefacts between adjacent tiles. |
| `tile_size` | Tile size for large point clouds processed in tiled mode. |
| `ground_label` | LAS classification code for ground points in your input data. |
| `rail_label` | LAS classification code for rail points in your input data. |

