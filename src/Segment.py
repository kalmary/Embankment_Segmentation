import laspy
import psycopg2
import numpy as np
import pathlib as pth
import open3d as o3d
import scipy.ndimage as ndi
 
from tqdm import tqdm
from scipy.spatial import cKDTree
from utils import plot_cloud
from shapely import wkt as shapely_wkt
from shapely.geometry import LineString, MultiLineString
 
# --------------------------------------------------
# POINT CLOUD PREPROCESSING
# --------------------------------------------------
def remove_outliers(points: np.ndarray, nb_neighbors:int=40, std_ratio:float=2.):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=2.0)
    return ind
 
def voxel_subsample_vectorized(xyz, voxel_size=0.10):
    keys     = np.floor(xyz / voxel_size).astype(np.int32)
    centers  = (keys + 0.5) * voxel_size
    dists_sq = np.sum((xyz - centers) ** 2, axis=1)
 
    keys_min  = keys.min(axis=0)
    keys      = keys - keys_min
    key_range = keys.max(axis=0) + 1
 
    rx, ry, rz = int(key_range[0]), int(key_range[1]), int(key_range[2])
    assert (ry * rz * rx) < np.iinfo(np.int64).max, "key encoding overflow"
 
    key_enc = (keys[:, 0].astype(np.int64) * ry * rz +
               keys[:, 1].astype(np.int64) * rz +
               keys[:, 2].astype(np.int64))
    
    order      = np.lexsort((dists_sq, key_enc))
    key_sorted = key_enc[order]
    _, first   = np.unique(key_sorted, return_index=True)
    chosen     = order[first]
 
    return chosen
# --------------------------------------------------
# I/O
# --------------------------------------------------
def load_data(las_path):
    las = laspy.read(las_path)
    xyz = np.stack([
        np.asarray(las.x, dtype=np.float64),
        np.asarray(las.y, dtype=np.float64),
        np.asarray(las.z, dtype=np.float64),
    ], axis=1)
 
    labels    = np.array(las.classification, dtype=np.int32)
 
    ground_embankment_mask = (labels == 2) #| (labels == 2) #| (labels == 1)
    xyz = xyz[ground_embankment_mask]
    labels = labels[ground_embankment_mask]

    del las
    return xyz, labels
 
# --------------------------------------------------
# DB PARAMS
# --------------------------------------------------
def load_db_params(path):
    params = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            k, v = line.split("=", 1)
            params[k.strip()] = v.strip()
    return params
 
# --------------------------------------------------
# LOAD RAILS FROM POSTGIS
# --------------------------------------------------
def load_tracks_from_db(db_params, bbox):
    conn = psycopg2.connect(**db_params)
    xmin, ymin, xmax, ymax = bbox
    query = """
        SELECT ST_AsText(wspolrzedne_utm)
        FROM data.tory
        WHERE ST_Intersects(
            wspolrzedne_utm,
            ST_MakeEnvelope(%s, %s, %s, %s, ST_SRID(wspolrzedne_utm))
        );
    """
    cur = conn.cursor()
    cur.execute(query, (xmin, ymin, xmax, ymax))
    rows = cur.fetchall()
    conn.close()
    lines = []
    for (wkt_line,) in rows:
        geom = shapely_wkt.loads(wkt_line)
        if isinstance(geom, LineString):
            lines.append(geom)
        elif isinstance(geom, MultiLineString):
            lines.extend(list(geom.geoms))
    return lines
 
# --------------------------------------------------
# DENSIFY RAIL LINES
# --------------------------------------------------
def densify_lines(lines, step=0.5):
    pts = []
    for line in lines:
        length = line.length
        distances = np.arange(0, length + step, step)
        for d in distances:
            p = line.interpolate(d)
            pts.append((p.x, p.y))
    return np.array(pts)
 
# --------------------------------------------------
# Rail labelling
# --------------------------------------------------
def label_rail_points(xyz, db_param_path, rail_radius=0.5):
    db_params = load_db_params(pth.Path(db_param_path))
    xmin = float(xyz[:,0].min())
    xmax = float(xyz[:,0].max())
    ymin = float(xyz[:,1].min())
    ymax = float(xyz[:,1].max())
    bbox = (xmin, ymin, xmax, ymax)
    rails = load_tracks_from_db(db_params, bbox)
    if len(rails) == 0:
        return np.zeros(xyz.shape[0], dtype=np.uint8)
    rail_xy = densify_lines(rails, step=0.5)
    tree = cKDTree(rail_xy)
    dist, _ = tree.query(xyz[:, :2])
    labels = (dist <= rail_radius).astype(np.uint8)

    if np.unique(labels).shape[0]<2:
        raise ValueError("No rails detected.")

    return labels
 
# --------------------------------------------------
# Embankment growing function
# --------------------------------------------------
def grow_embankment_connected(xyz,
                            track_labels,
                            grid_cell_size=0.5,
                            max_dist_m=12.0,
                            crown_width_m=3.0,
                            min_slope=0.15,
                            max_slope=1.5):
    """
    Grow embankment mask outward from track points using iterative binary dilation on a 2D grid.
 
    Parameters
    ----------
    xyz : np.ndarray, shape (N, 3)
        Point cloud coordinates (X, Y, Z) in metres.
    track_labels : np.ndarray, shape (N,), dtype uint8
        Binary track labels (1 = rail, 0 = other). Output of `label_rail_points`.
    grid_cell_size : float, optional
        Raster cell size in metres. Default: 0.5.
    max_dist_m : float, optional
        Maximum embankment growth radius from track edge in metres. Default: 12.0.
    crown_width_m : float, optional
        Width of the flat track crown in metres. Slope check is skipped here. Default: 3.0.
    min_slope : float, optional
        Minimum terrain slope [m/m] to qualify as embankment slope. Default: 0.15.
    max_slope : float, optional
        Maximum terrain slope [m/m] to qualify as embankment slope. Default: 1.5.
 
    Returns
    -------
    np.ndarray, shape (N,), dtype uint8
        Binary embankment labels (1 = embankment, 0 = other).
    """
    print("Growing embankment mask...")
    
    new_final = np.zeros_like(track_labels, dtype=np.uint8)
    mask = (track_labels == 1)
    
    if not mask.sum():
        return new_final
        
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    
    x_min, y_min = x.min(), y.min()
    nx = int(np.ceil((x.max() - x_min) / grid_cell_size)) + 1
    ny = int(np.ceil((y.max() - y_min) / grid_cell_size)) + 1
    ix = np.clip(((x - x_min) / grid_cell_size).astype(np.int32), 0, nx - 1)
    iy = np.clip(((y - y_min) / grid_cell_size).astype(np.int32), 0, ny - 1)
 
    z_grid = np.full((ny, nx), np.nan, dtype=np.float32)
    order = np.argsort(z)
    z_grid[iy[order], ix[order]] = z[order]
 
    invalid_mask = np.isnan(z_grid)
    if invalid_mask.all(): return new_final
        
    _, indices = ndi.distance_transform_edt(invalid_mask, return_distances=True, return_indices=True)
    z_filled = z_grid[tuple(indices)]
    z_smoothed = ndi.gaussian_filter(z_filled, sigma=1.0)
 
    gy, gx = np.gradient(z_smoothed, grid_cell_size)
    grad = np.sqrt(gx**2 + gy**2)
    
    gm_tracks = np.zeros((ny, nx), dtype=bool)
    gm_tracks[iy[mask], ix[mask]] = True
 
    dist_from_track, nearest_track_idx = ndi.distance_transform_edt(~gm_tracks, return_distances=True, return_indices=True)
    dist_from_track *= grid_cell_size
 
    z_tracks_only = np.full((ny, nx), np.nan, dtype=np.float32)
    z_tracks_only[gm_tracks] = z_smoothed[gm_tracks]
 
    z_nearest_track = z_tracks_only[tuple(nearest_track_idx)]
 
    valid_elevation = z_smoothed <= (z_nearest_track + 0.5)
 
    valid_slope = (grad >= min_slope) & (grad <= max_slope)
    
    valid_growth = (dist_from_track <= crown_width_m) | (valid_slope & valid_elevation)
    
    valid_growth &= (dist_from_track <= max_dist_m)
 
    struct = ndi.generate_binary_structure(2, 2)
    expanded = gm_tracks.copy()
    max_iters = int(max_dist_m / grid_cell_size) + 5
    
    for i in range(max_iters):
        new_expanded = (ndi.binary_dilation(expanded, struct) & valid_growth) | gm_tracks
        if np.array_equal(expanded, new_expanded): break
        expanded = new_expanded
 
    final_expanded_pts = expanded[iy, ix]
    new_final[final_expanded_pts] = 1
    
    return new_final
 
# --------------------------------------------------
# Iter_titles
# --------------------------------------------------
def iter_tiles(xyz, tile_size=40.0, overlap=10.0, min_points=1024):
    mins     = xyz[:, :2].min(0)
    maxs     = xyz[:, :2].max(0)
    x_starts = np.arange(mins[0], maxs[0], tile_size)
    y_starts = np.arange(mins[1], maxs[1], tile_size)
 
    xy_starts = [(x0, y0) for x0 in x_starts for y0 in y_starts]
    pbar = tqdm(xy_starts, total=len(xy_starts), desc="Tiling", unit="cell", leave=False)
    
    for (x0, y0) in pbar:
        mask = (
            (xyz[:, 0] >= x0 - overlap) & (xyz[:, 0] < x0 + tile_size + overlap) &
            (xyz[:, 1] >= y0 - overlap) & (xyz[:, 1] < y0 + tile_size + overlap)
        )
        
        if mask.sum() < min_points:
            continue
 
        yield mask
 
# --------------------------------------------------
# Mask refinement
# --------------------------------------------------
def refine_mask_2d(xyz, mask, grid_cell_size=0.5, closing_radius=2, min_cluster_size=10):
    """
    Refine a binary point mask using 2D morphological operations.
    Projects points onto a raster grid, applies binary closing to bridge
    small gaps, fills interior holes, and removes small noise clusters.
 
    Parameters
    ----------
    xyz : np.ndarray, shape (N, 3)
        Point cloud coordinates (X, Y, Z) in metres.
    mask : np.ndarray, shape (N,), dtype bool
        Input binary mask to refine.
    grid_cell_size : float, optional
        Raster cell size in metres. Default: 0.5.
    closing_radius : int, optional
        Radius of the circular structuring element used for binary closing
        (in grid cells). Default: 2.
    min_cluster_size : int, optional
        Connected components smaller than this (in pixels) are removed as noise.
        Default: 10.
 
    Returns
    -------
    np.ndarray, shape (N,), dtype bool
        Refined binary mask mapped back onto the original points.
    """
 
    print(f"    Refining mask (closing radius: {closing_radius})...")
    
    x, y = xyz[:, 0], xyz[:, 1]
    x_min, y_min = x.min(), y.min()
    nx = int(np.ceil((x.max() - x_min) / grid_cell_size)) + 1
    ny = int(np.ceil((y.max() - y_min) / grid_cell_size)) + 1
    
    ix = np.clip(((x - x_min) / grid_cell_size).astype(np.int32), 0, nx - 1)
    iy = np.clip(((y - y_min) / grid_cell_size).astype(np.int32), 0, ny - 1)
 
    gm = np.zeros((ny, nx), dtype=bool)
    gm[iy[mask], ix[mask]] = True
 
    y_idx, x_idx = np.ogrid[-closing_radius:closing_radius+1, -closing_radius:closing_radius+1]
    struct = x_idx**2 + y_idx**2 <= closing_radius**2
 
    refined_gm = ndi.binary_closing(gm, structure=struct)
 
    refined_gm = ndi.binary_fill_holes(refined_gm)
 
    label_im, nb_labels = ndi.label(refined_gm)
    sizes = ndi.sum(refined_gm, label_im, range(nb_labels + 1))
    mask_size = sizes < min_cluster_size
    remove_pixel = mask_size[label_im]
    refined_gm[remove_pixel] = False
 
    refined_mask_pts = refined_gm[iy, ix]
    
    added = refined_mask_pts.sum() - mask.sum()
    print(f"      Refinement complete. Change: {added:,} points.")
    
    return refined_mask_pts
 
# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------
def SegmentEmbankment(
    xyz: np.ndarray,          
    db_param_path: str,
    ground_label:   int,
    rails_label:    int,      
    voxel_size:     float = 0.10,
    rail_radius:    float = 0.50,
    grid_cell_size: float = 0.50,
    max_dist_m:     float = 15.0,
    crown_width_m:  float = 3.0,
    min_slope:      float = 0.05,
    max_slope:      float = 5.5,
    closing_radius: int   = 2,
    min_cluster_size: int = 5,
    tile_size:      float = 300.0,  
    overlap:        float = 10.0,
    verbose:        bool = False
) -> np.ndarray:              # (N,) uint8 — 0=grunt, 1=tory, 2=nasyp
    """
    Segment a point cloud into ground (0), rails (1), and embankment (2).
 
    Runs the full pipeline: voxel subsampling → rail labelling from PostGIS
    → embankment growing → mask refinement → outlier removal.
 
    Parameters
    ----------
    xyz           : raw point cloud (labels 1, 2, 11)
    db_param_path : plik z parametrami połączenia do PostGIS
 
    Returns
    -------
    xyz : np.ndarray, shape (M, 3)
        Cleaned point cloud after outlier removal (M ≤ N).
    vis_labels : np.ndarray, shape (M,), dtype uint8
        Per-point class labels: 0 = ground, 1 = rail, 2 = embankment.
    """
 
    # Voxel subsampling
    idx = voxel_subsample_vectorized(xyz, voxel_size=voxel_size)
    xyz = xyz[idx]
 
    # Rail labelling from PostGIS
    print("Rail labelling from PostGIS ...")
    track_labels = label_rail_points(xyz, db_param_path, rail_radius=rail_radius)
 
    processed_xyz = []
    processed_labels = []
 
    for mask in iter_tiles(xyz, tile_size=tile_size, overlap=overlap):
        chunk_xyz = xyz[mask]
        chunk_track_labels = track_labels[mask]
 
        if chunk_track_labels.sum() == 0:
            continue

        # Embankment growing for each chunk
        chunk_embankment_labels = grow_embankment_connected(
            chunk_xyz, chunk_track_labels,
            grid_cell_size=grid_cell_size,
            max_dist_m=max_dist_m,
            crown_width_m=crown_width_m,
            min_slope=min_slope,
            max_slope=max_slope,
        )
 
        # 5. Mask refinement for each chunk
        mask_to_fix = chunk_embankment_labels == 1
        final_mask  = refine_mask_2d(
            chunk_xyz, mask_to_fix,
            grid_cell_size=grid_cell_size,
            closing_radius=closing_radius,
            min_cluster_size=min_cluster_size,
        )
        chunk_embankment_labels = np.zeros_like(chunk_embankment_labels)
        chunk_embankment_labels[final_mask] = 1
 
        # 6. Outlier removal for each chunk
        idx_clean = remove_outliers(chunk_xyz.astype(np.float32))
        chunk_xyz = chunk_xyz[idx_clean]
        chunk_embankment_labels = chunk_embankment_labels[idx_clean]
        chunk_track_labels = chunk_track_labels[idx_clean]
 
        # 7. Mapping (0 - ground, 1 - rail, 2 - embankment)
        vis_labels = np.zeros(len(chunk_xyz), dtype=np.uint8)
        vis_labels[chunk_track_labels == 1]      = 1
        vis_labels[chunk_embankment_labels == 1] = 2
 
        processed_xyz.append(chunk_xyz)
        processed_labels.append(vis_labels)
 
    if not processed_xyz:
        return np.empty((0, 3)), np.empty((0,), dtype=np.uint8)
 
    # Merging chunks together
    final_xyz = np.vstack(processed_xyz)
    final_vis_labels = np.concatenate(processed_labels)
 
    # Cleaning overlapped fragments
    _, unique_idx = np.unique(final_xyz, axis=0, return_index=True)
    final_xyz = final_xyz[unique_idx]
    final_vis_labels = final_vis_labels[unique_idx]
 
    return final_xyz, final_vis_labels
# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    path      = pth.Path("/mnt/DATA_SSD/BRIK/Testing")
    db_params = "db_params.txt"
 
    for i, laz_path in enumerate(path.glob("*.las")):
        print(f"\n[{i+1}] {laz_path.name}")
 
        xyz_raw, _ = load_data(laz_path)
 
        xyz, vis_labels = SegmentEmbankment(xyz_raw, db_params)
 
        xyz_vis = xyz.copy()
        xyz_vis[:, :2] -= xyz_vis[:, :2].mean(axis=0)
        xyz_vis[:, 2]  -= xyz_vis[:, 2].min()
        xyz_vis = xyz_vis.astype(np.float32)
        plot_cloud(xyz_vis, vis_labels)
 
if __name__ == "__main__":
    main()