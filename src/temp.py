import json
import laspy
import psycopg2
import numpy as np
import pathlib as pth
import open3d as o3d
import scipy.ndimage as ndi

from tqdm import tqdm
from scipy.spatial import cKDTree
from plot_cloud import plot_cloud
from shapely import wkt as shapely_wkt
from shapely.geometry import LineString, MultiLineString



class EmbankmentSegmenter:
    def __init__(self, db_param_path: str, config_path: str, verbose: bool):
        self.db_param_path = db_param_path
        self.config_path = config_path
        self.verbose = verbose
        self._load_config(self.config_path)
    def _load_config(self, config_path: str):
        with open(config_path, "r") as f:
            cfg = json.load(f)

        self.voxel_size       = cfg["voxel_size"]
        self.rail_radius      = cfg["rail_radius"]
        self.grid_cell_size   = cfg["grid_cell_size"]
        self.max_dist_m       = cfg["max_dist_m"]
        self.crown_width_m    = cfg["crown_width_m"]
        self.min_slope        = cfg["min_slope"]
        self.max_slope        = cfg["max_slope"]
        self.closing_radius   = cfg["closing_radius"]
        self.min_cluster_size = cfg["min_cluster_size"]
        self.tile_size        = cfg["tile_size"]
        self.overlap          = cfg["overlap"]

    def segment_embankment(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Segment a point cloud into ground (0), rails (1), and embankment (2).
        Runs the full pipeline.
        """
        # 1. Voxel subsampling
        idx = self._voxel_subsample_vectorized(xyz)
        xyz = xyz[idx]

        # 2. Rail labelling from PostGIS
        if self.verbose:
            print("Rail labelling from PostGIS ...")
        track_labels = self._label_rail_points(xyz)

        processed_xyz = []
        processed_labels = []

        # 3. Podział na kafelki
        for mask in self._iter_tiles(xyz):
            chunk_xyz = xyz[mask]
            chunk_track_labels = track_labels[mask]

            if chunk_track_labels.sum() == 0:
                continue

            # 4. Embankment growing
            chunk_embankment_labels = self._grow_embankment_connected(chunk_xyz, chunk_track_labels)

            # 5. Mask refinement
            mask_to_fix = chunk_embankment_labels == 1
            final_mask  = self._refine_mask_2d(chunk_xyz, mask_to_fix)

            chunk_embankment_labels = np.zeros_like(chunk_embankment_labels)
            chunk_embankment_labels[final_mask] = 1

            # 6. Outlier removal
            idx_clean = self._remove_outliers(chunk_xyz.astype(np.float32))
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

        # 8. Łączenie kawałków w całość
        final_xyz = np.vstack(processed_xyz)
        final_vis_labels = np.concatenate(processed_labels)

        # 9. Czyszczenie nakładających się fragmentów
        _, unique_idx = np.unique(final_xyz, axis=0, return_index=True)
        final_xyz = final_xyz[unique_idx]
        final_vis_labels = final_vis_labels[unique_idx]

        return final_xyz, final_vis_labels
    # --------------------------------------------------
    # POINT CLOUD PREPROCESSING
    # --------------------------------------------------
    def _remove_outliers(self, points: np.ndarray, nb_neighbors:int=40, std_ratio:float=2.):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=2.0)
        return ind

    def _voxel_subsample_vectorized(self, xyz):
        keys     = np.floor(xyz / self.voxel_size).astype(np.int32)
        centers  = (keys + 0.5) * self.voxel_size
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
    # DB PARAMS
    # --------------------------------------------------
    def _load_db_params(self, path):
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
    def _load_tracks_from_db(self, db_params, bbox):
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
    def _densify_lines(self, lines, step=0.5):
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
    def _label_rail_points(self, xyz, rail_radius=0.5):
        db_params = self._load_db_params(pth.Path(self.db_param_path))
        xmin = float(xyz[:,0].min())
        xmax = float(xyz[:,0].max())
        ymin = float(xyz[:,1].min())
        ymax = float(xyz[:,1].max())
        bbox = (xmin, ymin, xmax, ymax)
        rails = self._load_tracks_from_db(db_params, bbox)
        if len(rails) == 0:
            return np.zeros(xyz.shape[0], dtype=np.uint8)
        rail_xy = self._densify_lines(rails, step=0.5)
        tree = cKDTree(rail_xy)
        dist, _ = tree.query(xyz[:, :2])
        labels = (dist <= rail_radius).astype(np.uint8)
        return labels

    # --------------------------------------------------
    # Embankment growing function
    # --------------------------------------------------
    def _grow_embankment_connected(self,
                                    xyz,
                                    track_labels):
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
        if self.verbose:
            print("Growing embankment mask...")

        new_final = np.zeros_like(track_labels, dtype=np.uint8)
        mask = (track_labels == 1)

        if not mask.sum():
            return new_final

        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        x_min, y_min = x.min(), y.min()
        nx = int(np.ceil((x.max() - x_min) / self.grid_cell_size)) + 1
        ny = int(np.ceil((y.max() - y_min) / self.grid_cell_size)) + 1
        ix = np.clip(((x - x_min) / self.grid_cell_size).astype(np.int32), 0, nx - 1)
        iy = np.clip(((y - y_min) / self.grid_cell_size).astype(np.int32), 0, ny - 1)

        z_grid = np.full((ny, nx), np.nan, dtype=np.float32)
        order = np.argsort(z)
        z_grid[iy[order], ix[order]] = z[order]

        invalid_mask = np.isnan(z_grid)
        if invalid_mask.all(): return new_final

        _, indices = ndi.distance_transform_edt(invalid_mask, return_distances=True, return_indices=True)
        z_filled = z_grid[tuple(indices)]
        z_smoothed = ndi.gaussian_filter(z_filled, sigma=1.0)

        gy, gx = np.gradient(z_smoothed, self.grid_cell_size)
        grad = np.sqrt(gx**2 + gy**2)

        gm_tracks = np.zeros((ny, nx), dtype=bool)
        gm_tracks[iy[mask], ix[mask]] = True

        dist_from_track, nearest_track_idx = ndi.distance_transform_edt(~gm_tracks, return_distances=True, return_indices=True)
        dist_from_track *= self.grid_cell_size

        z_tracks_only = np.full((ny, nx), np.nan, dtype=np.float32)
        z_tracks_only[gm_tracks] = z_smoothed[gm_tracks]

        z_nearest_track = z_tracks_only[tuple(nearest_track_idx)]

        valid_elevation = z_smoothed <= (z_nearest_track + 0.5)

        valid_slope = (grad >= self.min_slope) & (grad <= self.max_slope)

        valid_growth = (dist_from_track <= self.crown_width_m) | (valid_slope & valid_elevation)

        valid_growth &= (dist_from_track <= self.max_dist_m)

        struct = ndi.generate_binary_structure(2, 2)
        expanded = gm_tracks.copy()
        max_iters = int(self.max_dist_m / self.grid_cell_size) + 5

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
    def _iter_tiles(self, xyz, min_points=1024):
        mins     = xyz[:, :2].min(0)
        maxs     = xyz[:, :2].max(0)
        x_starts = np.arange(mins[0], maxs[0], self.tile_size)
        y_starts = np.arange(mins[1], maxs[1], self.tile_size)

        xy_starts = [(x0, y0) for x0 in x_starts for y0 in y_starts]
        pbar = tqdm(xy_starts, total=len(xy_starts), desc="Tiling", unit="cell", leave=False)

        for (x0, y0) in pbar:
            mask = (
                (xyz[:, 0] >= x0 - self.overlap) & (xyz[:, 0] < x0 + self.tile_size + self.overlap) &
                (xyz[:, 1] >= y0 - self.overlap) & (xyz[:, 1] < y0 + self.tile_size + self.overlap)
            )

            if mask.sum() < min_points:
                continue

            yield mask

    # --------------------------------------------------
    # Mask refinement
    # --------------------------------------------------
    def _refine_mask_2d(self, xyz, mask):
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
        if self.verbose:
            print(f"    Refining mask (closing radius: {self.closing_radius})...")

        x, y = xyz[:, 0], xyz[:, 1]
        x_min, y_min = x.min(), y.min()
        nx = int(np.ceil((x.max() - x_min) / self.grid_cell_size)) + 1
        ny = int(np.ceil((y.max() - y_min) / self.grid_cell_size)) + 1

        ix = np.clip(((x - x_min) / self.grid_cell_size).astype(np.int32), 0, nx - 1)
        iy = np.clip(((y - y_min) / self.grid_cell_size).astype(np.int32), 0, ny - 1)

        gm = np.zeros((ny, nx), dtype=bool)
        gm[iy[mask], ix[mask]] = True

        y_idx, x_idx = np.ogrid[-self.closing_radius:self.closing_radius+1, -self.closing_radius:self.closing_radius+1]
        struct = x_idx**2 + y_idx**2 <= self.closing_radius**2

        refined_gm = ndi.binary_closing(gm, structure=struct)

        refined_gm = ndi.binary_fill_holes(refined_gm)

        label_im, nb_labels = ndi.label(refined_gm)
        sizes = ndi.sum(refined_gm, label_im, range(nb_labels + 1))
        mask_size = sizes < self.min_cluster_size
        remove_pixel = mask_size[label_im]
        refined_gm[remove_pixel] = False

        refined_mask_pts = refined_gm[iy, ix]
        if self.verbose:
            added = refined_mask_pts.sum() - mask.sum()
            print(f"      Refinement complete. Change: {added:,} points.")

        return refined_mask_pts

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

    ground_embankment_mask = (labels == 2) | (labels == 11) | (labels == 1)
    xyz = xyz[ground_embankment_mask]
    labels = labels[ground_embankment_mask]

    del las
    return xyz, labels
# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    path      = pth.Path("/home/jakub-szota/Pobrane")
    db_params_path = "/home/jakub-szota/Dokumenty/TreeClustering/src/db_params.txt"
    embankment_config_path = "/home/jakub-szota/Dokumenty/TreeClustering/src/embankment_config.json"
    verbose = False
    segmenter = EmbankmentSegmenter(
        db_param_path=db_params_path,
        config_path=embankment_config_path,
        verbose=verbose
    )
    for i, laz_path in enumerate(path.glob("*.laz")):
        if verbose:
            print(f"\n[{i+1}] {laz_path.name}")

        xyz_raw, _ = load_data(laz_path)

        xyz, vis_labels = segmenter.segment_embankment(xyz_raw)
        if len(xyz > 0):
            xyz_vis = xyz.copy()
            xyz_vis[:, :2] -= xyz_vis[:, :2].mean(axis=0)
            xyz_vis[:, 2]  -= xyz_vis[:, 2].min()
            xyz_vis = xyz_vis.astype(np.float32)
            plot_cloud(xyz_vis, vis_labels)

if __name__ == "__main__":
    main()