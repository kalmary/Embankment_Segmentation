from utils import plot_cloud
from utils import voxel_subsample_vectorized, remove_outliers

from typing import Union
import pathlib as pth
import numpy as np
import laspy
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import psycopg2
from shapely import wkt as shapely_wkt
from shapely.geometry import LineString, MultiLineString
from dataclasses import dataclass, field
from tqdm import tqdm
import json

@dataclass
class PCD:
    points: np.ndarray = field(default_factory=lambda: np.array([]))
    labels: np.ndarray = field(default_factory=lambda: np.array([]))
    processed: np.ndarray = field(default_factory=lambda: np.zeros([], dtype=bool))

    def subsample(self, idx):
        """
        subselect points and labels. remember to update mask on original data!!!
        """

        self.points = self.points[idx]
        self.labels = self.labels[idx]

    def update_mask(self, mask: np.ndarray):
        """
        tells which points from old data have been used
        """
        self.processed[mask] = True

    def copy(self):
        return PCD(
            points=self.points.copy(),
            labels=self.labels.copy(),
            processed=self.processed.copy()
        )



class SegmentEmbankment:
    def __init__(self,
                 cfg: dict,
                 db_param_path: Union[str, pth.Path], 
                 verbose: bool = False):
    
        self.cfg = cfg
        self.__db_param = self._load_db_params(db_param_path)
        self.verbose = verbose

    def _load_db_params(self, path: Union[str, pth.Path]):
        path = pth.Path(path)
        print(path)
        
        params = {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                k, v = line.split("=", 1)
                params[k.strip()] = v.strip()
        print(params)
        return params

    def load_data(self, las_path: Union[str, pth.Path]) -> PCD:
        las_path = pth.Path(las_path)
        las = laspy.read(las_path)

        xyz = np.stack([
            np.asarray(las.x, dtype=np.float64),
            np.asarray(las.y, dtype=np.float64),
            np.asarray(las.z, dtype=np.float64),
        ], axis=1)
    
        labels    = np.array(las.classification, dtype=np.int32)

    
        ground_embankment_mask = (labels == self.cfg["ground_label"]) | (labels == self.cfg["rail_label"])
        xyz = xyz[ground_embankment_mask]
        labels = labels[ground_embankment_mask]
        labels = np.zeros(labels.shape, dtype=np.uint8)

        del las

        data = PCD(xyz, labels)

        return data
    
    @classmethod
    def from_config(cls, cfg_path: Union[str, pth.Path], db_param_path: Union[str, pth.Path], verbose: bool = False):
        cfg_path = pth.Path(cfg_path)
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)

        return cls(cfg=cfg, db_param_path=db_param_path, verbose=verbose)

    
    def __load_tracks_from_db(self, bbox):
        conn = psycopg2.connect(**self.__db_param)
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
    

    def _label_rail_points(self, xyz, rail_radius: float=0.5):
        xmin = float(xyz[:,0].min())
        xmax = float(xyz[:,0].max())
        ymin = float(xyz[:,1].min())
        ymax = float(xyz[:,1].max())
        bbox = (xmin, ymin, xmax, ymax)
        rails = self.__load_tracks_from_db(bbox)
        if len(rails) == 0:
            return np.zeros(xyz.shape[0], dtype=np.uint8)
        rail_xy = self._densify_lines(rails, step=0.5)
        tree = cKDTree(rail_xy)
        dist, _ = tree.query(xyz[:, :2])
        labels = (dist <= rail_radius).astype(np.uint8)

        if np.unique(labels).shape[0]<2:
            raise ValueError("No rails detected.")

        return labels
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
            print(f"    Refining mask (closing radius: {self.cfg["closing_radius"]})...")
        
        x, y = xyz[:, 0], xyz[:, 1]
        x_min, y_min = x.min(), y.min()
        nx = int(np.ceil((x.max() - x_min) / self.cfg["grid_cell_size"])) + 1
        ny = int(np.ceil((y.max() - y_min) / self.cfg["grid_cell_size"])) + 1
        
        ix = np.clip(((x - x_min) / self.cfg["grid_cell_size"]).astype(np.int32), 0, nx - 1)
        iy = np.clip(((y - y_min) / self.cfg["grid_cell_size"]).astype(np.int32), 0, ny - 1)

        gm = np.zeros((ny, nx), dtype=bool)
        gm[iy[mask], ix[mask]] = True

        y_idx, x_idx = np.ogrid[-self.cfg["closing_radius"]:self.cfg["closing_radius"]+1, -self.cfg["closing_radius"]:self.cfg["closing_radius"]+1]
        struct = x_idx**2 + y_idx**2 <= self.cfg["closing_radius"]**2

        refined_gm = ndi.binary_closing(gm, structure=struct)

        refined_gm = ndi.binary_fill_holes(refined_gm)

        label_im, nb_labels = ndi.label(refined_gm)
        sizes = ndi.sum(refined_gm, label_im, range(nb_labels + 1))
        mask_size = sizes < self.cfg["min_cluster_size"]
        remove_pixel = mask_size[label_im]
        refined_gm[remove_pixel] = False

        refined_mask_pts = refined_gm[iy, ix]
        if self.verbose:
            added = refined_mask_pts.sum() - mask.sum()
            print(f"      Refinement complete. Change: {added:,} points.")
        
        return refined_mask_pts
    
    @staticmethod
    def _densify_lines(lines, step=0.5):
        pts = []
        for line in lines:
            length = line.length
            distances = np.arange(0, length + step, step)
            for d in distances:
                p = line.interpolate(d)
                pts.append((p.x, p.y))
        return np.array(pts)
    
    def _iter_tiles(self, xyz: np.ndarray, tile_size: float=40.0, overlap: float=10.0, min_points: int=1024): # TODO make program aware of surviving points
        mins     = xyz[:, :2].min(0)
        maxs     = xyz[:, :2].max(0)
        x_starts = np.arange(mins[0], maxs[0], tile_size)
        y_starts = np.arange(mins[1], maxs[1], tile_size)
    
        xy_starts = [(x0, y0) for x0 in x_starts for y0 in y_starts]
        if self.verbose:
            pbar = tqdm(xy_starts, total=len(xy_starts), desc="Tiling", unit="cell", leave=False)
        else:
            pbar = xy_starts
        
        for (x0, y0) in pbar:
            mask = (
                (xyz[:, 0] >= x0 - overlap) & (xyz[:, 0] < x0 + tile_size + overlap) &
                (xyz[:, 1] >= y0 - overlap) & (xyz[:, 1] < y0 + tile_size + overlap)
            )
            
            if mask.sum() < min_points:
                continue
    
            yield mask

    def _grow_embankment_mask(self,
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
        nx = int(np.ceil((x.max() - x_min) / self.cfg["grid_cell_size"])) + 1
        ny = int(np.ceil((y.max() - y_min) / self.cfg["grid_cell_size"])) + 1
        ix = np.clip(((x - x_min) / self.cfg["grid_cell_size"]).astype(np.int32), 0, nx - 1)
        iy = np.clip(((y - y_min) / self.cfg["grid_cell_size"]).astype(np.int32), 0, ny - 1)

        z_grid = np.full((ny, nx), np.nan, dtype=np.float32)
        order = np.argsort(z)
        z_grid[iy[order], ix[order]] = z[order]

        invalid_mask = np.isnan(z_grid)
        if invalid_mask.all(): return new_final
            
        _, indices = ndi.distance_transform_edt(invalid_mask, return_distances=True, return_indices=True)
        z_filled = z_grid[tuple(indices)]
        z_smoothed = ndi.gaussian_filter(z_filled, sigma=1.0)

        gy, gx = np.gradient(z_smoothed, self.cfg["grid_cell_size"])
        grad = np.sqrt(gx**2 + gy**2)
        
        gm_tracks = np.zeros((ny, nx), dtype=bool)
        gm_tracks[iy[mask], ix[mask]] = True

        dist_from_track, nearest_track_idx = ndi.distance_transform_edt(~gm_tracks, return_distances=True, return_indices=True)
        dist_from_track *= self.cfg["grid_cell_size"]

        z_tracks_only = np.full((ny, nx), np.nan, dtype=np.float32)
        z_tracks_only[gm_tracks] = z_smoothed[gm_tracks]

        z_nearest_track = z_tracks_only[tuple(nearest_track_idx)]

        valid_elevation = z_smoothed <= (z_nearest_track + 0.5)

        valid_slope = (grad >= self.cfg["min_slope"]) & (grad <= self.cfg["max_slope"])
        
        valid_growth = (dist_from_track <= self.cfg["crown_width_m"]) | (valid_slope & valid_elevation)
        
        valid_growth &= (dist_from_track <= self.cfg["max_dist_m"])

        struct = ndi.generate_binary_structure(2, 2)
        expanded = gm_tracks.copy()
        max_iters = int(self.cfg["max_dist_m"] / self.cfg["grid_cell_size"]) + 5
        
        for i in range(max_iters):
            new_expanded = (ndi.binary_dilation(expanded, struct) & valid_growth) | gm_tracks
            if np.array_equal(expanded, new_expanded): break
            expanded = new_expanded

        final_expanded_pts = expanded[iy, ix]
        new_final[final_expanded_pts] = 1
        
        return new_final

    def _base_segm(self, data: PCD) -> PCD:
        track_labels = data.labels  # already set by _label_rail_points in segment()

        embankment_labels = self._grow_embankment_mask(data.points, track_labels)

        mask2fix = embankment_labels == 1
        mask2d = self._refine_mask_2d(data.points, mask2fix)

        embankment_labels = np.zeros_like(embankment_labels)
        embankment_labels[mask2d] = 1

        # 0 = ground, 1 = rail, 2 = embankment
        vis_labels = np.zeros(len(data.points), dtype=np.uint8)
        vis_labels[track_labels == 1] = 1
        vis_labels[embankment_labels == 1] = 2

        data.labels = vis_labels
        return data
    
    def _big_segm(self, data: PCD) -> PCD:
        data.processed = np.zeros(data.points.shape[0], dtype=bool)

        for mask in self._iter_tiles(data.points, **self._get_cfg("tile_size", "overlap", "min_points")):
            data_chunk = data.copy()
            data_chunk.subsample(mask)
            data_chunk = self._base_segm(data_chunk)
            data.update_mask(mask)
            data.labels[mask] = data_chunk.labels

        return data
    
    def _upsample_labels(self, data: PCD, k: int = 10, sigma: float = 1.0, chunk_size: int = 500_000) -> PCD:
        """
        Propagates labels from processed points to unprocessed ones via
        Gaussian-weighted KNN voting. Tree built on small processed set,
        large unprocessed set queried in chunks.
        """
        processed_mask   = data.processed
        unprocessed_mask = ~processed_mask

        if unprocessed_mask.sum() == 0:
            return data

        src_pts    = data.points[processed_mask]
        src_labels = data.labels[processed_mask]

        src_probs = np.zeros((src_labels.shape[0], 2), dtype=np.float32)
        src_probs[src_labels == 0, 0] = 1.0
        src_probs[src_labels == 1, 1] = 1.0

        tree = cKDTree(src_pts)

        query_pts  = data.points[unprocessed_mask]
        out_labels = np.zeros(query_pts.shape[0], dtype=np.uint8)

        pbar = range(0, query_pts.shape[0], chunk_size)
        if self.verbose:
            pbar = tqdm(pbar, total=query_pts.shape[0], desc="Upsampling", unit="chunk", leave=False)

        for start in pbar:
            end   = min(start + chunk_size, query_pts.shape[0])
            chunk = query_pts[start:end]

            dists, idxs  = tree.query(chunk, k=k)
            weights       = np.exp(-0.5 * (dists / sigma) ** 2)
            weights      /= weights.sum(axis=1, keepdims=True)

            neighbor_probs        = src_probs[idxs]                              # (C, k, 2)
            agg                   = (weights[:, :, None] * neighbor_probs).sum(axis=1)  # (C, 2)
            out_labels[start:end] = agg.argmax(axis=1).astype(np.uint8)

        data.labels[unprocessed_mask] = out_labels
        return data


    
    def segment(self, data: PCD) -> np.ndarray:
        """
        assumes data is already filtered (ground and rails only)
        """

        data.labels = self._label_rail_points(data.points)

        data.points -= data.points.mean(axis=0)
        data.points = data.points.astype(np.float32)

        data_org = data.copy()
        data_org.labels = np.zeros_like(data.labels, dtype=np.uint8)
        data_org.processed = np.zeros(data.points.shape[0], dtype=bool)  # dodaj tę linię

        n = data.points.shape[0]
        surviving = np.arange(n)

        mask = voxel_subsample_vectorized(data.points, voxel_size=0.1)
        surviving = surviving[mask]
        data.subsample(mask)

        mask = remove_outliers(data.points, nb_neighbors=40, std_ratio=2.0)
        surviving = surviving[mask]
        data.subsample(mask)

        # now surviving[i] = index in data_org of data.points[i]
        data_org.update_mask(surviving)

        if data.points.shape[0] < 5*10e6:
            data = self._base_segm(data)
            data_org.labels[surviving] = data.labels
        else:
            data = self._big_segm(data)
            tiled_surviving = surviving[data.processed]
            data_org.labels[tiled_surviving] = data.labels[data.processed]
        data_org = self._upsample_labels(data_org)

        return data_org.labels
    
def main():
    path = pth.Path("/home/jakub-szota/Pobrane")
    db_params_path = "/home/jakub-szota/Dokumenty/Embankment_Segmentation/src/db_params.txt"
    embankment_config_path = "/home/jakub-szota/Dokumenty/Embankment_Segmentation/src/embankment_config.json"
    verbose = True
    segmenter = SegmentEmbankment.from_config(
    cfg_path=embankment_config_path,
    db_param_path=db_params_path,
    verbose=verbose
    )
    for i, laz_path in enumerate(path.glob("*.laz")):
        if verbose:
            print(f"\n[{i+1}] {laz_path.name}")

        data = segmenter.load_data(laz_path)
        xyz_orig = data.points.copy()  # zapisz oryginalne punkty przed segment()
        labels = segmenter.segment(data)

        if len(labels) > 0:
            xyz_vis = xyz_orig  # użyj oryginalnych punktów
            xyz_vis[:, :2] -= xyz_vis[:, :2].mean(axis=0)
            xyz_vis[:, 2]  -= xyz_vis[:, 2].min()
            xyz_vis = xyz_vis.astype(np.float32)
            plot_cloud(xyz_vis, labels)


if __name__ == "__main__":
    main()