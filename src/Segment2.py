from utils import plot_cloud
from utils import voxel_subsample_vectorized, remove_outliers

from typing import Union
import pathlib as pth
import numpy as np
import laspy
from scipy.spatial import cKDTree
import psycopg2
from shapely import wkt as shapely_wkt
from shapely.geometry import LineString, MultiLineString
from dataclasses import dataclass, field
from tqdm import tqdm

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
        """
        returns full copy of the object
        """
        return PCD(points=self.points.copy(), labels=self.labels.copy())



class SegmentEmbankment:
    def __init__(self,
                 cfg: dict,
                 db_param_path: Union[str, pth.Path], 
                 verbose: bool = False):
    
        self.cfg = cfg
        self.__db_param = self.__load_db_params(db_param_path)
        self.verbose = verbose

    def __load_db_params(self, path: Union[str, pth.Path]):
        path = pth.Path(path)
        params = {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                k, v = line.split("=", 1)
                params[k.strip()] = v.strip()
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
    
    def _load_tracks_from_db(self, bbox):
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
        rails = self._load_tracks_from_db(bbox)
        if len(rails) == 0:
            return np.zeros(xyz.shape[0], dtype=np.uint8)
        rail_xy = self._densify_lines(rails, step=0.5)
        tree = cKDTree(rail_xy)
        dist, _ = tree.query(xyz[:, :2])
        labels = (dist <= rail_radius).astype(np.uint8)

        if np.unique(labels).shape[0]<2:
            raise ValueError("No rails detected.")

        return labels
    
    
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

    
    def _base_segm(self, data: PCD) -> PCD:

        # here update data.labels

        return data
    
    def _big_segm(self, data: PCD) -> PCD:
        
        for mask in self._iter_tiles(data.points,
                                     self.cfg["tile_size"],
                                     self.cfg["overlap"],
                                     self.cfg["min_points"]):

            data_chunk = data.copy()
            data_chunk.subsample(mask)

            data_chunk = self._base_segm(data_chunk)

            data.update_mask(mask)
            data.labels[mask] = data_chunk.labels

            del data_chunk

        return data

    def _upsample_labels(data: PCD) -> PCD:
        """
        casts coarse labels after segmentation to fine resolution (original pcd)
        """

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

        n = data.points.shape[0]
        surviving = np.arange(n)

        mask = voxel_subsample_vectorized(data.points, voxel_size=0.5)
        surviving = surviving[mask]
        data.subsample(mask)

        mask = remove_outliers(data.points, nb_neighbors=40, std_ratio=2.0)
        surviving = surviving[mask]
        data.subsample(mask)

        # now surviving[i] = index in data_org of data.points[i]
        data_org.update_mask(surviving)

        if data.points.shape[0] < 5*10e6:
            data = self._base_segm(data)
        else:
            data = self._big_segm(data)
        
        data_org.labels[surviving] = data.labels
        data_org = self._upsample_labels(data_org)

        return data_org.labels