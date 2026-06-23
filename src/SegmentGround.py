from __future__ import annotations

import json
import logging
import pathlib as pth
from typing import Union

import numpy as np
import psycopg2
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from shapely import wkt as shapely_wkt
from shapely.geometry import LineString, MultiLineString
from tqdm import tqdm

# from utils.plot_sections import *


logger = logging.getLogger(__name__)


class GroundSegmenter:
    def __init__(
        self,
        cfg: dict,
        db_param_path: Union[str, pth.Path],
        verbose: bool = False,
    ):
        self.distance_limit = float(cfg["distance_limit"])
        self.ground_label = int(cfg["ground_label"])
        self.rail_label = int(cfg["rail_label"])
        self.rail_radius = float(cfg["rail_radius"])
        self.embankment_label = int(cfg["embankment_label"])
        self.ditch_label = int(cfg["ditch_label"])

        self.length_min = float(cfg["length_min"])
        self.length_max = float(cfg["length_max"])
        self.length = self.length_max

        self.width_margin = float(cfg["width_margin"])

        self.max_curve_ratio = float(cfg["max_curve_ratio"])
        self.curve_resolution = float(cfg["curve_resolution"])

        # Centerline voxelization follows curvature-check resolution.
        self.voxel = self.curve_resolution

        self.graph_x_bin = float(cfg["graph_x_bin"])
        self.graph_uphill_slope = float(cfg["graph_uphill_slope"])

        # Convert configured distances to graph samples.
        if "graph_embankment_min_stop_m" in cfg:
            graph_embankment_min_stop_m = float(cfg["graph_embankment_min_stop_m"])
        elif "graph_min_uphill_m" in cfg:
            graph_embankment_min_stop_m = float(cfg["graph_min_uphill_m"])
        else:
            graph_embankment_min_stop_m = self._read_graph_distance_m(
                cfg,
                meter_key="graph_embankment_min_stop_m",
                legacy_points_key="graph_min_uphill_points",
            )

        self.graph_embankment_min_stop_points = self._graph_meters_to_points(
            graph_embankment_min_stop_m,
            minimum_points=1,
        )

        graph_min_embankment_m = self._read_graph_distance_m(
            cfg,
            meter_key="graph_min_embankment_m",
            legacy_points_key="graph_min_embankment_points",
            default_m=graph_embankment_min_stop_m,
        )
        self.graph_min_embankment_points = self._graph_meters_to_points(
            graph_min_embankment_m,
            minimum_points=1,
        )

        self.graph_noise_points = int(cfg["graph_noise_points"])
        self.graph_smooth_window = int(cfg["graph_smooth_window"])
        self.graph_max_gap_bins = float(cfg["graph_max_gap_bins"])

        # Ditch thresholds are separate from embankment thresholds.
        graph_ditch_min_downhill_m = self._read_graph_distance_m(
            cfg,
            meter_key="graph_ditch_min_downhill_m",
            legacy_points_key="graph_ditch_min_downhill_points",
            default_m=graph_embankment_min_stop_m,
        )
        self.graph_ditch_min_downhill_points = self._graph_meters_to_points(
            graph_ditch_min_downhill_m,
            minimum_points=1,
        )

        graph_ditch_min_uphill_m = self._read_graph_distance_m(
            cfg,
            meter_key="graph_ditch_min_uphill_m",
            legacy_points_key="graph_ditch_min_uphill_points",
            default_m=graph_embankment_min_stop_m,
        )
        self.graph_ditch_min_uphill_points = self._graph_meters_to_points(
            graph_ditch_min_uphill_m,
            minimum_points=1,
        )

        graph_ditch_immediate_m = self._read_graph_distance_m(
            cfg,
            meter_key="graph_ditch_immediate_points_m",
            legacy_points_key="graph_ditch_immediate_points",
            default_m=(self.graph_noise_points + 1) * self.graph_x_bin,
        )
        self.graph_ditch_immediate_points = self._graph_meters_to_points(
            graph_ditch_immediate_m,
            minimum_points=0,
        )

        graph_ditch_max_flat_m = self._read_graph_distance_m(
            cfg,
            meter_key="graph_ditch_max_flat_m",
            legacy_points_key="graph_ditch_max_flat_points",
            default_m=(self.graph_noise_points + 2) * self.graph_x_bin,
        )
        self.graph_ditch_max_flat_points = self._graph_meters_to_points(
            graph_ditch_max_flat_m,
            minimum_points=0,
        )

        graph_ditch_max_uphill_m = self._read_graph_distance_m(
            cfg,
            meter_key="graph_ditch_max_uphill_m",
            legacy_points_key="graph_ditch_max_uphill_points",
            default_m=self.distance_limit,
        )
        self.graph_ditch_max_uphill_points = self._graph_meters_to_points(
            graph_ditch_max_uphill_m,
            minimum_points=self.graph_ditch_min_uphill_points,
        )

        # Ditch search range, measured outward from the rail.
        self.graph_ditch_search_min_m = float(
            cfg.get("graph_ditch_search_min_m", 0.0)
        )
        self.graph_ditch_search_max_m = float(
            cfg.get("graph_ditch_search_max_m", self.distance_limit)
        )

        if self.graph_ditch_search_min_m < 0.0:
            raise ValueError("graph_ditch_search_min_m must be non-negative.")

        if self.graph_ditch_search_max_m < self.graph_ditch_search_min_m:
            raise ValueError(
                "graph_ditch_search_max_m must be >= graph_ditch_search_min_m."
            )

        # Gaussian smoothing distance along the centerline, in metres.
        self.smooth = bool(cfg.get("smooth", True))
        self.smooth_level = float(cfg.get("smooth_level", 10.0))

        self.verbose = bool(verbose)
        self.__db_param = self._load_db_params(db_param_path)

        self.label_tmp = 100

    @classmethod
    def from_config(
        cls,
        cfg_path: Union[str, pth.Path],
        db_param_path: Union[str, pth.Path],
        verbose: bool = False,
    ):
        cfg_path = pth.Path(cfg_path)
        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        return cls(cfg=cfg, db_param_path=db_param_path, verbose=verbose)

    def _read_graph_distance_m(
        self,
        cfg: dict,
        meter_key: str,
        legacy_points_key: str,
        default_m: float | None = None,
    ) -> float:
        if meter_key in cfg:
            return float(cfg[meter_key])

        if legacy_points_key in cfg:
            return float(cfg[legacy_points_key]) * self.graph_x_bin

        if default_m is not None:
            return float(default_m)

        raise KeyError(meter_key)

    def _graph_meters_to_points(
        self,
        value_m: float,
        minimum_points: int,
    ) -> int:
        if self.graph_x_bin <= 0.0:
            raise ValueError("graph_x_bin must be positive.")

        if value_m < 0.0:
            raise ValueError("Graph distance thresholds must be non-negative.")

        return max(minimum_points, int(np.ceil(value_m / self.graph_x_bin)))

    def _drop_tmp_labels(self, labels: np.ndarray) -> np.ndarray:
        """Replace temporary splitter labels with embankment labels."""
        out = labels.copy()
        out[out == self.label_tmp] = self.embankment_label
        return out

    @staticmethod
    def _load_db_params(path: Union[str, pth.Path]):
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

    def __load_tracks_from_db(self, bbox):
        try:
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
        except psycopg2.Error:
            logger.exception("Failed to retrieve rail tracks from the database for bbox %s", bbox)
            raise

        lines = []
        for (wkt_line,) in rows:
            geom = shapely_wkt.loads(wkt_line)

            if isinstance(geom, LineString):
                lines.append(geom)
            elif isinstance(geom, MultiLineString):
                lines.extend(list(geom.geoms))

        return lines

    def _label_rail_points(self, xyz: np.ndarray, rail_radius: float = 0.5) -> np.ndarray:
        if xyz.shape[0] == 0:
            return np.zeros(0, dtype=bool)

        xmin = float(xyz[:, 0].min())
        xmax = float(xyz[:, 0].max())
        ymin = float(xyz[:, 1].min())
        ymax = float(xyz[:, 1].max())

        bbox = (xmin, ymin, xmax, ymax)
        rails = self.__load_tracks_from_db(bbox)

        if len(rails) == 0:
            return np.zeros(xyz.shape[0], dtype=bool)

        rail_xy = self._densify_lines(rails, step=0.5)

        if rail_xy.shape[0] == 0:
            return np.zeros(xyz.shape[0], dtype=bool)

        tree = cKDTree(rail_xy)
        dist, _ = tree.query(xyz[:, :2])

        return dist <= rail_radius

    @staticmethod
    def _densify_lines(lines, step: float = 0.5) -> np.ndarray:
        pts = []

        for line in lines:
            length = line.length
            distances = np.arange(0.0, length + step, step)

            for d in distances:
                p = line.interpolate(d)
                pts.append((p.x, p.y))

        return np.asarray(pts, dtype=np.float64)

    def _rotated_part(
        self,
        pcd: np.ndarray,
        xy: np.ndarray,
        z: np.ndarray,
        idx: np.ndarray,
        s: float,
        s1: float,
        centerline: np.ndarray,
        center_s: np.ndarray,
    ):
        p0 = self._point_at_s(s, centerline, center_s)
        p1 = self._point_at_s(s1, centerline, center_s)

        forward = self._unit(p1 - p0)

        # XY is right-handed if Y is forward and Z is up:
        # right = forward x up
        right = np.array([forward[1], -forward[0]])

        rel = xy[idx] - p0

        x = rel @ right
        y = rel @ forward

        y_min = y.min()

        out = pcd[idx].copy()
        out[:, 0] = x
        out[:, 1] = y - y_min
        out[:, 2] = z[idx]

        if self.width_margin:
            x_mid = 0.5 * (x.min() + x.max())
            width = x.max() - x.min()

            keep = np.abs(out[:, 0] - x_mid) <= 0.5 * (
                width + 2.0 * self.width_margin
            )

            out = out[keep]
            idx = idx[keep]

        return out, idx

    def _best_cut_end(
        self,
        s: float,
        centerline: np.ndarray,
        center_s: np.ndarray,
    ) -> float:
        total = center_s[-1]
        s1 = min(s + self.length_max, total)

        bad_start = self._first_bad_curve_start(
            s0=s,
            s1=s1,
            centerline=centerline,
            center_s=center_s,
        )

        if bad_start is None:
            return s1

        min_end = min(s + self.length_min, total)

        if bad_start <= min_end:
            return min_end

        return bad_start

    def _first_bad_curve_start(
        self,
        s0: float,
        s1: float,
        centerline: np.ndarray,
        center_s: np.ndarray,
    ) -> float | None:
        if s1 - s0 <= self.length_min:
            return None

        samples = np.arange(s0, s1, self.curve_resolution)

        if len(samples) == 0 or samples[-1] != s1:
            samples = np.r_[samples, s1]

        for a, b in zip(samples[:-1], samples[1:]):
            if b <= a:
                continue

            ratio = self._curve_ratio_between(
                s0=a,
                s1=b,
                centerline=centerline,
                center_s=center_s,
            )

            if ratio > self.max_curve_ratio:
                return float(a)

        return None

    def _curve_ratio_between(
        self,
        s0: float,
        s1: float,
        centerline: np.ndarray,
        center_s: np.ndarray,
    ) -> float:
        p0 = self._point_at_s(s0, centerline, center_s)
        p1 = self._point_at_s(s1, centerline, center_s)

        arc_len = s1 - s0
        chord_len = np.linalg.norm(p1 - p0)

        if chord_len == 0:
            return np.inf

        return arc_len / chord_len

    def _build_centerline(self, xy: np.ndarray) -> np.ndarray:
        keys = np.floor(xy / self.voxel).astype(np.int64)

        _, inverse = np.unique(keys, axis=0, return_inverse=True)

        sums = np.zeros((inverse.max() + 1, 2), dtype=np.float64)
        counts = np.bincount(inverse)

        np.add.at(sums, inverse, xy)
        nodes = sums / counts[:, None]

        center = nodes.mean(axis=0)
        centered = nodes - center

        _, _, vh = np.linalg.svd(centered, full_matrices=False)

        forward = vh[0]
        right = np.array([forward[1], -forward[0]])

        u = centered @ forward
        v = centered @ right

        u0 = u.min()
        bin_size = self.voxel
        bins = np.floor((u - u0) / bin_size).astype(np.int64)

        unique_bins = np.unique(bins)

        trace_u = []
        trace_v = []

        for b in unique_bins:
            mask = bins == b

            if np.count_nonzero(mask) < 3:
                continue

            trace_u.append(np.median(u[mask]))
            trace_v.append(np.median(v[mask]))

        trace_u = np.asarray(trace_u, dtype=np.float64)
        trace_v = np.asarray(trace_v, dtype=np.float64)

        order = np.argsort(trace_u)
        trace_u = trace_u[order]
        trace_v = trace_v[order]

        keep = np.r_[True, np.diff(trace_u) > 1e-9]
        trace_u = trace_u[keep]
        trace_v = trace_v[keep]

        if len(trace_u) < 4:
            return center + trace_u[:, None] * forward + trace_v[:, None] * right

        # Bigger = stiffer. Rail should not bend aggressively.
        smoothing = len(trace_u) * self.voxel**2 * 25.0

        spline = UnivariateSpline(
            trace_u,
            trace_v,
            k=3,
            s=smoothing,
        )

        rough_len = trace_u[-1] - trace_u[0]
        n_samples = max(len(trace_u), int(rough_len / self.voxel))

        u_new = np.linspace(trace_u[0], trace_u[-1], n_samples)
        v_new = spline(u_new)

        return center + u_new[:, None] * forward + v_new[:, None] * right

    @staticmethod
    def _assign_points_to_centerline(
        xy: np.ndarray,
        centerline: np.ndarray,
        center_s: np.ndarray,
    ) -> np.ndarray:
        tree = cKDTree(centerline)
        _, nearest = tree.query(xy)

        return center_s[nearest]

    @staticmethod
    def _point_at_s(
        s: float,
        centerline: np.ndarray,
        center_s: np.ndarray,
    ) -> np.ndarray:
        i = np.searchsorted(center_s, s, side="right") - 1
        i = np.clip(i, 0, len(centerline) - 2)

        s0 = center_s[i]
        s1 = center_s[i + 1]

        t = 0.0 if s1 == s0 else (s - s0) / (s1 - s0)

        return (1.0 - t) * centerline[i] + t * centerline[i + 1]

    @staticmethod
    def _arc_length(line: np.ndarray) -> np.ndarray:
        d = np.diff(line, axis=0)
        ds = np.linalg.norm(d, axis=1)

        return np.r_[0.0, np.cumsum(ds)]

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)

        if norm == 0:
            raise ValueError("Cannot normalize zero-length vector.")

        return v / norm

    @staticmethod
    def cast_to_xz_plane(tile: np.ndarray, z_bin: float):
        x = tile[:, 0]
        z = tile[:, 2]

        z0 = z.min()
        bins = np.floor((z - z0) / z_bin).astype(np.int64)

        order = np.argsort(bins)
        bins = bins[order]
        x = x[order]

        unique_bins, start = np.unique(bins, return_index=True)

        x_sum = np.add.reduceat(x, start)
        count = np.diff(np.r_[start, len(x)])

        z_centers = z0 + (unique_bins + 0.5) * z_bin
        mean_x = x_sum / count

        return z_centers, mean_x

    @staticmethod
    def get_gradient(graph: np.ndarray) -> np.ndarray:
        x = graph[:, 0]
        z = graph[:, 1]

        dz_dx = np.gradient(z, x)

        return np.column_stack((x, dz_dx))

    def get_curve_ratio(
        self,
        s: float,
        centerline: np.ndarray,
        center_s: np.ndarray,
        length: float | None = None,
    ) -> float:
        if length is None:
            length = self.length_max

        s1 = min(s + length, center_s[-1])

        return self._curve_ratio_between(
            s0=s,
            s1=s1,
            centerline=centerline,
            center_s=center_s,
        )

    @staticmethod
    def _flip_graph_x(graph: np.ndarray | None) -> np.ndarray | None:
        if graph is None:
            return None

        flipped = graph.copy()
        flipped[:, 0] *= -1.0

        return flipped

    @staticmethod
    def _unflip_sections_x(
        emb: np.ndarray,
        ditch: np.ndarray,
        rest: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        def unflip(section: np.ndarray) -> np.ndarray:
            if section is None or len(section) == 0:
                return section

            out = section.copy()
            out[:, 0] *= -1.0
            return out

        return unflip(emb), unflip(ditch), unflip(rest)

    @staticmethod
    def _first_run(mask: np.ndarray, start: int, stop: int, min_points: int) -> tuple[int, int] | None:
        i = max(0, start)
        stop = min(stop, len(mask))

        while i < stop:
            if not mask[i]:
                i += 1
                continue

            run_start = i

            while i < stop and mask[i]:
                i += 1

            if i - run_start >= min_points:
                return run_start, i

        return None

    @staticmethod
    def _end_run(mask: np.ndarray, start: int) -> int:
        i = start

        while i < len(mask) and mask[i]:
            i += 1

        return i

    @staticmethod
    def _capped_uphill_end(
        uphill_mask: np.ndarray,
        uphill_start: int,
        uphill_end: int,
        max_uphill_points: int,
    ) -> int:
        end = GroundSegmenter._end_run(uphill_mask, uphill_end)
        return min(end, uphill_start + max_uphill_points)

    @staticmethod
    def _find_ditch_interval(
        emb_end: int,
        uphill_mask: np.ndarray,
        downhill_mask: np.ndarray,
        min_downhill_points: int,
        min_uphill_points: int,
        immediate_points: int,
        max_flat_points: int,
        max_uphill_points: int,
        search_start: int,
        search_stop: int,
    ) -> tuple[int, int, bool] | None:
        n = len(uphill_mask)

        search_start = max(0, min(int(search_start), n))
        search_stop = max(search_start, min(int(search_stop), n))

        if search_start >= search_stop:
            return None

        # Search only within the configured interval.
        immediate_start = max(emb_end, search_start)

        # Case 1: uphill starts immediately after the embankment.
        if immediate_start < search_stop:
            immediate_stop = min(
                search_stop,
                immediate_start + immediate_points + min_uphill_points,
            )
            immediate = GroundSegmenter._first_run(
                mask=uphill_mask,
                start=immediate_start,
                stop=immediate_stop,
                min_points=min_uphill_points,
            )

            if immediate is not None:
                uphill_start, uphill_end = immediate

                # Include the bottom before the uphill wall.
                if uphill_start - immediate_start <= immediate_points:
                    ditch_start = max(search_start, uphill_start - immediate_points)
                    return ditch_start, min(
                        search_stop,
                        GroundSegmenter._capped_uphill_end(
                            uphill_mask=uphill_mask,
                            uphill_start=uphill_start,
                            uphill_end=uphill_end,
                            max_uphill_points=max_uphill_points,
                        ),
                    ), True

        # Case 2: find a ditch elsewhere in the search range.
        i = search_start

        while i < search_stop:
            uphill = GroundSegmenter._first_run(
                mask=uphill_mask,
                start=i,
                stop=search_stop,
                min_points=min_uphill_points,
            )
            downhill = GroundSegmenter._first_run(
                mask=downhill_mask,
                start=i,
                stop=search_stop,
                min_points=min_downhill_points,
            )

            if uphill is None and downhill is None:
                return None

            # An uphill-only ditch candidate.
            uphill_candidate = None
            if uphill is not None:
                uphill_start, uphill_end = uphill
                uphill_candidate = (
                    uphill_start,
                    min(
                        search_stop,
                        GroundSegmenter._capped_uphill_end(
                            uphill_mask=uphill_mask,
                            uphill_start=uphill_start,
                            uphill_end=uphill_end,
                            max_uphill_points=max_uphill_points,
                        ),
                    ),
                    False,
                )

            # Downhill, optional flat bottom, then uphill.
            downhill_candidate = None
            if downhill is not None:
                downhill_start, downhill_end = downhill
                j = downhill_end

                while j < search_stop and downhill_mask[j]:
                    j += 1

                flat_stop = min(
                    search_stop,
                    j + max_flat_points + min_uphill_points,
                )
                uphill_after = GroundSegmenter._first_run(
                    mask=uphill_mask,
                    start=j,
                    stop=flat_stop,
                    min_points=min_uphill_points,
                )

                if uphill_after is not None:
                    uphill_start, uphill_end = uphill_after
                    downhill_candidate = (
                        downhill_start,
                        min(
                            search_stop,
                            GroundSegmenter._capped_uphill_end(
                                uphill_mask=uphill_mask,
                                uphill_start=uphill_start,
                                uphill_end=uphill_end,
                                max_uphill_points=max_uphill_points,
                            ),
                        ),
                        False,
                    )

            candidates = [
                candidate
                for candidate in (downhill_candidate, uphill_candidate)
                if candidate is not None
            ]

            if candidates:
                return min(candidates, key=lambda item: item[0])

            # Try after this downhill run.
            if downhill is not None:
                i = max(i + 1, downhill[1] + 1)
            else:
                return None

        return None

    @staticmethod
    def _trim_ditch_interval(
        interval: tuple[int, int],
        emb_end: int,
        search_start: int,
        search_stop: int,
        graph_len: int,
        allow_embankment_overlap: bool = False,
    ) -> tuple[int, int] | None:
        """Clip a ditch interval to the allowed graph range."""
        start, end = interval

        if allow_embankment_overlap:
            start = max(int(start), int(search_start), 0)
        else:
            start = max(int(start), int(emb_end), int(search_start), 0)

        end = min(int(end), int(search_stop), int(graph_len))

        if end <= start:
            return None

        return start, end

    @staticmethod
    def split_graph_by_gradient(
        graph: np.ndarray,
        uphill_slope: float,
        embankment_min_stop_points: int,
        min_embankment_points: int,
        noise_points: int,
        smooth_window: int,
        ditch_min_downhill_points: int | None = None,
        ditch_min_uphill_points: int | None = None,
        ditch_immediate_points: int | None = None,
        ditch_max_flat_points: int | None = None,
        ditch_max_uphill_points: int | None = None,
        ditch_search_min_m: float = 0.0,
        ditch_search_max_m: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split a rail-to-edge graph into embankment, ditch, and ground."""
        empty = np.empty((0, 2), dtype=np.float64)

        if graph is None or len(graph) < 2:
            return empty, empty, empty

        x = graph[:, 0].astype(np.float64)
        z = graph[:, 1].astype(np.float64)

        order = np.argsort(x)
        x = x[order]
        z = z[order]

        keep = np.r_[True, np.diff(x) > 1e-9]
        x = x[keep]
        z = z[keep]

        full_graph = np.column_stack((x, z))

        if len(x) < 2:
            return empty, empty, full_graph

        z_for_gradient = z.copy()

        if smooth_window >= 3 and len(z_for_gradient) >= smooth_window:
            if smooth_window % 2 == 0:
                smooth_window += 1

            pad = smooth_window // 2
            z_padded = np.pad(z_for_gradient, pad_width=pad, mode="edge")
            kernel = np.ones(smooth_window, dtype=np.float64) / smooth_window
            z_for_gradient = np.convolve(z_padded, kernel, mode="valid")

        dz_dx = np.gradient(z_for_gradient, x)

        stop_mask = dz_dx >= -uphill_slope
        emb_end = None

        min_embankment_points = max(1, int(min_embankment_points))
        i = max(noise_points, min_embankment_points)

        while i < len(stop_mask):
            if not stop_mask[i]:
                i += 1
                continue

            start = i

            while i < len(stop_mask) and stop_mask[i]:
                i += 1

            if i - start >= embankment_min_stop_points and start >= min_embankment_points:
                emb_end = start
                break

        if emb_end is None:
            return full_graph, empty, empty

        ditch_min_downhill_points = (
            embankment_min_stop_points
            if ditch_min_downhill_points is None
            else int(ditch_min_downhill_points)
        )
        ditch_min_uphill_points = (
            embankment_min_stop_points
            if ditch_min_uphill_points is None
            else int(ditch_min_uphill_points)
        )
        ditch_immediate_points = (
            noise_points + 1
            if ditch_immediate_points is None
            else int(ditch_immediate_points)
        )
        ditch_max_flat_points = (
            noise_points + 2
            if ditch_max_flat_points is None
            else int(ditch_max_flat_points)
        )
        ditch_max_uphill_points = (
            len(full_graph)
            if ditch_max_uphill_points is None
            else int(ditch_max_uphill_points)
        )
        ditch_max_uphill_points = max(
            ditch_min_uphill_points,
            ditch_max_uphill_points,
        )

        ditch_search_min_m = max(0.0, float(ditch_search_min_m))
        if ditch_search_max_m is None:
            ditch_search_max_m = float(x[-1] - x[0])
        else:
            ditch_search_max_m = float(ditch_search_max_m)

        if ditch_search_max_m < ditch_search_min_m:
            raise ValueError("ditch_search_max_m must be >= ditch_search_min_m.")

        # Measure ditch search distance from the rail side.
        x_from_side_start = x - x[0]
        ditch_search_start = int(
            np.searchsorted(x_from_side_start, ditch_search_min_m, side="left")
        )
        ditch_search_stop = int(
            np.searchsorted(x_from_side_start, ditch_search_max_m, side="right")
        )

        uphill_mask = dz_dx > uphill_slope
        downhill_mask = dz_dx < -uphill_slope

        ditch_interval = GroundSegmenter._find_ditch_interval(
            emb_end=emb_end,
            uphill_mask=uphill_mask,
            downhill_mask=downhill_mask,
            min_downhill_points=ditch_min_downhill_points,
            min_uphill_points=ditch_min_uphill_points,
            immediate_points=ditch_immediate_points,
            max_flat_points=ditch_max_flat_points,
            max_uphill_points=ditch_max_uphill_points,
            search_start=ditch_search_start,
            search_stop=ditch_search_stop,
        )

        if ditch_interval is None:
            embankment = full_graph[:emb_end]
            return embankment, empty, full_graph[emb_end:]

        ditch_start_raw, ditch_end_raw, allow_embankment_overlap = ditch_interval

        ditch_interval = GroundSegmenter._trim_ditch_interval(
            interval=(ditch_start_raw, ditch_end_raw),
            emb_end=emb_end,
            search_start=ditch_search_start,
            search_stop=ditch_search_stop,
            graph_len=len(full_graph),
            allow_embankment_overlap=allow_embankment_overlap,
        )

        if ditch_interval is None:
            embankment = full_graph[:emb_end]
            return embankment, empty, full_graph[emb_end:]

        ditch_start, ditch_end = ditch_interval

        # Keep embankment and ditch ranges contiguous.
        embankment = full_graph[:ditch_start]
        ditch = full_graph[ditch_start:ditch_end]
        rest = full_graph[ditch_end:]

        return embankment, ditch, rest

    @staticmethod
    def _mask_points_by_graph_section(
        points: np.ndarray,
        section: np.ndarray | None,
        x_padding: float,
    ) -> np.ndarray:
        mask = np.zeros(points.shape[0], dtype=bool)

        if section is None or len(section) == 0:
            return mask

        x_min = section[:, 0].min() - x_padding
        x_max = section[:, 0].max() + x_padding

        return (points[:, 0] >= x_min) & (points[:, 0] <= x_max)

    def _center_chunk_x_on_rail(
        self,
        points_chunk_rotated: np.ndarray,
        rail_mask: np.ndarray,
    ) -> bool:
        """Center local X on the rail band."""
        if np.count_nonzero(rail_mask) == 0:
            return False

        rail_x = points_chunk_rotated[rail_mask, 0]

        # Use the central rail band rather than tile edges.
        x_center = 0.5 * (
            np.percentile(rail_x, 5.0)
            + np.percentile(rail_x, 95.0)
        )

        points_chunk_rotated[:, 0] -= x_center

        return True

    def _find_nearest_points(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        rail_mask: np.ndarray,
    ) -> np.ndarray:
        nearest_points_mask = np.zeros(points.shape[0], dtype=bool)

        candidate_mask = (labels == self.ground_label) & ~rail_mask

        nearest_points_mask[rail_mask] = True

        if np.count_nonzero(candidate_mask) == 0:
            return nearest_points_mask

        rail_points = points[rail_mask]

        if rail_points.shape[0] == 0:
            return nearest_points_mask

        tree = cKDTree(
            rail_points[:, :2],
            copy_data=False,
        )

        distances, _ = tree.query(
            points[candidate_mask, :2],
            k=1,
            distance_upper_bound=self.distance_limit,
            workers=-1,
        )

        nearest_points_mask[candidate_mask] = distances < self.distance_limit

        return nearest_points_mask

    def iter_rectangles(
        self,
        pcd: np.ndarray,
        xy: np.ndarray,
        z: np.ndarray,
        centerline: np.ndarray,
        center_s: np.ndarray,
        point_s: np.ndarray
    ):
        s = 0.0
        total = center_s[-1]

        with tqdm(desc="Tiling", unit="tile", leave=False, position=1, disable=not self.verbose) as pbar:
            while s < total:
                s1 = self._best_cut_end(
                    s=s,
                    centerline=centerline,
                    center_s=center_s,
                )

                idx = np.flatnonzero((point_s >= s) & (point_s < s1))

                if len(idx):
                    pbar.update(1)
                    yield self._rotated_part(
                        pcd=pcd,
                        xy=xy,
                        z=z,
                        idx=idx,
                        s=s,
                        s1=s1,
                        centerline=centerline,
                        center_s=center_s,
                    )

                s = s1

    def _build_xz_graph(
        self,
        xz: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = xz[:, 0]
        z = xz[:, 1]

        x0 = x.min()
        bins = np.floor((x - x0) / self.graph_x_bin).astype(np.int64)

        order = np.argsort(bins)
        bins = bins[order]
        z = z[order]
        labels = labels[order]

        unique_bins, start = np.unique(bins, return_index=True)

        count = np.diff(np.r_[start, len(z)])
        x_centers = x0 + (unique_bins + 0.5) * self.graph_x_bin
        mean_z = np.add.reduceat(z, start) / count
        tmp_count = np.add.reduceat(labels == self.label_tmp, start)

        graph = np.column_stack((x_centers, mean_z))
        graph_labels = np.where(
            tmp_count > 0,
            self.label_tmp,
            self.ground_label,
        )

        return graph, graph_labels

    def _split_graph_into_sides(
        self,
        graph: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        rail_indices = np.flatnonzero(labels == self.label_tmp)

        if len(rail_indices) == 0:
            return None, None

        first_rail = rail_indices[0]
        last_rail = rail_indices[-1]

        left_graph = graph[:first_rail] if first_rail else None
        right_graph = graph[last_rail + 1:] if last_rail + 1 < len(graph) else None

        if left_graph is not None:
            left_graph = self._fill_graph_gaps(left_graph)

        if right_graph is not None:
            right_graph = self._fill_graph_gaps(right_graph)

        return left_graph, right_graph

    def _fill_graph_gaps(self, graph: np.ndarray) -> np.ndarray:
        if len(graph) < 2:
            return graph

        n_bins = int(round((graph[-1, 0] - graph[0, 0]) / self.graph_x_bin)) + 1
        x = graph[0, 0] + np.arange(n_bins) * self.graph_x_bin
        z = np.interp(x, graph[:, 0], graph[:, 1])

        return np.column_stack((x, z))

    @staticmethod
    def _has_graph_section(section: np.ndarray | None) -> bool:
        return section is not None and len(section) > 0

    def _mask_points_between_embankment_sides(
        self,
        points: np.ndarray,
        left_emb: np.ndarray | None,
        right_emb: np.ndarray | None,
        x_padding: float,
    ) -> np.ndarray:
        """Return the area between the two embankment sides."""
        mask = np.zeros(points.shape[0], dtype=bool)

        if not self._has_graph_section(left_emb):
            return mask

        if not self._has_graph_section(right_emb):
            return mask

        emb_x = np.concatenate((left_emb[:, 0], right_emb[:, 0]))
        x_min = float(emb_x.min()) - x_padding
        x_max = float(emb_x.max()) + x_padding

        return (points[:, 0] >= x_min) & (points[:, 0] <= x_max)

    def _apply_section_labels(
        self,
        labels_sectioned: np.ndarray,
        points_chunk_rotated: np.ndarray,
        left_emb: np.ndarray,
        left_ditch: np.ndarray,
        left_rest: np.ndarray,
        right_emb: np.ndarray,
        right_ditch: np.ndarray,
        right_rest: np.ndarray,
    ) -> np.ndarray:
        x_padding = 0.5 * self.graph_x_bin

        left_emb_mask = self._mask_points_by_graph_section(
            points=points_chunk_rotated,
            section=left_emb,
            x_padding=x_padding,
        )
        left_ditch_mask = self._mask_points_by_graph_section(
            points=points_chunk_rotated,
            section=left_ditch,
            x_padding=x_padding,
        )
        left_rest_mask = self._mask_points_by_graph_section(
            points=points_chunk_rotated,
            section=left_rest,
            x_padding=x_padding,
        )

        right_emb_mask = self._mask_points_by_graph_section(
            points=points_chunk_rotated,
            section=right_emb,
            x_padding=x_padding,
        )
        right_ditch_mask = self._mask_points_by_graph_section(
            points=points_chunk_rotated,
            section=right_ditch,
            x_padding=x_padding,
        )
        right_rest_mask = self._mask_points_by_graph_section(
            points=points_chunk_rotated,
            section=right_rest,
            x_padding=x_padding,
        )

        emb_mask = left_emb_mask | right_emb_mask
        ditch_mask = left_ditch_mask | right_ditch_mask
        rest_mask = left_rest_mask | right_rest_mask
        center_emb_mask = self._mask_points_between_embankment_sides(
            points=points_chunk_rotated,
            left_emb=left_emb,
            right_emb=right_emb,
            x_padding=x_padding,
        )

        # Ditch labels override overlapping embankment labels.
        labels_sectioned[rest_mask] = self.ground_label
        labels_sectioned[emb_mask | center_emb_mask] = self.embankment_label
        labels_sectioned[ditch_mask] = self.ditch_label

        return labels_sectioned

    # Boundary smoothing

    @staticmethod
    def _project_to_sl_frame(
        xy: np.ndarray,
        centerline: np.ndarray,
        center_s: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project XY points onto the nearest centerline segment."""
        if xy.shape[0] == 0:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty

        if centerline.shape[0] < 2:
            raise ValueError("centerline must contain at least two points.")

        seg_start = centerline[:-1].astype(np.float64, copy=False)
        seg_end = centerline[1:].astype(np.float64, copy=False)
        seg_vec = seg_end - seg_start
        seg_len2 = np.einsum("ij,ij->i", seg_vec, seg_vec)
        seg_len2 = np.where(seg_len2 < 1e-12, 1.0, seg_len2)

        seg_mid = 0.5 * (seg_start + seg_end)
        tree = cKDTree(seg_mid)
        _, seg_idx = tree.query(xy)

        a = seg_start[seg_idx]
        v = seg_vec[seg_idx]
        vv = seg_len2[seg_idx]

        t = np.einsum("ij,ij->i", xy - a, v) / vv
        t = np.clip(t, 0.0, 1.0)

        projection = a + t[:, None] * v

        seg_len = np.sqrt(vv)
        tangent = v / seg_len[:, None]
        right = np.column_stack((tangent[:, 1], -tangent[:, 0]))

        s_vals = center_s[seg_idx] + t * seg_len
        x_lateral = np.einsum("ij,ij->i", xy - projection, right)

        return s_vals, x_lateral

    @staticmethod
    def _smooth_boundary(
        s_centers: np.ndarray,
        boundary: np.ndarray,
        smooth_sigma_m: float,
    ) -> np.ndarray:
        """Fill gaps and Gaussian-smooth a boundary curve."""
        valid = ~np.isnan(boundary)

        if np.sum(valid) < 2:
            return boundary.copy()

        filled = boundary.copy()
        filled[~valid] = np.interp(
            s_centers[~valid],
            s_centers[valid],
            boundary[valid],
        )

        bin_size = (s_centers[-1] - s_centers[0]) / max(len(s_centers) - 1, 1)
        sigma_bins = max(0.0, float(smooth_sigma_m) / max(bin_size, 1e-9))

        if sigma_bins <= 1e-9:
            return filled

        return gaussian_filter1d(filled, sigma=sigma_bins, mode="nearest")

    @staticmethod
    def _build_horizontal_label_grid(
        s: np.ndarray,
        x_abs: np.ndarray,
        labels: np.ndarray,
        ground_label: int,
        embankment_label: int,
        ditch_label: int,
        s_bin_centers: np.ndarray,
        s_bin_size: float,
        x_bin_size: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bin labels on an (s, lateral-distance) grid."""
        n_s = len(s_bin_centers)

        if n_s == 0 or s.shape[0] == 0:
            empty_labels = np.empty((0, 0), dtype=np.int16)
            empty_occ = np.empty((0, 0), dtype=bool)
            empty_x = np.empty(0, dtype=np.float64)
            return empty_labels, empty_occ, empty_x

        x_bin_size = max(float(x_bin_size), 1e-6)
        x_max = max(0.0, float(np.max(x_abs)))
        n_x = max(1, int(np.ceil((x_max + x_bin_size) / x_bin_size)))
        x_centers = (np.arange(n_x, dtype=np.float64) + 0.5) * x_bin_size

        s_lo = s_bin_centers[0] - 0.5 * s_bin_size
        s_idx = np.floor((s - s_lo) / s_bin_size).astype(np.int64)
        s_idx = np.clip(s_idx, 0, n_s - 1)

        x_idx = np.floor(x_abs / x_bin_size).astype(np.int64)
        x_idx = np.clip(x_idx, 0, n_x - 1)

        flat = s_idx * n_x + x_idx
        size = n_s * n_x

        total = np.zeros(size, dtype=np.int32)
        ground_count = np.zeros(size, dtype=np.int32)
        emb_count = np.zeros(size, dtype=np.int32)
        ditch_count = np.zeros(size, dtype=np.int32)

        np.add.at(total, flat, 1)
        np.add.at(ground_count, flat[labels == ground_label], 1)
        np.add.at(emb_count, flat[labels == embankment_label], 1)
        np.add.at(ditch_count, flat[labels == ditch_label], 1)

        total = total.reshape(n_s, n_x)
        ground_count = ground_count.reshape(n_s, n_x)
        emb_count = emb_count.reshape(n_s, n_x)
        ditch_count = ditch_count.reshape(n_s, n_x)

        occupied = total > 0
        cell_labels = np.full((n_s, n_x), ground_label, dtype=np.int16)

        # Prefer ditch, then embankment, on ties.
        emb_wins = (emb_count >= ground_count) & (emb_count > 0)
        ditch_wins = (
            (ditch_count >= ground_count)
            & (ditch_count >= emb_count)
            & (ditch_count > 0)
        )

        cell_labels[emb_wins] = embankment_label
        cell_labels[ditch_wins] = ditch_label
        cell_labels[~occupied] = -1

        return cell_labels, occupied, x_centers

    @staticmethod
    def _boundary_curves_from_horizontal_grid(
        cell_labels: np.ndarray,
        occupied: np.ndarray,
        x_centers: np.ndarray,
        embankment_label: int,
        ditch_label: int,
        pcd_edge_margin: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract boundaries and mark rows that reach the cloud edge."""
        n_s = cell_labels.shape[0]

        emb_outer = np.full(n_s, np.nan, dtype=np.float64)
        ditch_inner = np.full(n_s, np.nan, dtype=np.float64)
        ditch_outer = np.full(n_s, np.nan, dtype=np.float64)
        edge_locked = np.zeros(n_s, dtype=bool)

        for row in range(n_s):
            occupied_cols = np.flatnonzero(occupied[row])

            if occupied_cols.size == 0:
                continue

            max_all_x = x_centers[occupied_cols[-1]]

            emb_cols = np.flatnonzero(cell_labels[row] == embankment_label)
            if emb_cols.size:
                outer = x_centers[emb_cols[-1]]
                if outer >= max_all_x - pcd_edge_margin:
                    edge_locked[row] = True
                else:
                    emb_outer[row] = outer

            ditch_cols = np.flatnonzero(cell_labels[row] == ditch_label)
            if ditch_cols.size:
                inner = x_centers[ditch_cols[0]]
                outer = x_centers[ditch_cols[-1]]
                if outer >= max_all_x - pcd_edge_margin:
                    edge_locked[row] = True
                else:
                    ditch_inner[row] = inner
                    ditch_outer[row] = outer

        return emb_outer, ditch_inner, ditch_outer, edge_locked

    def _smooth_label_boundaries(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        rail_mask: np.ndarray,
        centerline: np.ndarray,
        center_s: np.ndarray,
    ) -> np.ndarray:
        """Smooth labels in the centerline and lateral-distance frame."""
        relevant_mask = np.isin(
            labels,
            [self.ground_label, self.embankment_label, self.ditch_label],
        )

        if not np.any(relevant_mask):
            return labels.copy()

        rel_idx = np.flatnonzero(relevant_mask)
        rel_xy = points[rel_idx, :2]
        rel_labels = labels[rel_idx]

        s_vals, x_vals = self._project_to_sl_frame(rel_xy, centerline, center_s)

        s_bin_size = max(float(self.curve_resolution), 1e-6)
        x_bin_size = max(float(self.graph_x_bin), 1e-6)

        s0 = float(s_vals.min())
        s1 = float(s_vals.max())
        n_bins = max(4, int(np.ceil((s1 - s0) / s_bin_size)))
        s_centers = s0 + (np.arange(n_bins, dtype=np.float64) + 0.5) * s_bin_size
        s_lo = s_centers[0] - 0.5 * s_bin_size

        result = labels.copy()

        for side_sign in (+1.0, -1.0):
            # Keep the centerline on one side only.
            if side_sign > 0:
                side_mask = x_vals >= 0.0
            else:
                side_mask = x_vals < 0.0

            if not np.any(side_mask):
                continue

            side_all_idx = np.flatnonzero(side_mask)
            side_all_s = s_vals[side_all_idx]
            side_all_x = np.abs(x_vals[side_all_idx])
            side_all_labels = rel_labels[side_all_idx]

            labelled_mask = np.isin(
                side_all_labels,
                [self.embankment_label, self.ditch_label],
            )

            if not np.any(labelled_mask):
                continue

            # Keep the grid near labelled terrain.
            labelled_x_max = float(side_all_x[labelled_mask].max())
            x_limit = max(
                labelled_x_max + 3.0 * x_bin_size,
                float(self.distance_limit) + x_bin_size,
            )
            corridor_mask = side_all_x <= x_limit

            if not np.any(corridor_mask):
                continue

            side_idx = side_all_idx[corridor_mask]
            side_s = s_vals[side_idx]
            side_x = np.abs(x_vals[side_idx])
            side_labels = rel_labels[side_idx]

            cell_labels, occupied, x_centers = self._build_horizontal_label_grid(
                s=side_s,
                x_abs=side_x,
                labels=side_labels,
                ground_label=self.ground_label,
                embankment_label=self.embankment_label,
                ditch_label=self.ditch_label,
                s_bin_centers=s_centers,
                s_bin_size=s_bin_size,
                x_bin_size=x_bin_size,
            )

            if cell_labels.size == 0:
                continue

            emb_outer, ditch_inner, ditch_outer, edge_locked = (
                self._boundary_curves_from_horizontal_grid(
                    cell_labels=cell_labels,
                    occupied=occupied,
                    x_centers=x_centers,
                    embankment_label=self.embankment_label,
                    ditch_label=self.ditch_label,
                    pcd_edge_margin=x_bin_size,
                )
            )

            emb_valid = ~np.isnan(emb_outer)
            ditch_valid = ~np.isnan(ditch_outer) & ~np.isnan(ditch_inner)

            # Do not smooth a boundary at the cloud edge.
            if np.sum(emb_valid) < 2:
                result[rel_idx[side_idx]] = side_labels
                continue

            emb_smooth = self._smooth_boundary(
                s_centers,
                emb_outer,
                self.smooth_level,
            )

            if np.sum(ditch_valid) >= 2:
                ditch_inner_smooth = self._smooth_boundary(
                    s_centers,
                    ditch_inner,
                    self.smooth_level,
                )
                ditch_outer_smooth = self._smooth_boundary(
                    s_centers,
                    ditch_outer,
                    self.smooth_level,
                )
            else:
                ditch_inner_smooth = ditch_inner.copy()
                ditch_outer_smooth = ditch_outer.copy()

            emb_presence = emb_valid.astype(np.float64)
            ditch_presence = ditch_valid.astype(np.float64)

            emb_presence_smooth = self._smooth_boundary(
                s_centers,
                emb_presence,
                self.smooth_level,
            )
            ditch_presence_smooth = self._smooth_boundary(
                s_centers,
                ditch_presence,
                self.smooth_level,
            )

            side_s_idx = np.floor((side_s - s_lo) / s_bin_size).astype(np.int64)
            side_s_idx = np.clip(side_s_idx, 0, len(s_centers) - 1)
            locked_at_points = edge_locked[side_s_idx]

            smoothable = ~locked_at_points

            if not np.any(smoothable):
                result[rel_idx[side_idx]] = side_labels
                continue

            smooth_side_idx = side_idx[smoothable]
            smooth_s = s_vals[smooth_side_idx]
            smooth_x = np.abs(x_vals[smooth_side_idx])

            emb_at_pts = np.interp(smooth_s, s_centers, emb_smooth)
            emb_presence_at_pts = (
                np.interp(smooth_s, s_centers, emb_presence_smooth) > 0.3
            )

            ditch_inner_at_pts = np.interp(
                smooth_s,
                s_centers,
                ditch_inner_smooth,
            )
            ditch_outer_at_pts = np.interp(
                smooth_s,
                s_centers,
                ditch_outer_smooth,
            )
            ditch_presence_at_pts = (
                np.interp(smooth_s, s_centers, ditch_presence_smooth) > 0.3
            )

            # Reset only rows with a usable boundary.
            result[rel_idx[smooth_side_idx]] = self.ground_label

            emb_new = emb_presence_at_pts & (smooth_x <= emb_at_pts)
            result[rel_idx[smooth_side_idx[emb_new]]] = self.embankment_label

            ditch_new = (
                ditch_presence_at_pts
                & (smooth_x > ditch_inner_at_pts)
                & (smooth_x <= ditch_outer_at_pts)
            )
            result[rel_idx[smooth_side_idx[ditch_new]]] = self.ditch_label

            # Preserve cloud-edge rows.
            if np.any(locked_at_points):
                locked_side_idx = side_idx[locked_at_points]
                result[rel_idx[locked_side_idx]] = rel_labels[locked_side_idx]

        # Keep rail points on the embankment through smoothing.
        result[rail_mask] = self.embankment_label

        return result

    def segment(self, points: np.ndarray, labels: np.ndarray) -> np.ndarray:
        full_labels = np.asarray(labels, dtype=np.uint8).copy()

        with tqdm(desc="Filtering PCD", unit="step", total=3, leave=False, position=1, disable=not self.verbose) as pbar:
            ground_mask = (
                (full_labels == self.ground_label)
                | (full_labels == self.rail_label)
                | (full_labels == self.label_tmp)
            )
            ground_idx = np.flatnonzero(ground_mask)

            if ground_idx.size == 0:
                return full_labels

            ground_rail = points[ground_idx].copy()
            ground_rail_labels = full_labels[ground_idx].copy()
            original_rail_mask = ground_rail_labels == self.rail_label
            original_tmp_mask = ground_rail_labels == self.label_tmp
            ground_rail_labels[original_rail_mask | original_tmp_mask] = self.ground_label

            pbar.update(1)
            rail_mask = self._label_rail_points(
                ground_rail,
                rail_radius=self.rail_radius,
            )
            pbar.update(2)

            if np.count_nonzero(rail_mask) == 0:
                return full_labels

            ground_rail_labels[rail_mask] = self.label_tmp

            # Local normalization for numerical stability.
            ground_rail[:, :2] -= ground_rail[:, :2].mean(axis=0)
            ground_rail[:, 2] -= ground_rail[:, 2].min()
            ground_rail = ground_rail.astype(np.float32, copy=False)

            rail = ground_rail[rail_mask]
            pbar.update(1)

        with tqdm(desc="Finding centerline", unit="tile", total=2, leave=False, position=1, disable=not self.verbose) as pbar:
            centerline_xy = rail[:, :2]

            if centerline_xy.shape[0] == 0:
                return full_labels

            xy = ground_rail[:, :2]
            z = ground_rail[:, 2]

            centerline = self._build_centerline(centerline_xy)
            pbar.update(1)

            if centerline.shape[0] < 2:
                return full_labels

            center_s = self._arc_length(centerline)

            if center_s[-1] <= 0:
                return full_labels

            point_s = self._assign_points_to_centerline(
                xy=xy,
                centerline=centerline,
                center_s=center_s,
            )

            pbar.update(2)

        for points_chunk_rotated, indices in self.iter_rectangles(
            pcd=ground_rail,
            xy=xy,
            z=z,
            centerline=centerline,
            center_s=center_s,
            point_s=point_s,
        ):
            if points_chunk_rotated.shape[0] == 0:
                continue

            labels_chunk = ground_rail_labels[indices]
            rail_mask_chunk = rail_mask[indices]

            nearest_points_mask = self._find_nearest_points(
                points=points_chunk_rotated,
                labels=labels_chunk,
                rail_mask=rail_mask_chunk,
            )

            if np.count_nonzero(nearest_points_mask) == 0:
                continue

            points_nearest = points_chunk_rotated[nearest_points_mask]
            labels_nearest = labels_chunk[nearest_points_mask]
            section_indices = indices[nearest_points_mask]
            rail_mask_nearest = rail_mask_chunk[nearest_points_mask]

            # plot_xyz_cloud(points_nearest, labels_nearest)

            if points_nearest.shape[0] == 0:
                continue

            centered = self._center_chunk_x_on_rail(
                points_chunk_rotated=points_nearest,
                rail_mask=rail_mask_nearest,
            )

            if not centered:
                continue

            xz = points_nearest[:, [0, 2]]

            if xz.shape[0] == 0:
                continue
            
            graph, graph_labels = self._build_xz_graph(xz, labels_nearest)

            if graph.shape[0] < 2:
                continue

            left_graph, right_graph = self._split_graph_into_sides(
                graph,
                graph_labels,
            )

            # plot_xz_graph_split_by_rail(
            #     xz,
            #     graph,
            #     graph_labels,
            #     left_graph,
            #     right_graph,
            # )

            empty = np.empty((0, 2), dtype=np.float64)

            left_emb = empty
            left_ditch = empty
            left_rest = empty

            right_emb = empty
            right_ditch = empty
            right_rest = empty

            if left_graph is not None:
                left_graph_flipped = self._flip_graph_x(left_graph)

                left_emb, left_ditch, left_rest = self.split_graph_by_gradient(
                    graph=left_graph_flipped,
                    uphill_slope=self.graph_uphill_slope,
                    embankment_min_stop_points=self.graph_embankment_min_stop_points,
                    min_embankment_points=self.graph_min_embankment_points,
                    noise_points=self.graph_noise_points,
                    smooth_window=self.graph_smooth_window,
                    ditch_min_downhill_points=self.graph_ditch_min_downhill_points,
                    ditch_min_uphill_points=self.graph_ditch_min_uphill_points,
                    ditch_immediate_points=self.graph_ditch_immediate_points,
                    ditch_max_flat_points=self.graph_ditch_max_flat_points,
                    ditch_max_uphill_points=self.graph_ditch_max_uphill_points,
                    ditch_search_min_m=self.graph_ditch_search_min_m,
                    ditch_search_max_m=self.graph_ditch_search_max_m,
                )

                left_emb, left_ditch, left_rest = self._unflip_sections_x(
                    left_emb,
                    left_ditch,
                    left_rest,
                )

            if right_graph is not None:
                right_emb, right_ditch, right_rest = self.split_graph_by_gradient(
                    graph=right_graph,
                    uphill_slope=self.graph_uphill_slope,
                    embankment_min_stop_points=self.graph_embankment_min_stop_points,
                    min_embankment_points=self.graph_min_embankment_points,
                    noise_points=self.graph_noise_points,
                    smooth_window=self.graph_smooth_window,
                    ditch_min_downhill_points=self.graph_ditch_min_downhill_points,
                    ditch_min_uphill_points=self.graph_ditch_min_uphill_points,
                    ditch_immediate_points=self.graph_ditch_immediate_points,
                    ditch_max_flat_points=self.graph_ditch_max_flat_points,
                    ditch_max_uphill_points=self.graph_ditch_max_uphill_points,
                    ditch_search_min_m=self.graph_ditch_search_min_m,
                    ditch_search_max_m=self.graph_ditch_search_max_m,
                )

            # plot_xz_side_sections(
            #     xz,
            #     left_emb,
            #     left_ditch,
            #     left_rest,
            #     right_emb,
            #     right_ditch,
            #     right_rest,
            # )

            labels_sectioned = labels_nearest.copy()

            labels_sectioned = self._apply_section_labels(
                labels_sectioned=labels_sectioned,
                points_chunk_rotated=points_nearest,
                left_emb=left_emb,
                left_ditch=left_ditch,
                left_rest=left_rest,
                right_emb=right_emb,
                right_ditch=right_ditch,
                right_rest=right_rest,
            )

            full_indices = ground_idx[section_indices]
            labels_sectioned[rail_mask_nearest] = self.embankment_label
            full_labels[full_indices] = labels_sectioned

        # Remove temporary labels before smoothing.
        full_labels[ground_idx] = self._drop_tmp_labels(full_labels[ground_idx])

        # Smooth boundaries if enabled.
        if self.smooth:
            full_labels[ground_idx] = self._smooth_label_boundaries(
                points=ground_rail,
                labels=full_labels[ground_idx],
                rail_mask=rail_mask,
                centerline=centerline,
                center_s=center_s,
            )

        # Do not return temporary labels.
        full_labels[ground_idx] = self._drop_tmp_labels(full_labels[ground_idx])

        original_rail_mask = labels[ground_idx] == self.rail_label
        rail_on_embankment_mask = original_rail_mask & (
            full_labels[ground_idx] == self.embankment_label
        )
        full_labels[ground_idx[original_rail_mask]] = self.ground_label
        full_labels[ground_idx[rail_on_embankment_mask]] = self.rail_label

        # plot_cloud(
        #     ground_rail,
        #     full_labels[ground_idx],
        #     title="After smoothing",
        # )

        return full_labels

if __name__ == "__main__":
    import laspy
    from utils.plot_cloud import plot_cloud

    las_file = laspy.read(
        "/Users/michalsiniarski/Documents/DATA/BRIK/LAW2PROCESS/MOD/16-25_mod.laz"
    )

    points = np.vstack((las_file.x, las_file.y, las_file.z)).T
    labels = np.asarray(las_file.classification)


    cfg_path = pth.Path(__file__).parent / "ground_segm_config.json"
    db_param_path = pth.Path(__file__).parent / "db_params.txt"

    cutter = GroundSegmenter.from_config(
        cfg_path=cfg_path,
        db_param_path=db_param_path,
        verbose=True,
    )


    labels_sectioned = cutter.segment(points, labels)
    plot_cloud(points, labels_sectioned)
