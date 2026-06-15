from __future__ import annotations

import json
import pathlib as pth
from typing import Union

import numpy as np
import psycopg2
from scipy.interpolate import UnivariateSpline
from scipy.spatial import cKDTree
from shapely import wkt as shapely_wkt
from shapely.geometry import LineString, MultiLineString

from utils.plot_sections import *


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
        self.graph_min_uphill_points = int(cfg["graph_min_uphill_points"])
        self.graph_min_embankment_points = int(
            cfg.get("graph_min_embankment_points", self.graph_min_uphill_points)
        )
        self.graph_noise_points = int(cfg["graph_noise_points"])
        self.graph_smooth_window = int(cfg["graph_smooth_window"])
        self.graph_max_gap_bins = float(cfg["graph_max_gap_bins"])

        # Ditch detection is deliberately separate from embankment detection.
        # Embankment may stop at a flat bottom; ditch may start immediately
        # with uphill, or later with a downhill->uphill valley pattern.
        self.graph_ditch_min_downhill_points = int(
            cfg.get("graph_ditch_min_downhill_points", self.graph_min_uphill_points)
        )
        self.graph_ditch_min_uphill_points = int(
            cfg.get("graph_ditch_min_uphill_points", self.graph_min_uphill_points)
        )
        self.graph_ditch_immediate_points = int(
            cfg.get("graph_ditch_immediate_points", self.graph_noise_points + 1)
        )
        self.graph_ditch_max_flat_points = int(
            cfg.get("graph_ditch_max_flat_points", self.graph_noise_points + 2)
        )
        self.smoothing = bool(cfg.get("smoothing", True))

        self.verbose = bool(verbose)
        self.__db_param = self._load_db_params(db_param_path)

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
    def _find_ditch_interval(
        emb_end: int,
        uphill_mask: np.ndarray,
        downhill_mask: np.ndarray,
        min_downhill_points: int,
        min_uphill_points: int,
        immediate_points: int,
        max_flat_points: int,
    ) -> tuple[int, int] | None:
        n = len(uphill_mask)

        if emb_end >= n:
            return None

        # Case 1: embankment already reached the ditch bottom.
        # Then ditch starts directly after embankment and is represented
        # by the stable uphill wall.
        immediate_stop = min(n, emb_end + immediate_points + min_uphill_points)
        immediate = GroundSegmenter._first_run(
            mask=uphill_mask,
            start=emb_end,
            stop=immediate_stop,
            min_points=min_uphill_points,
        )

        if immediate is not None:
            uphill_start, uphill_end = immediate

            # Include a short flat/noisy transition before uphill as ditch bottom,
            # but keep ditch attached to embankment.
            if uphill_start - emb_end <= immediate_points:
                return emb_end, GroundSegmenter._end_run(uphill_mask, uphill_end)

        # Case 2: a ditch appears later. In this version, if a later ditch is
        # found, embankment will be extended up to the ditch start by the caller.
        # Therefore the final topology remains:
        #     embankment -> ditch -> rest
        # not:
        #     embankment -> rest -> ditch -> rest
        i = emb_end

        while i < n:
            uphill = GroundSegmenter._first_run(
                mask=uphill_mask,
                start=i,
                stop=n,
                min_points=min_uphill_points,
            )
            downhill = GroundSegmenter._first_run(
                mask=downhill_mask,
                start=i,
                stop=n,
                min_points=min_downhill_points,
            )

            if uphill is None and downhill is None:
                return None

            # Uphill-only candidate. This covers a ditch whose downhill side was
            # already consumed by embankment/flat bottom detection.
            uphill_candidate = None
            if uphill is not None:
                uphill_start, uphill_end = uphill
                uphill_candidate = (
                    uphill_start,
                    GroundSegmenter._end_run(uphill_mask, uphill_end),
                )

            # Downhill -> optional flat/noisy bottom -> uphill candidate.
            downhill_candidate = None
            if downhill is not None:
                downhill_start, downhill_end = downhill
                j = downhill_end

                while j < n and downhill_mask[j]:
                    j += 1

                flat_stop = min(n, j + max_flat_points + min_uphill_points)
                uphill_after = GroundSegmenter._first_run(
                    mask=uphill_mask,
                    start=j,
                    stop=flat_stop,
                    min_points=min_uphill_points,
                )

                if uphill_after is not None:
                    _, uphill_end = uphill_after
                    downhill_candidate = (
                        downhill_start,
                        GroundSegmenter._end_run(uphill_mask, uphill_end),
                    )

            candidates = [
                candidate
                for candidate in (downhill_candidate, uphill_candidate)
                if candidate is not None
            ]

            if candidates:
                return min(candidates, key=lambda item: item[0])

            # The first downhill was not followed by uphill quickly enough.
            # Continue after it and look for the next possible structure.
            if downhill is not None:
                i = max(i + 1, downhill[1] + 1)
            else:
                return None

        return None

    @staticmethod
    def split_graph_by_gradient(
        graph: np.ndarray,
        uphill_slope: float,
        min_uphill_points: int,
        min_embankment_points: int,
        noise_points: int,
        smooth_window: int,
        ditch_min_downhill_points: int | None = None,
        ditch_min_uphill_points: int | None = None,
        ditch_immediate_points: int | None = None,
        ditch_max_flat_points: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Input graph must always be oriented:
            x increases from rail outward.

        Embankment:
            starts near rail and continues while the profile goes downhill.
            ends when flat OR uphill lasts for at least min_uphill_points.

        Ditch:
            is searched after embankment and can be either:
            - immediate uphill after embankment, when embankment reached bottom;
            - later downhill -> optional flat/noisy bottom -> uphill.

        Embankment stop condition is ignored until at least
        min_embankment_points graph samples are included.

        No minimum depth test is used.
        """
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

            if i - start >= min_uphill_points and start >= min_embankment_points:
                emb_end = start
                break

        if emb_end is None:
            return full_graph, empty, empty

        ditch_min_downhill_points = (
            min_uphill_points
            if ditch_min_downhill_points is None
            else int(ditch_min_downhill_points)
        )
        ditch_min_uphill_points = (
            min_uphill_points
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
        )

        if ditch_interval is None:
            embankment = full_graph[:emb_end]
            return embankment, empty, full_graph[emb_end:]

        ditch_start, ditch_end = ditch_interval
        ditch_start = max(emb_end, ditch_start)
        ditch_end = min(len(full_graph), max(ditch_start, ditch_end))

        if ditch_end <= ditch_start:
            embankment = full_graph[:emb_end]
            return embankment, empty, full_graph[emb_end:]

        # Important: if ditch starts later, embankment is extended to the ditch
        # start. This keeps labels contiguous: embankment -> ditch -> rest.
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
        """
        Center local X coordinate on rail points.

        This replaces the incorrect min/max tile centering. Min/max centering is
        unstable because the terrain extent is usually asymmetric. Rail-based
        centering keeps x=0 tied to the track/centerline reference.
        """
        if np.count_nonzero(rail_mask) == 0:
            return False

        rail_x = points_chunk_rotated[rail_mask, 0]

        # Robust center of rail band. Better than min/max of the whole tile.
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
        point_s: np.ndarray,
    ):
        s = 0.0
        total = center_s[-1]

        while s < total:
            s1 = self._best_cut_end(
                s=s,
                centerline=centerline,
                center_s=center_s,
            )

            idx = np.flatnonzero((point_s >= s) & (point_s < s1))

            if len(idx):
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

    def _build_xz_graph(self, xz: np.ndarray) -> np.ndarray:
        x = xz[:, 0]
        z = xz[:, 1]

        x0 = x.min()
        bins = np.floor((x - x0) / self.graph_x_bin).astype(np.int64)

        order = np.argsort(bins)
        bins = bins[order]
        z = z[order]

        unique_bins, start = np.unique(bins, return_index=True)

        count = np.diff(np.r_[start, len(z)])
        x_centers = x0 + (unique_bins + 0.5) * self.graph_x_bin
        mean_z = np.add.reduceat(z, start) / count

        return np.column_stack((x_centers, mean_z))

    def _split_graph_into_sides(
        self,
        graph: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        max_gap = self.graph_x_bin * self.graph_max_gap_bins
        split_at = np.flatnonzero(np.diff(graph[:, 0]) > max_gap) + 1
        graph_parts = [part for part in np.split(graph, split_at) if len(part)]

        left_graph = None
        right_graph = None

        if len(graph_parts) == 1:
            if np.mean(graph_parts[0][:, 0]) < 0:
                left_graph = graph_parts[0]
            else:
                right_graph = graph_parts[0]

        elif len(graph_parts) > 1:
            side_graphs = sorted(graph_parts, key=len, reverse=True)[:2]

            left_graph, right_graph = sorted(
                side_graphs,
                key=lambda part: np.mean(part[:, 0]),
            )

        return left_graph, right_graph

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

        # Order matters. Ditch should override embankment/rest if padding overlaps.
        labels_sectioned[rest_mask] = self.ground_label
        labels_sectioned[emb_mask] = self.embankment_label
        labels_sectioned[ditch_mask] = self.ditch_label

        return labels_sectioned

    @staticmethod
    def _centerline_coordinates(
        xy: np.ndarray,
        centerline: np.ndarray,
        center_s: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        tree = cKDTree(centerline)
        _, nearest = tree.query(xy)

        previous = np.maximum(nearest - 1, 0)
        following = np.minimum(nearest + 1, len(centerline) - 1)
        tangent = centerline[following] - centerline[previous]
        tangent_norm = np.linalg.norm(tangent, axis=1)
        valid = tangent_norm > 0
        tangent[valid] /= tangent_norm[valid, None]
        tangent[~valid] = np.array([1.0, 0.0])

        right = np.column_stack((tangent[:, 1], -tangent[:, 0]))
        offset = np.sum((xy - centerline[nearest]) * right, axis=1)

        return center_s[nearest], offset

    def _fit_label_border(
        self,
        station: np.ndarray,
        offset: np.ndarray,
        label_mask: np.ndarray,
        side: int,
    ):
        mask = label_mask & (offset * side > 0)

        if np.count_nonzero(mask) < 4:
            return None

        station_side = station[mask]
        offset_side = offset[mask]
        bin_size = max(self.length_min, self.curve_resolution)
        bins = np.floor((station_side - station_side.min()) / bin_size).astype(np.int64)

        border_station = []
        border_offset = []
        percentile = 90.0 if side > 0 else 10.0

        for bin_id in np.unique(bins):
            bin_mask = bins == bin_id

            if np.count_nonzero(bin_mask) < 2:
                continue

            border_station.append(np.median(station_side[bin_mask]))
            border_offset.append(np.percentile(offset_side[bin_mask], percentile))

        border_station = np.asarray(border_station, dtype=np.float64)
        border_offset = np.asarray(border_offset, dtype=np.float64)

        if border_station.size < 2:
            return None

        order = np.argsort(border_station)
        border_station = border_station[order]
        border_offset = border_offset[order]
        keep = np.r_[True, np.diff(border_station) > 1e-9]
        border_station = border_station[keep]
        border_offset = border_offset[keep]

        if border_station.size < 2:
            return None

        degree = min(3, border_station.size - 1)
        smoothing = border_station.size * self.graph_x_bin**2

        return UnivariateSpline(
            border_station,
            border_offset,
            k=degree,
            s=smoothing,
            ext=3,
        )

    def _smooth_labels(
        self,
        ground_xy: np.ndarray,
        original_labels: np.ndarray,
        full_labels: np.ndarray,
        ground_idx: np.ndarray,
        rail_mask: np.ndarray,
        centerline: np.ndarray,
        center_s: np.ndarray,
    ) -> np.ndarray:
        if not self.smoothing:
            return full_labels

        ground_labels = full_labels[ground_idx].copy()
        station, offset = self._centerline_coordinates(
            xy=ground_xy,
            centerline=centerline,
            center_s=center_s,
        )

        embankment_mask = ground_labels == self.embankment_label
        ditch_mask = ground_labels == self.ditch_label

        left_embankment = self._fit_label_border(
            station, offset, embankment_mask, side=-1
        )
        right_embankment = self._fit_label_border(
            station, offset, embankment_mask, side=1
        )
        left_ditch = self._fit_label_border(station, offset, ditch_mask, side=-1)
        right_ditch = self._fit_label_border(station, offset, ditch_mask, side=1)

        if left_embankment is not None and right_embankment is not None:
            left_border = left_embankment(station)
            right_border = right_embankment(station)
            inside_embankment = (offset >= left_border) & (offset <= right_border)
            ground_labels[inside_embankment] = self.embankment_label

        for side, embankment_border, ditch_border in (
            (-1, left_embankment, left_ditch),
            (1, right_embankment, right_ditch),
        ):
            if embankment_border is None or ditch_border is None:
                continue

            embankment_offset = embankment_border(station)
            ditch_offset = ditch_border(station)
            side_mask = offset * side > 0
            between_borders = (
                side_mask
                & (offset * side > embankment_offset * side)
                & (offset * side <= ditch_offset * side)
            )
            replace_with_ground = between_borders & (ground_labels != self.ditch_label)
            ground_labels[replace_with_ground] = self.ground_label

            outside_ditch = side_mask & (offset * side > ditch_offset * side)
            ground_labels[outside_ditch & (ground_labels == self.ditch_label)] = (
                self.ground_label
            )

        ground_labels[rail_mask] = self.embankment_label
        full_labels[ground_idx] = ground_labels
        full_labels[original_labels != self.ground_label] = original_labels[
            original_labels != self.ground_label
        ]

        return full_labels

    def segment(self, points: np.ndarray, labels: np.ndarray) -> np.ndarray:
        full_labels = labels.copy()

        ground_mask = full_labels == self.ground_label
        ground_idx = np.flatnonzero(ground_mask)

        if ground_idx.size == 0:
            return full_labels

        ground_rail = points[ground_idx].copy()
        ground_rail_labels = full_labels[ground_idx].copy()

        rail_mask = self._label_rail_points(
            ground_rail,
            rail_radius=self.rail_radius,
        )

        if np.count_nonzero(rail_mask) == 0:
            return full_labels

        full_labels[ground_idx[rail_mask]] = self.embankment_label

        # Local normalization for numerical stability.
        ground_rail[:, :2] -= ground_rail[:, :2].mean(axis=0)
        ground_rail[:, 2] -= ground_rail[:, 2].min()
        ground_rail = ground_rail.astype(np.float32, copy=False)

        rail = ground_rail[rail_mask]
        centerline_xy = rail[:, :2]

        if centerline_xy.shape[0] == 0:
            return full_labels

        xy = ground_rail[:, :2]
        z = ground_rail[:, 2]

        centerline = self._build_centerline(centerline_xy)

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

            centered = self._center_chunk_x_on_rail(
                points_chunk_rotated=points_chunk_rotated,
                rail_mask=rail_mask_chunk,
            )

            if not centered:
                continue

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

            if points_nearest.shape[0] == 0:
                continue

            xz = points_nearest[~rail_mask_nearest][:, [0, 2]]

            if xz.shape[0] == 0:
                continue

            graph = self._build_xz_graph(xz)

            if graph.shape[0] < 2:
                continue

            left_graph, right_graph = self._split_graph_into_sides(graph)

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
                    min_uphill_points=self.graph_min_uphill_points,
                    min_embankment_points=self.graph_min_embankment_points,
                    noise_points=self.graph_noise_points,
                    smooth_window=self.graph_smooth_window,
                    ditch_min_downhill_points=self.graph_ditch_min_downhill_points,
                    ditch_min_uphill_points=self.graph_ditch_min_uphill_points,
                    ditch_immediate_points=self.graph_ditch_immediate_points,
                    ditch_max_flat_points=self.graph_ditch_max_flat_points,
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
                    min_uphill_points=self.graph_min_uphill_points,
                    min_embankment_points=self.graph_min_embankment_points,
                    noise_points=self.graph_noise_points,
                    smooth_window=self.graph_smooth_window,
                    ditch_min_downhill_points=self.graph_ditch_min_downhill_points,
                    ditch_min_uphill_points=self.graph_ditch_min_uphill_points,
                    ditch_immediate_points=self.graph_ditch_immediate_points,
                    ditch_max_flat_points=self.graph_ditch_max_flat_points,
                )

            plot_xz_side_sections(
                xz=xz,
                left_emb=left_emb,
                left_ditch=left_ditch,
                left_rest=left_rest,
                right_emb=right_emb,
                right_ditch=right_ditch,
                right_rest=right_rest,
            )

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

        return self._smooth_labels(
            ground_xy=xy,
            original_labels=labels,
            full_labels=full_labels,
            ground_idx=ground_idx,
            rail_mask=rail_mask,
            centerline=centerline,
            center_s=center_s,
        )

if __name__ == "__main__":
    import laspy
    from utils.plot_cloud import plot_cloud

    las_file = laspy.read(
        "/Users/michalsiniarski/Documents/DATA/BRIK/GRAJEWO-TEST/14-32_mini_mod.laz"
    )

    points = np.vstack((las_file.x, las_file.y, las_file.z)).T
    labels = np.asarray(las_file.classification)

    cfg_path = pth.Path(__file__).parent / "ground_segm_config.json"
    db_param_path = pth.Path(__file__).parent / "db_params.txt"

    cutter = GroundSegmenter.from_config(
        cfg_path=cfg_path,
        db_param_path=db_param_path,
        verbose=False,
    )

    labels_sectioned = cutter.segment(points, labels)
