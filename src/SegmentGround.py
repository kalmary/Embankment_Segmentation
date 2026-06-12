from __future__ import annotations

import numpy as np

from scipy.spatial import cKDTree
from scipy.interpolate import UnivariateSpline


def plot_xz_graph(xz: np.ndarray, graph: np.ndarray, point_size: float = 1.0):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.scatter(
        xz[:, 0],
        xz[:, 1],
        s=point_size,
        alpha=0.25,
        label="XZ points",
    )

    ax.plot(
        graph[:, 0],
        graph[:, 1],
        linewidth=2,
        marker=".",
        color="red",
        label="mean Z graph",
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("XZ section and graph")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    plt.show()


def plot_xz_side_graphs(
    xz: np.ndarray,
    left_graph: np.ndarray | None,
    right_graph: np.ndarray | None,
    point_size: float = 1.0,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.scatter(
        xz[:, 0],
        xz[:, 1],
        s=point_size,
        alpha=0.25,
        label="XZ points",
    )

    if left_graph is not None and len(left_graph):
        ax.plot(
            left_graph[:, 0],
            left_graph[:, 1],
            linewidth=2,
            marker=".",
            label="left mean Z",
        )

    if right_graph is not None and len(right_graph):
        ax.plot(
            right_graph[:, 0],
            right_graph[:, 1],
            linewidth=2,
            marker=".",
            label="right mean Z",
        )

    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("XZ section split into left/right graphs")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    plt.show()


def plot_xz_side_sections(
    xz: np.ndarray,
    left_1: np.ndarray | None,
    left_2: np.ndarray | None,
    right_1: np.ndarray | None,
    right_2: np.ndarray | None,
    point_size: float = 1.0,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.scatter(
        xz[:, 0],
        xz[:, 1],
        s=point_size,
        alpha=0.20,
        label="XZ points",
    )

    if left_1 is not None and len(left_1):
        ax.plot(
            left_1[:, 0],
            left_1[:, 1],
            linewidth=3,
            marker=".",
            label="left section 1",
        )

    if left_2 is not None and len(left_2):
        ax.plot(
            left_2[:, 0],
            left_2[:, 1],
            linewidth=3,
            marker=".",
            label="left section 2",
        )

    if right_1 is not None and len(right_1):
        ax.plot(
            right_1[:, 0],
            right_1[:, 1],
            linewidth=3,
            marker=".",
            label="right section 1",
        )

    if right_2 is not None and len(right_2):
        ax.plot(
            right_2[:, 0],
            right_2[:, 1],
            linewidth=3,
            marker=".",
            label="right section 2",
        )

    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("XZ side graphs split by uphill / flattening")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    plt.show()


class CurvedCutter:
    def __init__(
        self,
        distance_limit: float = 10.0,
        ground_label: int = 1,
        rail_label: int = 0,
        embankment_label: int = 10,
        ditch_label: int = 20,
        length_min: float = 1.0,
        length_max: float = 30.0,
        width_margin: float = 0.0,
        max_curve_ratio: float = 1.08,
        curve_resolution: float = 1.0,
        graph_x_bin: float = 0.25,
        graph_uphill_slope: float = 0.10,
        graph_min_uphill_points: int = 3,
        graph_noise_points: int = 2,
        graph_smooth_window: int = 3,
        graph_max_gap_bins: float = 3.0,
    ):
        self.distance_limit = float(distance_limit)
        self.ground_label = int(ground_label)
        self.rail_label = int(rail_label)
        self.embankment_label = int(embankment_label)
        self.ditch_label = int(ditch_label)

        self.length_min = float(length_min)
        self.length_max = float(length_max)
        self.length = self.length_max

        self.width_margin = float(width_margin)

        self.max_curve_ratio = float(max_curve_ratio)
        self.curve_resolution = float(curve_resolution)

        # Centerline voxelization follows curvature-check resolution.
        self.voxel = self.curve_resolution

        self.graph_x_bin = float(graph_x_bin)
        self.graph_uphill_slope = float(graph_uphill_slope)
        self.graph_min_uphill_points = int(graph_min_uphill_points)
        self.graph_noise_points = int(graph_noise_points)
        self.graph_smooth_window = int(graph_smooth_window)
        self.graph_max_gap_bins = float(graph_max_gap_bins)

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

        x_mid = 0.5 * (x.min() + x.max())
        y_min = y.min()

        out = pcd[idx].copy()
        out[:, 0] = x - x_mid
        out[:, 1] = y - y_min
        out[:, 2] = z[idx]

        if self.width_margin:
            keep = np.abs(out[:, 0]) <= 0.5 * (
                x.max() - x.min() + 2.0 * self.width_margin
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

            ratio = self._curve_ratio_between(a, b, centerline, center_s)

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

    def _build_centerline(self, xy: np.ndarray):
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

        # Bigger = stiffer. Rail/embankment should not bend aggressively.
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
    ):
        tree = cKDTree(centerline)
        _, nearest = tree.query(xy)
        return center_s[nearest]

    @staticmethod
    def _point_at_s(
        s: float,
        centerline: np.ndarray,
        center_s: np.ndarray,
    ):
        i = np.searchsorted(center_s, s, side="right") - 1
        i = np.clip(i, 0, len(centerline) - 2)

        s0 = center_s[i]
        s1 = center_s[i + 1]

        t = 0.0 if s1 == s0 else (s - s0) / (s1 - s0)

        return (1.0 - t) * centerline[i] + t * centerline[i + 1]

    @staticmethod
    def _arc_length(line: np.ndarray):
        d = np.diff(line, axis=0)
        ds = np.linalg.norm(d, axis=1)
        return np.r_[0.0, np.cumsum(ds)]

    @staticmethod
    def _unit(v: np.ndarray):
        return v / np.linalg.norm(v)

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
        return self._curve_ratio_between(s, s1, centerline, center_s)

    @staticmethod
    def get_graph(xz: np.ndarray, x_bin: float) -> np.ndarray:
        x = xz[:, 0]
        z = xz[:, 1]

        x0 = x.min()
        bins = np.floor((x - x0) / x_bin).astype(np.int64)

        order = np.argsort(bins)
        bins = bins[order]
        z = z[order]

        unique_bins, start = np.unique(bins, return_index=True)

        z_sum = np.add.reduceat(z, start)
        count = np.diff(np.r_[start, len(z)])

        x_centers = x0 + (unique_bins + 0.5) * x_bin
        mean_z = z_sum / count

        return np.column_stack((x_centers, mean_z))

    @staticmethod
    def _split_graph_by_x_gap(
        graph: np.ndarray,
        max_gap: float,
    ) -> list[np.ndarray]:
        if graph is None or len(graph) == 0:
            return []

        order = np.argsort(graph[:, 0])
        graph = graph[order]

        gaps = np.diff(graph[:, 0])
        split_after = np.flatnonzero(gaps > max_gap)

        parts = []
        start = 0

        for idx in split_after:
            end = idx + 1
            part = graph[start:end]

            if len(part):
                parts.append(part)

            start = end

        part = graph[start:]

        if len(part):
            parts.append(part)

        return parts

    @staticmethod
    def _pick_two_side_components(
        graph_parts: list[np.ndarray],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if len(graph_parts) == 0:
            return None, None

        if len(graph_parts) == 1:
            part = graph_parts[0]

            if np.mean(part[:, 0]) < 0:
                return part, None

            return None, part

        # Pick two largest continuous components.
        parts = sorted(graph_parts, key=len, reverse=True)[:2]

        # Assign by actual X position, not by sign around zero.
        parts = sorted(parts, key=lambda g: np.mean(g[:, 0]))

        left_graph = parts[0]
        right_graph = parts[1]

        return left_graph, right_graph

    @staticmethod
    def get_side_graphs(
        xz: np.ndarray,
        x_bin: float,
        max_gap_bins: float = 3.0,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        graph = CurvedCutter.get_graph(
            xz=xz,
            x_bin=x_bin,
        )

        graph_parts = CurvedCutter._split_graph_by_x_gap(
            graph=graph,
            max_gap=x_bin * max_gap_bins,
        )

        left_graph, right_graph = CurvedCutter._pick_two_side_components(
            graph_parts=graph_parts,
        )

        return left_graph, right_graph

    @staticmethod
    def split_graph_by_gradient(
        graph: np.ndarray,
        side: str,
        uphill_slope: float,
        min_uphill_points: int,
        noise_points: int,
        smooth_window: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Split one already-separated continuous side graph into:
            section_1: from graph start to the end of the first meaningful uphill part
            section_2: the rest

        If slope is not above uphill_slope, it is treated as flat/downhill.

        side:
            "left"  -> outward direction is -X
            "right" -> outward direction is +X
        """
        empty = np.empty((0, 2), dtype=np.float64)

        if graph is None or len(graph) < 2:
            return empty, empty

        x = graph[:, 0].astype(np.float64)
        z = graph[:, 1].astype(np.float64)

        if side == "left":
            outward = -x
        elif side == "right":
            outward = x
        else:
            raise ValueError(f"side must be 'left' or 'right', got {side}")

        order = np.argsort(outward)
        x = x[order]
        z = z[order]
        outward = outward[order]

        keep = np.r_[True, np.diff(outward) > 1e-9]
        x = x[keep]
        z = z[keep]
        outward = outward[keep]

        full_graph = np.column_stack((x, z))

        if len(outward) < 2:
            return empty, full_graph

        z_for_gradient = z.copy()

        if smooth_window >= 3 and len(z_for_gradient) >= smooth_window:
            if smooth_window % 2 == 0:
                smooth_window += 1

            pad = smooth_window // 2
            z_padded = np.pad(z_for_gradient, pad_width=pad, mode="edge")
            kernel = np.ones(smooth_window, dtype=np.float64) / smooth_window
            z_for_gradient = np.convolve(z_padded, kernel, mode="valid")

        dz_dout = np.gradient(z_for_gradient, outward)

        uphill_mask = dz_dout > uphill_slope

        run_start = None
        run_end = None

        i = 0

        while i < len(uphill_mask):
            if not uphill_mask[i]:
                i += 1
                continue

            candidate_start = i

            while i < len(uphill_mask) and uphill_mask[i]:
                i += 1

            candidate_end = i
            candidate_len = candidate_end - candidate_start

            if candidate_len < min_uphill_points:
                continue

            if candidate_end <= noise_points:
                continue

            run_start = candidate_start
            run_end = candidate_end
            break

        if run_start is None:
            return empty, full_graph

        split_idx = run_end

        section_1 = full_graph[:split_idx]
        section_2 = full_graph[split_idx:]

        return section_1, section_2

    @staticmethod
    def split_side_graphs_by_gradient(
        left_graph: np.ndarray | None,
        right_graph: np.ndarray | None,
        uphill_slope: float,
        min_uphill_points: int,
        noise_points: int,
        smooth_window: int,
    ):
        left_1, left_2 = None, None
        right_1, right_2 = None, None

        if left_graph is not None and len(left_graph):
            left_1, left_2 = CurvedCutter.split_graph_by_gradient(
                graph=left_graph,
                side="left",
                uphill_slope=uphill_slope,
                min_uphill_points=min_uphill_points,
                noise_points=noise_points,
                smooth_window=smooth_window,
            )

        if right_graph is not None and len(right_graph):
            right_1, right_2 = CurvedCutter.split_graph_by_gradient(
                graph=right_graph,
                side="right",
                uphill_slope=uphill_slope,
                min_uphill_points=min_uphill_points,
                noise_points=noise_points,
                smooth_window=smooth_window,
            )

        return left_1, left_2, right_1, right_2

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

        mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max)

        return mask

    def cast_sections_to_labels(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        left_1: np.ndarray | None,
        left_2: np.ndarray | None,
        right_1: np.ndarray | None,
        right_2: np.ndarray | None,
    ) -> np.ndarray:
        labels_out = labels.copy()

        x_padding = 0.5 * self.graph_x_bin

        left_1_mask = self._mask_points_by_graph_section(
            points=points,
            section=left_1,
            x_padding=x_padding,
        )
        right_1_mask = self._mask_points_by_graph_section(
            points=points,
            section=right_1,
            x_padding=x_padding,
        )

        left_2_mask = self._mask_points_by_graph_section(
            points=points,
            section=left_2,
            x_padding=x_padding,
        )
        right_2_mask = self._mask_points_by_graph_section(
            points=points,
            section=right_2,
            x_padding=x_padding,
        )

        section_2_mask = left_2_mask | right_2_mask
        section_1_mask = left_1_mask | right_1_mask

        labels_out[section_2_mask] = self.ground_label
        labels_out[section_1_mask] = self.ditch_label

        return labels_out

    def _find_nearest_points(
        self,
        points: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        nearest_points_mask = np.zeros(points.shape[0], dtype=bool)

        embankment_mask = labels == self.embankment_label
        candidate_indices = np.flatnonzero(labels != self.rail_label)

        if candidate_indices.size == 0:
            return nearest_points_mask

        embankment_points = points[embankment_mask]

        if embankment_points.shape[0] == 0:
            return nearest_points_mask

        tree = cKDTree(
            embankment_points[:, :2],
            copy_data=False,
        )

        distances, _ = tree.query(
            points[candidate_indices, :2],
            k=1,
            distance_upper_bound=self.distance_limit,
            workers=-1,
        )

        nearest_points_mask[candidate_indices] = distances < self.distance_limit

        return nearest_points_mask

    def iter_rectangles(
        self,
        pcd: np.ndarray,
        labels: np.ndarray,
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

    def segment(self, points: np.ndarray, labels: np.ndarray) -> np.ndarray:
        labels_out = labels.copy()

        points[:, :2] = points[:, :2] - np.mean(points[:, :2], axis=0)
        points[:, 2] = points[:, 2] - np.min(points[:, 2], axis=0)
        points = points.astype(np.float32)

        mask_embankment = labels == self.embankment_label

        embankment_points = points[mask_embankment]
        centerline_xy = embankment_points[:, :2]

        del embankment_points, mask_embankment

        mask_ground_embankment = (
            (labels == self.embankment_label)
            | (labels == self.rail_label)
            | (labels == self.ground_label)
        )

        ground_embankment_indices = np.flatnonzero(mask_ground_embankment)
        ground_embankment_points = points[mask_ground_embankment]
        ground_embankment_labels = labels[mask_ground_embankment]

        xy = ground_embankment_points[:, :2]
        z = ground_embankment_points[:, 2]

        centerline = self._build_centerline(centerline_xy)

        center_s = self._arc_length(centerline)
        point_s = self._assign_points_to_centerline(
            xy=xy,
            centerline=centerline,
            center_s=center_s,
        )

        for points_chunk_rotated, indices in self.iter_rectangles(
            pcd=ground_embankment_points,
            labels=ground_embankment_labels,
            xy=xy,
            z=z,
            centerline=centerline,
            center_s=center_s,
            point_s=point_s,
        ):
            nearest_points_mask = self._find_nearest_points(
                points_chunk_rotated,
                ground_embankment_labels[indices],
            )

            points_chunk_rotated = points_chunk_rotated[nearest_points_mask]
            labels_nearest = ground_embankment_labels[indices][nearest_points_mask]
            section_indices = indices[nearest_points_mask]

            no_embankment_mask = labels_nearest != self.embankment_label
            points_chunk_rotated = points_chunk_rotated[no_embankment_mask]
            labels_nearest = labels_nearest[no_embankment_mask]
            section_indices = section_indices[no_embankment_mask]

            if len(points_chunk_rotated) == 0:
                continue

            xz = points_chunk_rotated[:, [0, 2]]

            left_graph, right_graph = self.get_side_graphs(
                xz=xz,
                x_bin=self.graph_x_bin,
                max_gap_bins=self.graph_max_gap_bins,
            )

            left_1, left_2, right_1, right_2 = self.split_side_graphs_by_gradient(
                left_graph=left_graph,
                right_graph=right_graph,
                uphill_slope=self.graph_uphill_slope,
                min_uphill_points=self.graph_min_uphill_points,
                noise_points=self.graph_noise_points,
                smooth_window=self.graph_smooth_window,
            )

            labels_sectioned = self.cast_sections_to_labels(
                points=points_chunk_rotated,
                labels=labels_nearest,
                left_1=left_1,
                left_2=left_2,
                right_1=right_1,
                right_2=right_2,
            )

            # Debug plotting if needed:
            # plot_xz_side_sections(
            #     xz=xz,
            #     left_1=left_1,
            #     left_2=left_2,
            #     right_1=right_1,
            #     right_2=right_2,
            # )

            # from utils.plot_cloud import plot_cloud
            # plot_cloud(points_chunk_rotated, labels_sectioned)

            labels_out[ground_embankment_indices[section_indices]] = labels_sectioned

        return labels_out


if __name__ == "__main__":
    import laspy
    from utils.plot_cloud import plot_cloud

    las_file = laspy.read(
        "/Users/michalsiniarski/Documents/DATA/BRIK/GRAJEWO-TEST/14-32_mini_mod.laz"
    )

    points = np.vstack((las_file.x, las_file.y, las_file.z)).T
    labels = np.asarray(las_file.classification)

    cutter = CurvedCutter(
        distance_limit=8.0,
        ground_label=1,
        rail_label=0,
        embankment_label=10,
        ditch_label=11,
        length_min=0.5,
        length_max=10.,
        width_margin=0.0,
        max_curve_ratio=1.03,
        curve_resolution=0.25,
        graph_x_bin=0.25,
        graph_uphill_slope=0.2,
        graph_min_uphill_points=3,
        graph_noise_points=2,
        graph_smooth_window=3,
        graph_max_gap_bins=3.0,
    )

    labels_sectioned = cutter.segment(points, labels)
    plot_cloud(points, labels_sectioned)