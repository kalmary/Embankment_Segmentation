import json
import pathlib as pth
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GroundConfig:
    distance_limit: float
    ground_label: int
    rail_label: int
    rail_radius: float
    embankment_label: int
    ditch_label: int
    length_min: float
    length_max: float
    width_margin: float
    max_curve_ratio: float
    curve_resolution: float
    graph_x_bin: float
    graph_uphill_slope: float
    graph_embankment_min_stop_points: int
    graph_min_embankment_points: int
    graph_noise_points: int
    graph_smooth_window: int
    graph_max_gap_bins: float
    graph_ditch_min_downhill_points: int
    graph_ditch_min_uphill_points: int
    graph_ditch_immediate_points: int
    graph_ditch_max_flat_points: int
    graph_ditch_max_uphill_points: int
    graph_ditch_search_min_m: float
    graph_ditch_search_max_m: float
    smooth: bool
    smooth_level: float

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]):
        distance_limit = float(config["distance_limit"])
        graph_x_bin = float(config["graph_x_bin"])

        if "graph_embankment_min_stop_m" in config:
            graph_embankment_min_stop_m = float(
                config["graph_embankment_min_stop_m"]
            )
        elif "graph_min_uphill_m" in config:
            graph_embankment_min_stop_m = float(config["graph_min_uphill_m"])
        else:
            graph_embankment_min_stop_m = cls._read_graph_distance_m(
                config,
                graph_x_bin,
                meter_key="graph_embankment_min_stop_m",
                legacy_points_key="graph_min_uphill_points",
            )

        graph_embankment_min_stop_points = cls._graph_meters_to_points(
            graph_embankment_min_stop_m,
            graph_x_bin,
            minimum_points=1,
        )
        graph_min_embankment_m = cls._read_graph_distance_m(
            config,
            graph_x_bin,
            meter_key="graph_min_embankment_m",
            legacy_points_key="graph_min_embankment_points",
            default_m=graph_embankment_min_stop_m,
        )
        graph_min_embankment_points = cls._graph_meters_to_points(
            graph_min_embankment_m,
            graph_x_bin,
            minimum_points=1,
        )

        graph_noise_points = int(config["graph_noise_points"])
        graph_ditch_min_downhill_m = cls._read_graph_distance_m(
            config,
            graph_x_bin,
            meter_key="graph_ditch_min_downhill_m",
            legacy_points_key="graph_ditch_min_downhill_points",
            default_m=graph_embankment_min_stop_m,
        )
        graph_ditch_min_downhill_points = cls._graph_meters_to_points(
            graph_ditch_min_downhill_m,
            graph_x_bin,
            minimum_points=1,
        )
        graph_ditch_min_uphill_m = cls._read_graph_distance_m(
            config,
            graph_x_bin,
            meter_key="graph_ditch_min_uphill_m",
            legacy_points_key="graph_ditch_min_uphill_points",
            default_m=graph_embankment_min_stop_m,
        )
        graph_ditch_min_uphill_points = cls._graph_meters_to_points(
            graph_ditch_min_uphill_m,
            graph_x_bin,
            minimum_points=1,
        )
        graph_ditch_immediate_m = cls._read_graph_distance_m(
            config,
            graph_x_bin,
            meter_key="graph_ditch_immediate_points_m",
            legacy_points_key="graph_ditch_immediate_points",
            default_m=(graph_noise_points + 1) * graph_x_bin,
        )
        graph_ditch_immediate_points = cls._graph_meters_to_points(
            graph_ditch_immediate_m,
            graph_x_bin,
            minimum_points=0,
        )
        graph_ditch_max_flat_m = cls._read_graph_distance_m(
            config,
            graph_x_bin,
            meter_key="graph_ditch_max_flat_m",
            legacy_points_key="graph_ditch_max_flat_points",
            default_m=(graph_noise_points + 2) * graph_x_bin,
        )
        graph_ditch_max_flat_points = cls._graph_meters_to_points(
            graph_ditch_max_flat_m,
            graph_x_bin,
            minimum_points=0,
        )
        graph_ditch_max_uphill_m = cls._read_graph_distance_m(
            config,
            graph_x_bin,
            meter_key="graph_ditch_max_uphill_m",
            legacy_points_key="graph_ditch_max_uphill_points",
            default_m=distance_limit,
        )
        graph_ditch_max_uphill_points = cls._graph_meters_to_points(
            graph_ditch_max_uphill_m,
            graph_x_bin,
            minimum_points=graph_ditch_min_uphill_points,
        )

        graph_ditch_search_min_m = float(
            config.get("graph_ditch_search_min_m", 0.0)
        )
        graph_ditch_search_max_m = float(
            config.get("graph_ditch_search_max_m", distance_limit)
        )
        if graph_ditch_search_min_m < 0.0:
            raise ValueError("graph_ditch_search_min_m must be non-negative.")
        if graph_ditch_search_max_m < graph_ditch_search_min_m:
            raise ValueError(
                "graph_ditch_search_max_m must be >= graph_ditch_search_min_m."
            )

        return cls(
            distance_limit=distance_limit,
            ground_label=int(config["ground_label"]),
            rail_label=int(config["rail_label"]),
            rail_radius=float(config["rail_radius"]),
            embankment_label=int(config["embankment_label"]),
            ditch_label=int(config["ditch_label"]),
            length_min=float(config["length_min"]),
            length_max=float(config["length_max"]),
            width_margin=float(config["width_margin"]),
            max_curve_ratio=float(config["max_curve_ratio"]),
            curve_resolution=float(config["curve_resolution"]),
            graph_x_bin=graph_x_bin,
            graph_uphill_slope=float(config["graph_uphill_slope"]),
            graph_embankment_min_stop_points=graph_embankment_min_stop_points,
            graph_min_embankment_points=graph_min_embankment_points,
            graph_noise_points=graph_noise_points,
            graph_smooth_window=int(config["graph_smooth_window"]),
            graph_max_gap_bins=float(config["graph_max_gap_bins"]),
            graph_ditch_min_downhill_points=graph_ditch_min_downhill_points,
            graph_ditch_min_uphill_points=graph_ditch_min_uphill_points,
            graph_ditch_immediate_points=graph_ditch_immediate_points,
            graph_ditch_max_flat_points=graph_ditch_max_flat_points,
            graph_ditch_max_uphill_points=graph_ditch_max_uphill_points,
            graph_ditch_search_min_m=graph_ditch_search_min_m,
            graph_ditch_search_max_m=graph_ditch_search_max_m,
            smooth=bool(config.get("smooth", True)),
            smooth_level=float(config.get("smooth_level", 10.0)),
        )

    @staticmethod
    def _read_graph_distance_m(
        config: Mapping[str, Any],
        graph_x_bin: float,
        meter_key: str,
        legacy_points_key: str,
        default_m: float | None = None,
    ) -> float:
        if meter_key in config:
            return float(config[meter_key])
        if legacy_points_key in config:
            return float(config[legacy_points_key]) * graph_x_bin
        if default_m is not None:
            return float(default_m)
        raise KeyError(meter_key)

    @staticmethod
    def _graph_meters_to_points(
        value_m: float,
        graph_x_bin: float,
        minimum_points: int,
    ) -> int:
        if graph_x_bin <= 0.0:
            raise ValueError("graph_x_bin must be positive.")
        if value_m < 0.0:
            raise ValueError("Graph distance thresholds must be non-negative.")
        return max(minimum_points, int(np.ceil(value_m / graph_x_bin)))


def test_ground_config_parses_documented_values_and_derived_thresholds():
    config_path = pth.Path(__file__).with_name("ground_segm_config.json")
    config = json.loads(config_path.read_text())

    parsed = GroundConfig.from_mapping(config)

    assert parsed.distance_limit == 25.0
    assert parsed.ground_label == 1
    assert parsed.rail_label == 0
    assert parsed.graph_x_bin == 0.25
    assert parsed.graph_embankment_min_stop_points == 3
    assert parsed.graph_min_embankment_points == 7
    assert parsed.graph_ditch_min_downhill_points == 2
    assert parsed.graph_ditch_min_uphill_points == 2
    assert parsed.graph_ditch_immediate_points == 3
    assert parsed.graph_ditch_max_flat_points == 4
    assert parsed.graph_ditch_max_uphill_points == 8
    assert parsed.graph_ditch_search_min_m == 6.0
    assert parsed.graph_ditch_search_max_m == 16.0
    assert parsed.smooth is True
    assert parsed.smooth_level == 20.0
