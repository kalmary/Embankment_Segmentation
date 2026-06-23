import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv


def plot_xyz_cloud(
    points: np.ndarray,
    labels: np.ndarray | None = None,
    point_size: float = 1.0,
):
    points = np.asarray(points)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")

    if labels is not None and len(labels) != len(points):
        raise ValueError(
            f"labels must have the same length as points: {len(labels)} != {len(points)}"
        )

    cloud = pv.PolyData(points)
    plotter = pv.Plotter(notebook=False, off_screen=False)

    if labels is None:
        plotter.add_mesh(cloud, color="green", point_size=point_size)
    else:
        cloud["label"] = labels
        plotter.add_mesh(
            cloud,
            scalars="label",
            cmap="tab20",
            point_size=point_size,
        )

    plotter.add_axes(xlabel="X", ylabel="Y", zlabel="Z")
    plotter.reset_camera()
    plotter.show()


def plot_centerline(centerline: np.ndarray):
    centerline = np.asarray(centerline)

    if centerline.ndim != 2 or centerline.shape[1] != 2:
        raise ValueError(
            f"centerline must have shape (N, 2), got {centerline.shape}"
        )

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(centerline[:, 0], centerline[:, 1], linewidth=2, label="centerline")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Centerline")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    plt.show()


def plot_xz_graph_split_by_rail(
    xz: np.ndarray,
    graph: np.ndarray,
    graph_labels: np.ndarray,
    left_graph: np.ndarray | None,
    right_graph: np.ndarray | None,
    rail_label: int = 0,
    point_size: float = 1.0,
):
    if len(graph_labels) != len(graph):
        raise ValueError(
            f"graph_labels must have the same length as graph: "
            f"{len(graph_labels)} != {len(graph)}"
        )

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.scatter(
        xz[:, 0],
        xz[:, 1],
        s=point_size,
        alpha=0.15,
        color="lightblue",
        label="XZ points",
    )

    ax.plot(
        graph[:, 0],
        graph[:, 1],
        linewidth=1.5,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label="full graph",
    )

    rail_mask = graph_labels == rail_label
    ax.scatter(
        graph[rail_mask, 0],
        graph[rail_mask, 1],
        s=30,
        color="black",
        marker="x",
        label="rail bins",
        zorder=10,
    )

    if left_graph is not None and len(left_graph):
        ax.plot(
            left_graph[:, 0],
            left_graph[:, 1],
            linewidth=3,
            color="blue",
            label="left side",
            zorder=10,
        )

    if right_graph is not None and len(right_graph):
        ax.plot(
            right_graph[:, 0],
            right_graph[:, 1],
            linewidth=3,
            color="red",
            label="right side",
            zorder=10,
        )

    ax.axvline(0.0, linewidth=1, color="black", linestyle=":")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("XZ graph split by rail bins")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.show()


def plot_xz_graph(xz: np.ndarray, graph: np.ndarray, point_size: float = 1.0):

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
    left_emb: np.ndarray | None,
    left_ditch: np.ndarray | None,
    left_rest: np.ndarray | None,
    right_emb: np.ndarray | None,
    right_ditch: np.ndarray | None,
    right_rest: np.ndarray | None,
    point_size: float = 1.0,
    max_plot_gap: float | None = None,
):
    import numpy as np
    import matplotlib.pyplot as plt

    def _plot_section(
        ax,
        section: np.ndarray | None,
        color: str,
        label: str,
        linewidth: float = 3.0,
    ):
        if section is None or len(section) == 0:
            return

        section = np.asarray(section, dtype=np.float64)

        order = np.argsort(section[:, 0])
        section = section[order]

        keep = np.r_[True, np.diff(section[:, 0]) > 1e-9]
        section = section[keep]

        if len(section) == 0:
            return

        if max_plot_gap is None:
            # Robust fallback: detect visible breaks from local X spacing.
            dx = np.diff(section[:, 0])

            if len(dx) == 0:
                parts = [section]
            else:
                positive_dx = dx[dx > 1e-9]

                if len(positive_dx) == 0:
                    parts = [section]
                else:
                    typical_dx = np.median(positive_dx)
                    gap_limit = max(3.0 * typical_dx, 1e-6)
                    split_at = np.flatnonzero(dx > gap_limit) + 1
                    parts = [part for part in np.split(section, split_at) if len(part)]
        else:
            split_at = np.flatnonzero(np.diff(section[:, 0]) > max_plot_gap) + 1
            parts = [part for part in np.split(section, split_at) if len(part)]

        first = True

        for part in parts:
            ax.plot(
                part[:, 0],
                part[:, 1],
                linewidth=linewidth,
                marker=".",
                color=color,
                label=label if first else None,
            )
            first = False

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.scatter(
        xz[:, 0],
        xz[:, 1],
        s=point_size,
        alpha=0.20,
        color="blue",
        label="XZ points",
    )

    _plot_section(ax, left_emb, "red", "left embankment")
    _plot_section(ax, left_ditch, "orange", "left ditch")
    _plot_section(ax, left_rest, "green", "left rest")

    _plot_section(ax, right_emb, "magenta", "right embankment")
    _plot_section(ax, right_ditch, "gold", "right ditch")
    _plot_section(ax, right_rest, "lime", "right rest")

    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("XZ side graphs split by uphill / flattening")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    plt.show()
