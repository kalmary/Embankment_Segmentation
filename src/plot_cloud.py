import numpy as np
import laspy
import pathlib as pth
from utils.plot_cloud import plot_cloud

def main():
    folder = pth.Path("/home/jakub-szota/Pobrane/test_nowy")

    files = sorted(folder.glob("*.las")) + sorted(folder.glob("*.laz"))

    if not files:
        print(f"Brak plików .las/.laz w {folder}")
        return

    for i, path in enumerate(files):
        print(f"[{i+1}/{len(files)}] {path.name}")

        las = laspy.read(path)
        xyz = np.stack([
            np.asarray(las.x, dtype=np.float32),
            np.asarray(las.y, dtype=np.float32),
            np.asarray(las.z, dtype=np.float32),
        ], axis=1)
        labels = np.asarray(las.classification, dtype=np.int32)
        del las

        xyz[:, :2] -= xyz[:, :2].mean(axis=0)
        xyz[:, 2]  -= xyz[:, 2].min()

        plot_cloud(xyz, labels)


if __name__ == "__main__":
    main()