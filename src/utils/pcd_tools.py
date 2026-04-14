import open3d as o3d
import numpy as np

def remove_outliers(points: np.ndarray, nb_neighbors:int=40, std_ratio:float=2.):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    mask = np.zeros(points.shape[0], dtype=bool)
    mask[ind] = True
    return mask

 
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
 
    mask = np.zeros(xyz.shape[0], dtype=bool)
    mask[chosen] = True
    return mask